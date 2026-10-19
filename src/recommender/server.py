"""The recommender as a web service for the signed-in users of the web app.

POST /recommend/chat  {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
    Needs `Authorization: Bearer <Cognito access token>`. The response is NDJSON: {"type": "wait"}
    lines every few seconds while the models work (they keep CloudFront's origin timeout from firing on
    a long answer), then one {"type": "answer", "answer": "..."} or {"type": "error", "error": "..."}.
GET /recommend/health
    No auth: whether the library is loaded, and its version.

Configuration comes from the environment (on EC2, /etc/recommender.env):
  RECOMMENDER_BUCKET, RECOMMENDER_PREFIX   where bundle.py published the library (pulled at start)
  RECOMMENDER_DATA_DIR                     where to keep it (default /var/lib/recommender)
  RECOMMENDER_CONFIG                       serve a local workspace instead, e.g. settings.toml
  COGNITO_ISSUER, COGNITO_CLIENT_ID        which tokens to accept
  COGNITO_JWKS_URL                         the issuer's keys, when not at <issuer>/.well-known/jwks.json
  RECOMMENDER_WORKERS                      questions answered at once (default 8)
  RECOMMENDER_CORS_ORIGINS                 comma-separated origins for a page on another origin
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import jwt
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# How long a question waits for a free pipeline before the user hears "busy".
QUEUE_SECONDS = 30
# Seconds between keepalive lines; well inside CloudFront's 60-second origin read timeout.
KEEPALIVE_SECONDS = 10
# Wait between attempts to load the library, e.g. before the first publish.
RETRY_SECONDS = 60


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


class Unauthorized(Exception):
    pass


@dataclass
class TokenVerifier:
    """Accept only access tokens that this user pool issued to this app client."""
    issuer: str
    client_id: str
    jwks_url: str = ""

    def __post_init__(self):
        self.jwks = jwt.PyJWKClient(self.jwks_url or self.issuer.rstrip("/") + "/.well-known/jwks.json",
                                    cache_keys=True, lifespan=3600)

    def __call__(self, authorization: str | None) -> dict:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Unauthorized("Missing bearer token")
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.issuer,
                                options={"require": ["exp", "iss", "sub"], "verify_aud": False})
        except (jwt.PyJWTError, requests.RequestException) as error:
            raise Unauthorized(str(error)) from error
        # Cognito access tokens carry the app client in client_id (ID tokens put it in aud instead).
        if claims.get("token_use") != "access" or claims.get("client_id") != self.client_id:
            raise Unauthorized("Not an access token for this app")
        return claims


class PipelinePool:
    """N pipelines over one loaded library; each answers one question at a time."""

    def __init__(self, settings, size: int):
        from .rag import RagPipeline
        first = RagPipeline(settings)
        if first.anime_tools() is None:
            raise RuntimeError(f"No AniList catalog under {settings.documents}.")
        # Load the embedding model and the passage index once, before cloning shares them.
        first.vector_search("warm up", [])
        self.idle = queue.Queue()
        self.idle.put(first)
        for _ in range(size - 1):
            self.idle.put(first.clone())
        self.size = size

    def acquire(self, timeout: float):
        try:
            return self.idle.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, pipeline) -> None:
        self.idle.put(pipeline)


def http_status(error: BaseException) -> int:
    """429 when Bedrock throttled any call behind this error, else 502."""
    while error is not None:
        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) in (429, 503):
            return 429
        error = error.__cause__ or error.__context__
    return 502


def load_library():
    """Settings of the library to serve: a pulled bundle, or a local workspace."""
    from .config import load_settings
    if config := os.environ.get("RECOMMENDER_CONFIG"):
        return load_settings(config, "anilist"), "local"
    from .bundle import MANIFEST, pull
    bucket = os.environ["RECOMMENDER_BUCKET"]
    release = pull(bucket, Path(os.environ.get("RECOMMENDER_DATA_DIR", "/var/lib/recommender")),
                   os.environ.get("RECOMMENDER_PREFIX", "recommender"))
    version = json.loads((release / MANIFEST).read_text(encoding="utf-8"))["version"]
    return load_settings(release / "settings.toml", "anilist"), version


def create_app(verify=None, pool_factory=None) -> FastAPI:
    """The service; tests pass their own verifier and pool."""
    app = FastAPI(title="AniList recommender", docs_url=None, redoc_url=None)
    if origins := [o.strip() for o in os.environ.get("RECOMMENDER_CORS_ORIGINS", "").split(",") if o.strip()]:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"])
    verify = verify or TokenVerifier(os.environ["COGNITO_ISSUER"], os.environ["COGNITO_CLIENT_ID"],
                                     os.environ.get("COGNITO_JWKS_URL", ""))
    state = {"pool": None, "version": "", "error": ""}

    def load():
        # Retry until a library is published and loads; /health reports the last error meanwhile.
        while True:
            try:
                if pool_factory:
                    state["pool"], state["version"] = pool_factory()
                else:
                    settings, version = load_library()
                    state["pool"] = PipelinePool(settings, int(os.environ.get("RECOMMENDER_WORKERS", "8")))
                    state["version"] = version
                state["error"] = ""
                return
            except Exception as error:
                state["error"] = f"{type(error).__name__}: {error}"
                print(f"Recommender failed to load, retrying in {RETRY_SECONDS} s: {state['error']}", flush=True)
                time.sleep(RETRY_SECONDS)

    # Load in the background so the health check answers (not ready) while the library downloads.
    threading.Thread(target=load, daemon=True).start()

    @app.get("/recommend/health")
    def health():
        ready = state["pool"] is not None
        return JSONResponse({"ready": ready, "version": state["version"], "error": state["error"]},
                            status_code=200 if ready else 503)

    @app.post("/recommend/chat")
    async def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
        try:
            claims = verify(authorization)
        except Unauthorized:
            raise HTTPException(401, "Your session has ended. Sign in again.")
        pool = state["pool"]
        if pool is None:
            raise HTTPException(503, "The recommender is starting. Try again in a minute.")
        pipeline = await run_in_threadpool(pool.acquire, QUEUE_SECONDS)
        if pipeline is None:
            raise HTTPException(429, "The assistant is busy. Try again in a moment.")
        history = [turn.model_dump() for turn in request.history]
        result = {}
        done = threading.Event()

        def work():
            try:
                result["answer"] = pipeline.ask(request.message, history)
            except Exception as error:
                print(f"Recommender failed for user {claims['sub']}: {type(error).__name__}: {error}", flush=True)
                result["status"] = http_status(error)
            finally:
                pool.release(pipeline)
                done.set()

        threading.Thread(target=work, daemon=True).start()

        async def lines():
            while not await run_in_threadpool(done.wait, KEEPALIVE_SECONDS):
                yield json.dumps({"type": "wait"}) + "\n"
            if "answer" in result:
                yield json.dumps({"type": "answer", "answer": result["answer"]}) + "\n"
            else:
                message = ("The assistant is busy. Try again in a moment." if result["status"] == 429
                           else "The assistant couldn't answer. Try again.")
                yield json.dumps({"type": "error", "status": result["status"], "error": message}) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.exception_handler(HTTPException)
    async def error_body(_, error: HTTPException):
        # Same body as the chat Lambda, which web/app.js already reads.
        return JSONResponse({"error": error.detail}, status_code=error.status_code)

    return app


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, proxy_headers=True, log_level="info")
