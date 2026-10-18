# /// script
# requires-python = ">=3.12"
# dependencies = ["typer>=0.27"]
# ///
"""Drive the sign-in and chat flow against the stack deployed on LocalStack (`just apply`).

LocalStack's Cognito has no passwordless sign-in (SignUp without a password and the EMAIL_OTP
challenge are not emulated), so locally every user gets LOCAL_PASSWORD and signs in with
USER_AUTH + PASSWORD. Everything else is the real thing: the pre sign-up Lambda gates the
domain, the confirmation code goes through SES, the API checks the Cognito token and queues the
question, and the recommender worker answers it from the published catalog bundle through Bedrock,
which LocalStack runs with a small Ollama model on CPU. A question is asynchronous: POST /api/chat
returns a job, and GET /api/chat/{job_id} returns its answer once the worker is done.

Only the standard library plus Typer for the command line: Cognito's sign-up and sign-in calls
need no AWS signature, so this talks to the same JSON API the page does.

Run it with `uv run --script scripts/local_flow.py --help`, or through the `just` wrappers
(`just flow --help`, `just signup`, `just chat`, ...).
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated

import typer

# Asked about the fixture catalog that `just recommender-publish fixture` serves (tests/fixtures/).
FIXTURE_QUESTION = "Recommend anime like Clockwork Lighthouse."

ENDPOINT = os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566").rstrip("/")
STATE_DIR = Path(__file__).resolve().parent.parent / "live" / ".state" / "local"
LOCAL_PASSWORD = "Local-only-Passw0rd!"  # never used outside LocalStack


# ---------- Deployed stack ----------


def outputs(unit: str) -> dict:
    path = STATE_DIR / unit / "terraform.tfstate"
    if not path.exists():
        sys.exit(f"No state for {unit}: run `just apply` first.")
    return {k: v["value"] for k, v in json.loads(path.read_text())["outputs"].items()}


def gateway_url(url: str) -> str:
    """LocalStack serves every emulated host name on its gateway port."""
    port = ENDPOINT.rsplit(":", 1)[-1]
    url = url.replace("https://", "http://")
    return url if re.search(r"://[^/]+:\d+", url) else f"{url}:{port}"


def stack() -> dict:
    auth, api, web, recommender = outputs("auth"), outputs("chat-api"), outputs("web"), outputs("recommender")
    return {
        "page": gateway_url(web["url"]),
        "api": gateway_url(api["api_endpoint"]),
        "user_pool_id": auth["user_pool_id"],
        "client_id": auth["user_pool_client_id"],
        "ses_identity": auth["ses_identity"],
        "model_id": recommender["model_id"],
        "model_api": recommender["model_api"],
        "dead_letters": recommender["dead_letter_queue_url"],
    }


# ---------- HTTP ----------


def request(method: str, url: str, body=None, headers=None) -> tuple[int, dict, str]:
    data = body.encode() if isinstance(body, str) else body
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.status, dict(res.headers), res.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read().decode()


class CognitoError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def cognito(action: str, body: dict) -> dict:
    status, _, text = request(
        "POST",
        f"{ENDPOINT}/",
        json.dumps({"ClientId": stack()["client_id"], **body}),
        {
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": f"AWSCognitoIdentityProviderService.{action}",
        },
    )
    data = json.loads(text or "{}")
    if status != 200:
        code = str(data.get("__type", "Error")).split("#")[-1]
        raise CognitoError(code, data.get("message") or data.get("Message") or text)
    return data


# ---------- Flow ----------


def signed(service: str, content_type: str) -> dict:
    """Headers LocalStack routes to `service`. It reads the credential scope but checks no signature."""
    region = stack()["user_pool_id"].split("_")[0]
    return {
        "Content-Type": content_type,
        "Authorization": f"AWS4-HMAC-SHA256 Credential=test/20240101/{region}/{service}/aws4_request, "
        "SignedHeaders=host, Signature=0",
    }


def verify_sender():
    """Verify the stack's sending identity in LocalStack's SES, which LocalStack does at once.

    Cognito's code emails are rejected until the sender is verified, and live/local can't create
    the identity itself (LocalStack can't tag SESv2 identities). This is also the first SES call,
    which makes LocalStack load SES and with it the /_aws/ses inbox. The signature is not checked,
    only the service in the credential scope.
    """
    identity = stack()["ses_identity"]
    if "@" in identity:
        params = {"Action": "VerifyEmailIdentity", "EmailAddress": identity}
    else:
        params = {"Action": "VerifyDomainIdentity", "Domain": identity}
    status, _, text = request(
        "POST",
        f"{ENDPOINT}/",
        urllib.parse.urlencode({**params, "Version": "2010-12-01"}),
        signed("ses", "application/x-www-form-urlencoded"),
    )
    if status != 200:
        raise SystemExit(f"Cannot verify {identity} in LocalStack's SES: HTTP {status} {text[:200]}")


def codes(email: str | None = None) -> list[tuple[str, str]]:
    """(recipient, code) for every code email in LocalStack's SES inbox, oldest first."""
    status, _, text = request("GET", f"{ENDPOINT}/_aws/ses")
    if status != 200:
        verify_sender()
        status, _, text = request("GET", f"{ENDPOINT}/_aws/ses")
    if status != 200:
        raise SystemExit(f"Cannot read LocalStack's SES inbox ({ENDPOINT}/_aws/ses): HTTP {status} {text[:200]}")
    found = []
    for msg in json.loads(text or "{}").get("messages", []):
        body = (msg.get("Body") or {}).get("text_part") or ""
        match = re.search(r"\b(\d{6,8})\b", body)
        for to in msg.get("Destination", {}).get("ToAddresses", []):
            if match and (email is None or to.lower() == email.lower()):
                found.append((to, match.group(1)))
    return found


def latest_code(email: str, after: int = 0) -> str:
    """The newest code for email, waiting until there are more than `after` of them."""
    for _ in range(20):
        mine = codes(email)
        if len(mine) > after:
            return mine[-1][1]
        time.sleep(0.5)
    raise SystemExit(f"No code for {email} in LocalStack's SES inbox ({ENDPOINT}/_aws/ses).")


def signup(email: str) -> str:
    verify_sender()
    before = len(codes(email))
    cognito(
        "SignUp",
        {
            "Username": email,
            "Password": LOCAL_PASSWORD,
            "UserAttributes": [{"Name": "email", "Value": email}],
        },
    )
    code = latest_code(email, before)
    cognito("ConfirmSignUp", {"Username": email, "ConfirmationCode": code})
    return code


def sign_in(email: str) -> dict:
    res = cognito(
        "InitiateAuth",
        {
            "AuthFlow": "USER_AUTH",
            "AuthParameters": {
                "USERNAME": email,
                "PREFERRED_CHALLENGE": "PASSWORD",
                "PASSWORD": LOCAL_PASSWORD,
            },
        },
    )
    if "AuthenticationResult" not in res:
        raise SystemExit(f"Sign-in returned a challenge instead of tokens: {res.get('ChallengeName')}")
    return res["AuthenticationResult"]


def wait_for_model(timeout: int = 600):
    """Wait until LocalStack's Bedrock answers with the chat model.

    BEDROCK_PREWARM pulls and loads the model in the background after LocalStack reports healthy,
    so the first chat could otherwise outlast API Gateway's 30-second limit.
    """
    if stack()["model_api"] == "stub":
        return  # The worker's stand-in model needs no warm-up.
    if stack()["model_api"] == "ollama":
        return wait_for_ollama(stack()["model_id"], timeout)
    model_id = stack()["model_id"]
    url = f"{ENDPOINT}/model/{urllib.parse.quote(model_id, safe='')}/converse"
    body = json.dumps({"messages": [{"role": "user", "content": [{"text": "Say OK."}]}], "inferenceConfig": {"maxTokens": 5}})
    deadline = time.monotonic() + timeout
    while True:
        status, _, text = request("POST", url, body, signed("bedrock", "application/json"))
        if status == 200:
            return
        # A request or licence problem won't fix itself (Bedrock needs a LocalStack plan that includes it).
        if (400 <= status < 500 and status not in (408, 429)) or "licen" in text.lower():
            raise SystemExit(f"LocalStack's Bedrock refused {model_id}: HTTP {status} {text[:300]}")
        if time.monotonic() > deadline:
            raise SystemExit(f"LocalStack's Bedrock did not answer with {model_id} within {timeout}s: HTTP {status} {text[:300]}")
        time.sleep(5)


def wait_for_ollama(model: str, timeout: int):
    """Load the model in the Compose ollama service (`just up` pulled it), so the first answer is quick."""
    url = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11435")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say OK."}], "stream": False,
                       "options": {"num_predict": 5}})
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _, text = request("POST", f"{url}/api/chat", body, {"Content-Type": "application/json"})
        except OSError as err:  # not listening yet
            status, text = 0, str(err)
        if status == 200:
            return
        if status == 404:
            raise SystemExit(f"Ollama has no {model}: run `just up`, which pulls it ({text[:200]}).")
        if time.monotonic() > deadline:
            raise SystemExit(f"Ollama at {url} did not answer within {timeout}s: HTTP {status} {text[:200]}")
        time.sleep(3)


def api_call(method: str, url: str, token: str | None, body: dict | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, _, text = request(method, url, json.dumps(body) if body is not None else None, headers)
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, {"raw": text}


def chat(base: str, message: str, token: str | None, conversation_id: str | None = None) -> tuple[int, dict]:
    """Queue a question: 202 with the job, or the error."""
    body = {"message": message, **({"conversation_id": conversation_id} if conversation_id else {})}
    return api_call("POST", f"{base}/api/chat", token, body)


def answer_timeout() -> int:
    """Seconds to wait for an answer: the stand-in needs seconds; a real model on CPU, minutes, and the
    first answer after a deploy also downloads the full catalog bundle."""
    return 180 if stack()["model_api"] == "stub" else 1200


def wait_for_answer(base: str, job_id: str, token: str, timeout: int | None = None) -> dict:
    """Poll the job until the worker is done with it (as the page does)."""
    timeout = timeout or answer_timeout()
    deadline = time.monotonic() + timeout
    while True:
        status, body = api_call("GET", f"{base}/api/chat/{job_id}", token)
        if status != 200 or body.get("status") in ("done", "failed"):
            return {"http_status": status, **body}
        if time.monotonic() > deadline:
            raise SystemExit(f"No answer to job {job_id} within {timeout}s: {body}")
        time.sleep(3)


def dead_letters() -> int:
    """Messages in the recommender's dead-letter queue."""
    status, _, text = request(
        "POST",
        f"{ENDPOINT}/",
        json.dumps({"QueueUrl": stack()["dead_letters"], "AttributeNames": ["ApproximateNumberOfMessages"]}),
        {**signed("sqs", "application/x-amz-json-1.0"), "X-Amz-Target": "AmazonSQS.GetQueueAttributes"},
    )
    if status != 200:
        raise SystemExit(f"Cannot read the dead-letter queue: HTTP {status} {text[:200]}")
    return int(json.loads(text)["Attributes"]["ApproximateNumberOfMessages"])


def session_snippet(email: str) -> str:
    """The same shape app.js stores in sessionStorage after a real sign-in."""
    result = sign_in(email)
    session = {
        "email": email,
        "accessToken": result["AccessToken"],
        "idToken": result["IdToken"],
        "refreshToken": result.get("RefreshToken"),
        "expiresAt": int(time.time() * 1000) + result.get("ExpiresIn", 3600) * 1000,
    }
    return f"sessionStorage.setItem('rag.session', {json.dumps(json.dumps(session))}); location.reload();"


# ---------- End-to-end checks ----------


def check(ok: bool, what: str, detail=""):
    if not ok:
        raise SystemExit(f"FAIL {what} {detail}".rstrip())
    print(f"ok   {what}")


def e2e():
    s = stack()

    status, _, html = request("GET", f"{s['page']}/")
    check(status == 200 and 'src="config.js"' in html, "CloudFront serves the page", status)
    status, _, config = request("GET", f"{s['page']}/config.js")
    check(status == 200 and s["client_id"] in config, "config.js carries the app client ID", status)

    try:
        cognito(
            "SignUp",
            {
                "Username": "someone@gmail.com",
                "Password": LOCAL_PASSWORD,
                "UserAttributes": [{"Name": "email", "Value": "someone@gmail.com"}],
            },
        )
        check(False, "pre sign-up Lambda rejects other domains", "(sign-up succeeded)")
    except CognitoError as err:
        check(
            err.code == "UserLambdaValidationException" and "Uppsala" in str(err),
            "pre sign-up Lambda rejects other domains",
            err,
        )

    email = f"e2e-{uuid.uuid4().hex[:8]}@student.uu.se"
    signup(email)
    check(True, f"{email} registers with the code sent through SES")

    token = sign_in(email)["AccessToken"]
    check(bool(token), "the confirmed user gets an access token")

    started = time.monotonic()
    wait_for_model()
    check(True, f"the model {s['model_id']} is ready (after {time.monotonic() - started:.0f}s)")

    for name, base in (("API", s["api"]), ("CloudFront /api", s["page"])):
        status, _ = chat(base, "hello", None)
        check(status == 401, f"{name}: no token is rejected", status)
        status, _ = chat(base, "hello", "not.a.token")
        check(status == 401, f"{name}: a bad token is rejected", status)

    status, body = chat(s["api"], "   ", token)
    check(status == 400, "an empty message is refused", (status, body))

    # Through CloudFront, as the page asks: a question about the fixture catalog, then a follow-up.
    status, job = chat(s["page"], FIXTURE_QUESTION, token)
    check(status == 202 and job.get("status") == "pending", "a question is queued for the recommender", (status, job))
    started = time.monotonic()
    result = wait_for_answer(s["page"], job["job_id"], token)
    # The model's words vary from run to run; check that an answer came back, not what it says.
    check(result.get("status") == "done" and (result.get("answer") or "").strip() != "",
          f"the worker answers it ({time.monotonic() - started:.0f}s)", result)
    print(f"     {result['answer'][:160]!r}")
    print(f"     trace {result.get('trace_id')} in Phoenix (http://localhost:6006)")

    status, follow = chat(s["page"], "Tell me more about the first one.", token, job["conversation_id"])
    check(status == 202 and follow["conversation_id"] == job["conversation_id"], "a follow-up joins the conversation",
          (status, follow))
    result = wait_for_answer(s["page"], follow["job_id"], token)
    check(result.get("status") in ("done", "failed") and (result.get("answer") or result.get("error")),
          f"the follow-up gets an answer or a reason ({result.get('status')})", result)

    other = f"e2e-{uuid.uuid4().hex[:8]}@student.uu.se"
    signup(other)
    status, _ = api_call("GET", f"{s['page']}/api/chat/{job['job_id']}", sign_in(other)["AccessToken"])
    check(status == 404, "another user can't read the answer", status)

    check(dead_letters() == 0, "no question ended in the dead-letter queue")

    print("e2e: sign-in and chat flow passed")


# ---------- CLI ----------

app = typer.Typer(
    help="Drive the sign-in and chat flow on the stack deployed to LocalStack (`just apply`).",
    no_args_is_help=True,
    add_completion=False,
)

Email = Annotated[str, typer.Argument(help="Address to use, e.g. me@student.uu.se")]


@app.command("outputs")
def show_outputs():
    """Page URL, API URL, user pool and client IDs, the recommender's model and dead-letter queue."""
    for key, value in stack().items():
        typer.echo(f"{key:13} {value}")


@app.command("codes")
def show_codes(email: Annotated[str | None, typer.Argument(help="Only codes sent to this address")] = None):
    """Codes LocalStack's SES has "sent", newest last."""
    for to, code in codes(email):
        typer.echo(f"{to:40} {code}")


@app.command("signup")
def signup_command(email: Email):
    """Register EMAIL and confirm it with the emailed code."""
    typer.echo(f"Registered {email} with code {signup(email)} (local password: {LOCAL_PASSWORD})")


@app.command()
def token(email: Email):
    """Print an access token for EMAIL."""
    typer.echo(sign_in(email)["AccessToken"])


@app.command("chat")
def chat_command(
    email: Email,
    message: Annotated[str, typer.Argument(help="Message for the chat")],
    conversation: Annotated[str | None, typer.Option(help="Continue this conversation_id")] = None,
):
    """Sign in as EMAIL, ask MESSAGE through CloudFront and wait for the answer."""
    wait_for_model()
    token = sign_in(email)["AccessToken"]
    status, job = chat(stack()["page"], message, token, conversation)
    if status != 202:
        raise SystemExit(f"{status} {json.dumps(job)}")
    typer.echo(f"Queued job {job['job_id']} in conversation {job['conversation_id']}; waiting for the worker...")
    result = wait_for_answer(stack()["page"], job["job_id"], token)
    typer.echo(result.get("answer") or f"{result.get('status')}: {result.get('error')}")
    typer.echo(f"\nTrace {result.get('trace_id')} (Phoenix: http://localhost:6006). "
               f"Follow up with --conversation {job['conversation_id']}")


@app.command()
def session(email: Email):
    """Print a snippet for the browser console that signs the page in as EMAIL."""
    typer.echo(session_snippet(email))


@app.command("e2e")
def e2e_command():
    """Run every check; exits non-zero on the first failure."""
    e2e()


if __name__ == "__main__":
    try:
        app()
    except CognitoError as err:
        sys.exit(str(err))
