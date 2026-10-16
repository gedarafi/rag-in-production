"""Drive the sign-in and chat flow against the stack deployed on LocalStack (`just apply`).

LocalStack's Cognito has no passwordless sign-in (SignUp without a password and the EMAIL_OTP
challenge are not emulated), so locally every user gets LOCAL_PASSWORD and signs in with
USER_AUTH + PASSWORD. Everything else is the real thing: the pre sign-up Lambda gates the
domain, the confirmation code goes through SES, and the API checks the Cognito token.

Only the standard library: Cognito's sign-up and sign-in calls need no AWS signature, so this
talks to the same JSON API the page does.

Usage: python3 local_flow.py <command> [args]   (see `just --list` for the wrappers)
  outputs                 page URL, API URL, user pool and client IDs
  codes [EMAIL]           codes LocalStack's SES has "sent", newest last
  signup EMAIL            register EMAIL and confirm it with the emailed code
  token EMAIL             print an access token for EMAIL
  chat EMAIL MESSAGE      send MESSAGE to the chat through CloudFront
  session EMAIL           print a snippet for the browser console that signs the page in
  e2e                     run every check; exits non-zero on the first failure
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
    auth, api, web = outputs("auth"), outputs("chat-api"), outputs("web")
    return {
        "page": gateway_url(web["url"]),
        "api": gateway_url(api["api_endpoint"]),
        "user_pool_id": auth["user_pool_id"],
        "client_id": auth["user_pool_client_id"],
        "ses_identity": auth["ses_identity"],
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


def verify_sender():
    """Verify the stack's sending identity in LocalStack's SES, which LocalStack does at once.

    Cognito's code emails are rejected until the sender is verified, and live/local can't create
    the identity itself (LocalStack can't tag SESv2 identities). This is also the first SES call,
    which makes LocalStack load SES and with it the /_aws/ses inbox. The signature is not checked,
    only the service in the credential scope.
    """
    s = stack()
    identity, region = s["ses_identity"], s["user_pool_id"].split("_")[0]
    if "@" in identity:
        params = {"Action": "VerifyEmailIdentity", "EmailAddress": identity}
    else:
        params = {"Action": "VerifyDomainIdentity", "Domain": identity}
    status, _, text = request(
        "POST",
        f"{ENDPOINT}/",
        urllib.parse.urlencode({**params, "Version": "2010-12-01"}),
        {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"AWS4-HMAC-SHA256 Credential=test/20240101/{region}/ses/aws4_request, "
            "SignedHeaders=host, Signature=0",
        },
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


def chat(base: str, message: str, token: str | None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, _, text = request("POST", f"{base}/api/chat", json.dumps({"message": message}), headers)
    try:
        return status, json.loads(text)
    except json.JSONDecodeError:
        return status, {"raw": text}


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

    for name, base in (("API", s["api"]), ("CloudFront /api", s["page"])):
        status, _ = chat(base, "hello", None)
        check(status == 401, f"{name}: no token is rejected", status)
        status, _ = chat(base, "hello", "not.a.token")
        check(status == 401, f"{name}: a bad token is rejected", status)
        status, body = chat(base, "hello", token)
        check(
            status == 200 and body.get("status") == "under_construction",
            f"{name}: a signed-in user gets the placeholder answer",
            (status, body),
        )

    status, body = chat(s["api"], "   ", token)
    check(status == 400, "an empty message is refused", (status, body))

    print("e2e: sign-in and chat flow passed")


# ---------- CLI ----------


def main(argv: list[str]):
    cmd, args = (argv[0], argv[1:]) if argv else ("help", [])
    if cmd == "outputs":
        for key, value in stack().items():
            print(f"{key:13} {value}")
    elif cmd == "codes":
        for to, code in codes(args[0] if args else None):
            print(f"{to:40} {code}")
    elif cmd == "signup" and len(args) == 1:
        print(f"Registered {args[0]} with code {signup(args[0])} (local password: {LOCAL_PASSWORD})")
    elif cmd == "token" and len(args) == 1:
        print(sign_in(args[0])["AccessToken"])
    elif cmd == "chat" and len(args) == 2:
        status, body = chat(stack()["page"], args[1], sign_in(args[0])["AccessToken"])
        print(status, json.dumps(body, indent=2))
    elif cmd == "session" and len(args) == 1:
        print(session_snippet(args[0]))
    elif cmd == "e2e":
        e2e()
    else:
        print(__doc__)
        sys.exit(0 if cmd == "help" else 2)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except CognitoError as err:
        sys.exit(str(err))
