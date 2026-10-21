// Passwordless sign-in with Cognito email one-time codes, then a chat box.
// Talks to the Cognito API directly (no SDK, no build step). Tokens live in sessionStorage,
// so closing the tab signs the user out.
"use strict";

const cfg = window.APP_CONFIG;
const $ = (id) => document.getElementById(id);

// ---------- Cognito API ----------

class CognitoError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

async function cognito(action, body) {
  const res = await fetch(`https://cognito-idp.${cfg.region}.amazonaws.com/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-amz-json-1.1",
      "X-Amz-Target": `AWSCognitoIdentityProviderService.${action}`,
    },
    body: JSON.stringify({ ClientId: cfg.userPoolClientId, ...body }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const code = String(data.__type || "Error").split("#").pop();
    throw new CognitoError(code, data.message || data.Message || "Something went wrong.");
  }
  return data;
}

function friendly(err) {
  switch (err.code) {
    case "UserLambdaValidationException":
      // Message from the pre sign-up Lambda, after Cognito's "PreSignUp failed with error " prefix.
      return err.message.replace(/^PreSignUp failed with error\s*/i, "").replace(/[.\s]*$/, ".");
    case "CodeMismatchException":
    case "NotAuthorizedException":
      return "That code is not right, or it has expired. Check it or send a new one.";
    case "ExpiredCodeException":
      return "That code has expired. Send a new one.";
    case "LimitExceededException":
    case "TooManyRequestsException":
    case "TooManyFailedAttemptsException":
      return "Too many attempts. Wait a few minutes and try again.";
    case "CodeDeliveryFailureException":
      return "We couldn't send the email. Try again later.";
    default:
      return err.message || "Something went wrong. Try again.";
  }
}

// ---------- Session storage ----------

const STORE_KEY = "rag.session";

function saveSession(result, email) {
  const prev = loadSession() || {};
  const session = {
    email: email || prev.email,
    accessToken: result.AccessToken,
    idToken: result.IdToken,
    refreshToken: result.RefreshToken || prev.refreshToken,
    expiresAt: Date.now() + (result.ExpiresIn || 3600) * 1000,
  };
  try { sessionStorage.setItem(STORE_KEY, JSON.stringify(session)); } catch { /* storage blocked */ }
  memorySession = session;
  return session;
}

let memorySession = null;

function loadSession() {
  if (memorySession) return memorySession;
  try {
    memorySession = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null");
  } catch {
    memorySession = null;
  }
  return memorySession;
}

function clearSession() {
  memorySession = null;
  history = [];
  try { sessionStorage.removeItem(STORE_KEY); } catch { /* storage blocked */ }
}

async function validAccessToken() {
  const s = loadSession();
  if (!s) return null;
  if (Date.now() < s.expiresAt - 60_000) return s.accessToken;
  if (!s.refreshToken) return null;
  try {
    const { AuthenticationResult } = await cognito("InitiateAuth", {
      AuthFlow: "REFRESH_TOKEN_AUTH",
      AuthParameters: { REFRESH_TOKEN: s.refreshToken },
    });
    return saveSession(AuthenticationResult).accessToken;
  } catch {
    clearSession();
    return null;
  }
}

// ---------- Sign-in flow ----------
// New users: SignUp -> code by email -> ConfirmSignUp -> signed in.
// Returning users: InitiateAuth (USER_AUTH, EMAIL_OTP) -> code by email -> RespondToAuthChallenge.

const flow = { email: "", mode: "", session: "" };

function emailAllowed(email) {
  const domain = email.split("@")[1] || "";
  return email.includes("@") && cfg.allowedEmailDomains.includes(domain);
}

async function startWithEmail(email) {
  flow.email = email;
  try {
    await cognito("SignUp", {
      Username: email,
      UserAttributes: [{ Name: "email", Value: email }],
    });
    flow.mode = "signup";
  } catch (err) {
    if (err.code !== "UsernameExistsException") throw err;
    await startSignIn();
  }
}

async function startSignIn() {
  try {
    const res = await cognito("InitiateAuth", {
      AuthFlow: "USER_AUTH",
      AuthParameters: { USERNAME: flow.email, PREFERRED_CHALLENGE: "EMAIL_OTP" },
    });
    await handleChallenge(res);
    flow.mode = "signin";
  } catch (err) {
    // Registered earlier but never entered the code: finish the registration instead.
    if (err.code !== "UserNotConfirmedException") throw err;
    await cognito("ResendConfirmationCode", { Username: flow.email });
    flow.mode = "signup";
  }
}

async function handleChallenge(res) {
  if (res.ChallengeName === "SELECT_CHALLENGE") {
    res = await cognito("RespondToAuthChallenge", {
      ChallengeName: "SELECT_CHALLENGE",
      Session: res.Session,
      ChallengeResponses: { USERNAME: flow.email, ANSWER: "EMAIL_OTP" },
    });
  }
  if (res.ChallengeName !== "EMAIL_OTP") {
    throw new CognitoError("UnexpectedChallenge", "Email sign-in isn't available for this account.");
  }
  flow.session = res.Session;
}

async function submitCode(code) {
  if (flow.mode === "signup") {
    const confirmed = await cognito("ConfirmSignUp", { Username: flow.email, ConfirmationCode: code });
    // The confirmation session signs the new user straight in, without a second code.
    try {
      const { AuthenticationResult } = await cognito("InitiateAuth", {
        AuthFlow: "USER_AUTH",
        AuthParameters: { USERNAME: flow.email },
        Session: confirmed.Session,
      });
      if (AuthenticationResult) return saveSession(AuthenticationResult, flow.email);
    } catch { /* fall through to a normal sign-in */ }
    await startSignIn();
    throw new CognitoError("Registered", "Your email is confirmed. We sent one more code to sign you in.");
  }

  const res = await cognito("RespondToAuthChallenge", {
    ChallengeName: "EMAIL_OTP",
    Session: flow.session,
    ChallengeResponses: { USERNAME: flow.email, EMAIL_OTP_CODE: code },
  });
  if (res.Session) flow.session = res.Session; // a wrong code returns a fresh session for retrying
  if (!res.AuthenticationResult) throw new CognitoError("CodeMismatchException", "Wrong code.");
  return saveSession(res.AuthenticationResult, flow.email);
}

async function resendCode() {
  if (flow.mode === "signup") {
    await cognito("ResendConfirmationCode", { Username: flow.email });
  } else {
    await startSignIn();
  }
}

async function signOut() {
  const s = loadSession();
  clearSession();
  if (s?.refreshToken) {
    // Revokes the refresh token and the tokens issued from it.
    await cognito("RevokeToken", { Token: s.refreshToken }).catch(() => {});
  }
  show("email");
}

// ---------- Chat ----------

// The anime recommender (on EC2, under /recommend/*) answers when config.js enables it: `recommender`
// when CloudFront routes it, `recommenderBase` when the page calls it directly in local development.
const useRecommender = Boolean(cfg && (cfg.recommender || cfg.recommenderBase));
// Earlier turns of this conversation, so follow-ups like "something shorter" have context.
let history = [];

async function sendMessage(text) {
  const token = await validAccessToken();
  if (!token) {
    show("email");
    throw new Error("Your session has ended. Sign in again.");
  }
  if (useRecommender) return askRecommender(text, token);
  const res = await fetch(`${cfg.apiBase}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ message: text }),
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) {
    clearSession();
    show("email");
    throw new Error("Your session has ended. Sign in again.");
  }
  if (res.status === 429) throw new Error("The assistant is busy. Try again in a moment.");
  if (!res.ok) throw new Error(data.error || "The assistant couldn't answer. Try again.");
  return data.answer;
}

async function askRecommender(text, token) {
  const res = await fetch(`${cfg.recommenderBase || cfg.apiBase}/recommend/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ message: text, history }),
  });
  if (res.status === 401) {
    clearSession();
    show("email");
    throw new Error("Your session has ended. Sign in again.");
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    if (res.status === 429) throw new Error("The assistant is busy. Try again in a moment.");
    throw new Error(data.error || "The assistant couldn't answer. Try again.");
  }
  // NDJSON: {"type":"wait"} lines while the models work, then one answer or error line.
  const lines = (await res.text()).trim().split("\n");
  const last = JSON.parse(lines[lines.length - 1] || "{}");
  if (last.type !== "answer") throw new Error(last.error || "The assistant couldn't answer. Try again.");
  history = [...history, { role: "user", content: text }, { role: "assistant", content: last.answer }].slice(-20);
  return last.answer;
}

function addMessage(text, who, pending = false) {
  const li = document.createElement("li");
  li.className = `from-${who}${pending ? " pending" : ""}`;
  li.textContent = text; // never innerHTML: answers will contain document text later
  $("messages").append(li);
  li.scrollIntoView({ block: "end", behavior: "smooth" });
  return li;
}

// ---------- UI ----------

function show(view) {
  for (const name of ["email", "code", "chat"]) $(`view-${name}`).hidden = name !== view;
  const s = loadSession();
  $("user-box").hidden = view !== "chat";
  if (view === "chat" && s) $("user-email").textContent = s.email || "";
  const focus = { email: "email", code: "code", chat: "message" }[view];
  $(focus)?.focus();
}

async function busy(button, fn) {
  button.disabled = true;
  try { return await fn(); } finally { button.disabled = false; }
}

function bind() {
  $("domain-hint").textContent = `Allowed: ${cfg.allowedEmailDomains.map((d) => "@" + d).join(", ")}`;

  $("email-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const email = $("email").value.trim().toLowerCase();
    $("email-error").textContent = "";
    if (!emailAllowed(email)) {
      $("email-error").textContent = `Use an email ending in ${cfg.allowedEmailDomains.map((d) => "@" + d).join(" or ")}.`;
      return;
    }
    busy($("email-submit"), async () => {
      try {
        await startWithEmail(email);
        $("code-email").textContent = email;
        $("code").value = "";
        $("code-error").textContent = "";
        show("code");
      } catch (err) {
        $("email-error").textContent = friendly(err);
      }
    });
  });

  $("code").addEventListener("input", () => { $("code-error").textContent = ""; });

  $("code-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const code = $("code").value.replace(/\s/g, "");
    if (!/^\d{6,8}$/.test(code)) {
      $("code-error").textContent = "Enter the code from the email (digits only).";
      return;
    }
    busy($("code-submit"), async () => {
      try {
        await submitCode(code);
        show("chat");
      } catch (err) {
        $("code").value = "";
        $("code-error").textContent = err.code === "Registered" ? err.message : friendly(err);
      }
    });
  });

  $("resend").addEventListener("click", () => busy($("resend"), async () => {
    try {
      await resendCode();
      $("code-error").textContent = "We sent a new code.";
    } catch (err) {
      $("code-error").textContent = friendly(err);
    }
  }));

  $("change-email").addEventListener("click", () => show("email"));
  $("sign-out").addEventListener("click", signOut);

  if (useRecommender) {
    $("message").placeholder =
      'Ask about any anime or for recommendations, e.g. "What is Assassination Classroom about?" or "Recommend five comedy anime like it"';
  }
  $("message").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("message").value.trim();
    $("chat-error").textContent = "";
    if (!text) return;
    $("message").value = "";
    addMessage(text, "user");
    const pending = addMessage("Thinking…", "bot", true);
    busy($("send"), async () => {
      try {
        pending.textContent = await sendMessage(text);
        pending.classList.remove("pending");
      } catch (err) {
        pending.remove();
        $("chat-error").textContent = err.message;
      }
    });
  });
}

async function init() {
  if (!cfg || !cfg.region || !cfg.userPoolClientId) {
    $("config-error").hidden = false;
    return;
  }
  bind();
  show((await validAccessToken()) ? "chat" : "email");
}

init();
