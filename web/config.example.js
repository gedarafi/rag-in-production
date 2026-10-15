// Local development only. Copy to config.js (gitignored) and fill in from `just aws-outputs`.
// When deployed, CloudFront serves a generated config.js and apiBase stays "" (same origin).
window.APP_CONFIG = {
  region: "eu-north-1",
  userPoolClientId: "REPLACE_WITH_user_pool_client_id",
  allowedEmailDomains: ["uu.se", "student.uu.se"],
  // Locally the page calls the API directly; the chat-api allows http://localhost:8000 via CORS.
  apiBase: "https://REPLACE_WITH_api_domain",
};
