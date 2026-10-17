"""Cognito pre sign-up trigger: only allow registration from approved email domains."""

import os

ALLOWED_DOMAINS = frozenset(
    d.strip().lower() for d in os.environ.get("ALLOWED_DOMAINS", "").split(",") if d.strip()
)

REJECTION = "Registration is limited to Uppsala University email addresses."


def handler(event, context):
    email = event["request"]["userAttributes"].get("email", "").strip().lower()
    local, sep, domain = email.rpartition("@")

    # Exact domain match blocks look-alikes such as x@evil-uu.se or x@uu.se.example.com.
    if not sep or not local or domain not in ALLOWED_DOMAINS:
        # Cognito returns this to the client as "PreSignUp failed with error <message>."
        raise Exception(REJECTION)

    # Not auto-confirmed: the user still proves they own the inbox with the emailed code.
    return event
