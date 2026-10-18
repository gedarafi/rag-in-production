import importlib

import pytest


@pytest.fixture
def load_handler(monkeypatch):
    """The trigger reads ALLOWED_DOMAINS at import, so each allow-list reloads the module."""

    def load(domains: str):
        monkeypatch.setenv("ALLOWED_DOMAINS", domains)
        from src.auth import pre_signup

        return importlib.reload(pre_signup).handler

    return load


@pytest.fixture
def handler(load_handler):
    return load_handler("uu.se, student.uu.se")


def event(email: str) -> dict:
    return {"request": {"userAttributes": {"email": email}}, "response": {}}


@pytest.mark.parametrize("email", ["anna@uu.se", "Bo.Ek@Student.UU.se", "  carl@uu.se "])
def test_allows_listed_domains_case_insensitive(handler, email):
    result = handler(event(email), None)
    assert result["response"] == {}, "must not auto-confirm or auto-verify"


@pytest.mark.parametrize("email", ["x@gmail.com", "x@evil-uu.se", "x@uu.se.example.com", "x@sub.uu.se", "@uu.se", "uu.se", ""])
def test_rejects_other_and_look_alike_domains(handler, email):
    with pytest.raises(Exception, match="Uppsala University"):
        handler(event(email), None)


def test_missing_email_attribute_is_rejected(handler):
    with pytest.raises(Exception, match="Uppsala University"):
        handler({"request": {"userAttributes": {}}, "response": {}}, None)


def test_empty_allow_list_rejects_everyone(load_handler):
    handler = load_handler("")
    with pytest.raises(Exception, match="Uppsala University"):
        handler(event("anna@uu.se"), None)
