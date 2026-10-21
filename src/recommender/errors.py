"""Errors whose message is written for the person asking, so the web service may show it."""


class UserFacingError(RuntimeError):
    """A failure the user can act on; its message never names hosts, paths or upstream errors."""
