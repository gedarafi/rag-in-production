import importlib
import os
import unittest


def load_handler(domains: str):
    os.environ["ALLOWED_DOMAINS"] = domains
    import pre_signup

    return importlib.reload(pre_signup).handler


def event(email: str) -> dict:
    return {"request": {"userAttributes": {"email": email}}, "response": {}}


class PreSignUpTest(unittest.TestCase):
    def setUp(self):
        self.handler = load_handler("uu.se, student.uu.se")

    def test_allows_listed_domains_case_insensitive(self):
        for email in ["anna@uu.se", "Bo.Ek@Student.UU.se", "  carl@uu.se "]:
            with self.subTest(email=email):
                result = self.handler(event(email), None)
                self.assertEqual(result["response"], {}, "must not auto-confirm or auto-verify")

    def test_rejects_other_and_look_alike_domains(self):
        for email in ["x@gmail.com", "x@evil-uu.se", "x@uu.se.example.com", "x@sub.uu.se", "@uu.se", "uu.se", ""]:
            with self.subTest(email=email), self.assertRaises(Exception):
                self.handler(event(email), None)

    def test_missing_email_attribute_is_rejected(self):
        with self.assertRaises(Exception):
            self.handler({"request": {"userAttributes": {}}, "response": {}}, None)

    def test_empty_allow_list_rejects_everyone(self):
        handler = load_handler("")
        with self.assertRaises(Exception):
            handler(event("anna@uu.se"), None)


if __name__ == "__main__":
    unittest.main()
