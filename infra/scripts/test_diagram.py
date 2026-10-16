"""Tests for diagram.build: which services and arrows a deployed state turns into (no rendering)."""

import unittest

from diagram import BEDROCK, USERS, build

ACCOUNT = "123456789012"
POOL_ID = "eu-north-1_AbCdEf"
CHAT_ARN = f"arn:aws:lambda:eu-north-1:{ACCOUNT}:function:rag-chat"


def state(*resources):
    return {"values": {"root_module": {"resources": [{"mode": "managed", **r} for r in resources]}}}


def resource(type_, name, /, **values):
    return {"type": type_, "name": name, "address": f"{type_}.{name}", "values": values}


UNITS = {
    "auth": state(
        resource("aws_cognito_user_pool", "this", id=POOL_ID, arn=f"arn:aws:cognito-idp:eu-north-1:{ACCOUNT}:userpool/{POOL_ID}", name="rag-users"),
        resource("aws_cognito_user_pool_client", "web", id="client123456"),
    ),
    "chat-api": state(
        resource("aws_apigatewayv2_api", "this", id="abc123defg", name="rag-api"),
        resource("aws_lambda_function", "chat", id="rag-chat", arn=CHAT_ARN, function_name="rag-chat",
                 environment=[{"variables": {"MODEL_ID": "us.amazon.nova-micro-v1:0"}}]),
        # A second Lambda whose name starts with the first one's must not catch its arrows.
        resource("aws_lambda_function", "chatbot", id="rag-chatbot", arn=CHAT_ARN + "bot", function_name="rag-chatbot"),
        resource("aws_apigatewayv2_integration", "chat", id="int1", api_id="abc123defg",
                 integration_uri=f"arn:aws:apigateway:eu-north-1:lambda:path/2015-03-31/functions/{CHAT_ARN}/invocations"),
        resource("aws_apigatewayv2_authorizer", "cognito", id="auth1", api_id="abc123defg",
                 jwt_configuration=[{"issuer": f"https://cognito-idp.eu-north-1.amazonaws.com/{POOL_ID}"}]),
    ),
    "web": state(
        resource("aws_s3_bucket", "site", id=f"rag-web-{ACCOUNT}", arn=f"arn:aws:s3:::rag-web-{ACCOUNT}", bucket=f"rag-web-{ACCOUNT}",
                 policy='{"Condition": {"StringEquals": {"AWS:SourceArn": "arn:aws:cloudfront::123456789012:distribution/E1ABCDEF"}}}'),
        resource("aws_cloudfront_distribution", "site", id="E1ABCDEF", arn=f"arn:aws:cloudfront::{ACCOUNT}:distribution/E1ABCDEF",
                 comment="rag web", origin=[{"domain_name": f"rag-web-{ACCOUNT}.s3.eu-north-1.amazonaws.com"},
                                           {"domain_name": "abc123defg.execute-api.eu-north-1.amazonaws.com"}]),
    ),
}


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.graph = build(UNITS, ACCOUNT)

    def edge(self, src, dst):
        return self.graph.edges.get((src, dst))

    def test_draws_services_but_not_helpers(self):
        kinds = sorted(n.kind for n in self.graph.nodes.values())
        self.assertEqual(kinds, sorted([
            "aws_cognito_user_pool", "aws_apigatewayv2_api", "aws_lambda_function", "aws_lambda_function",
            "aws_s3_bucket", "aws_cloudfront_distribution", BEDROCK, USERS,
        ]))

    def test_cloudfront_serves_the_bucket_and_the_api(self):
        self.assertEqual(self.edge("web/aws_cloudfront_distribution.site", "web/aws_s3_bucket.site"), "static page")
        self.assertEqual(self.edge("web/aws_cloudfront_distribution.site", "chat-api/aws_apigatewayv2_api.this"), "/api/*")

    def test_a_bucket_policy_does_not_draw_an_arrow_back_to_cloudfront(self):
        self.assertIsNone(self.edge("web/aws_s3_bucket.site", "web/aws_cloudfront_distribution.site"))

    def test_connectors_link_the_api_to_the_lambda_and_cognito(self):
        self.assertEqual(self.edge("chat-api/aws_apigatewayv2_api.this", "chat-api/aws_lambda_function.chat"), "invokes")
        self.assertEqual(self.edge("chat-api/aws_apigatewayv2_api.this", "auth/aws_cognito_user_pool.this"), "checks JWT")
        self.assertIsNone(self.edge("chat-api/aws_apigatewayv2_api.this", "chat-api/aws_lambda_function.chatbot"))

    def test_the_lambda_calls_its_bedrock_model(self):
        self.assertEqual(self.edge("chat-api/aws_lambda_function.chat", BEDROCK), "Converse")
        self.assertIn("us.amazon.nova-micro-v1:0", self.graph.nodes[BEDROCK].label)

    def test_users_reach_cloudfront_and_cognito(self):
        self.assertEqual(self.edge(USERS, "web/aws_cloudfront_distribution.site"), "HTTPS")
        self.assertEqual(self.edge(USERS, "auth/aws_cognito_user_pool.this"), "sign in")

    def test_hides_the_account_id(self):
        self.assertEqual(self.graph.nodes["web/aws_s3_bucket.site"].label, "S3\nrag-web-<account>")

    def test_nothing_deployed_draws_nothing(self):
        graph = build({"auth": {"values": {}}})
        self.assertEqual((graph.nodes, graph.edges), ({}, {}))


if __name__ == "__main__":
    unittest.main()
