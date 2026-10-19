"""Tests for diagram.build: which services and arrows a deployed state turns into (no rendering)."""

import pytest

from diagram import BEDROCK, USERS, bedrock_key, build

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


@pytest.fixture(scope="module")
def graph():
    return build(UNITS, ACCOUNT)


def edge(graph, src, dst):
    return graph.edges.get((src, dst))


def test_draws_services_but_not_helpers(graph):
    kinds = sorted(n.kind for n in graph.nodes.values())
    assert kinds == sorted([
        "aws_cognito_user_pool", "aws_apigatewayv2_api", "aws_lambda_function", "aws_lambda_function",
        "aws_s3_bucket", "aws_cloudfront_distribution", BEDROCK, USERS,
    ])


def test_cloudfront_serves_the_bucket_and_the_api(graph):
    assert edge(graph, "web/aws_cloudfront_distribution.site", "web/aws_s3_bucket.site") == "static page"
    assert edge(graph, "web/aws_cloudfront_distribution.site", "chat-api/aws_apigatewayv2_api.this") == "/api/*"


def test_a_bucket_policy_does_not_draw_an_arrow_back_to_cloudfront(graph):
    assert edge(graph, "web/aws_s3_bucket.site", "web/aws_cloudfront_distribution.site") is None


def test_connectors_link_the_api_to_the_lambda_and_cognito(graph):
    assert edge(graph, "chat-api/aws_apigatewayv2_api.this", "chat-api/aws_lambda_function.chat") == "invokes"
    assert edge(graph, "chat-api/aws_apigatewayv2_api.this", "auth/aws_cognito_user_pool.this") == "checks JWT"
    assert edge(graph, "chat-api/aws_apigatewayv2_api.this", "chat-api/aws_lambda_function.chatbot") is None


def test_the_lambda_calls_its_bedrock_model(graph):
    nova = bedrock_key("us.amazon.nova-micro-v1:0")
    assert edge(graph, "chat-api/aws_lambda_function.chat", nova) == "Converse"
    assert graph.nodes[nova].label == "Bedrock\nus.amazon.nova-micro-v1:0"


def test_the_lambda_calls_bedrock_mantle_in_openai_mode():
    lam = resource("aws_lambda_function", "chat", id="rag-chat", arn=CHAT_ARN, function_name="rag-chat",
                   environment=[{"variables": {"MODEL_ID": "google.gemma-3-4b-it", "MODEL_API": "openai"}}])
    graph = build({"chat-api": state(lam)}, ACCOUNT)
    gemma = bedrock_key("google.gemma-3-4b-it")
    assert graph.edges.get(("chat-api/aws_lambda_function.chat", gemma)) == "Chat Completions"
    assert graph.nodes[gemma].label == "Bedrock Mantle\ngoogle.gemma-3-4b-it"


RECOMMENDER_BUCKET = f"rag-recommender-{ACCOUNT}"
EIP_DNS = "ec2-3-128-128-76.eu-north-1.compute.amazonaws.com"
RECOMMENDER = state(
    resource("aws_s3_bucket", "this", id=RECOMMENDER_BUCKET, arn=f"arn:aws:s3:::{RECOMMENDER_BUCKET}", bucket=RECOMMENDER_BUCKET),
    resource("aws_instance", "host", id="i-0123456789abcdef0", arn=f"arn:aws:ec2:eu-north-1:{ACCOUNT}:instance/i-0123456789abcdef0",
             public_dns="ec2-13-48-1-2.eu-north-1.compute.amazonaws.com", tags={"Name": "rag-recommender"},
             user_data=(f"cat > /etc/recommender.env <<'ENV'\nRECOMMENDER_BUCKET={RECOMMENDER_BUCKET}\n"
                        f"COGNITO_ISSUER=https://cognito-idp.eu-north-1.amazonaws.com/{POOL_ID}\n"
                        "RAG_PROVIDER=bedrock\nRAG_BEDROCK_MODEL=qwen.qwen3-32b\nENV\n")),
    resource("aws_eip", "host", id="eipalloc-0123456789", instance="i-0123456789abcdef0", public_dns=EIP_DNS, public_ip="3.128.128.76"),
)


def test_cloudfront_reaches_the_recommender_through_its_elastic_ip():
    web = resource("aws_cloudfront_distribution", "site", id="E1ABCDEF", comment="rag web", origin=[{"domain_name": EIP_DNS}])
    graph = build({"auth": UNITS["auth"], "recommender-host": RECOMMENDER, "web": state(web)}, ACCOUNT)
    host = "recommender-host/aws_instance.host"
    assert graph.nodes[host].label == "EC2\nrag-recommender"
    assert edge(graph, "web/aws_cloudfront_distribution.site", host) == "/recommend/*"
    assert edge(graph, host, "recommender-host/aws_s3_bucket.this") == "pulls library"
    assert edge(graph, host, "auth/aws_cognito_user_pool.this") == "checks JWT"
    assert edge(graph, host, bedrock_key("qwen.qwen3-32b")) == "Chat Completions"
    assert not any("aws_eip" in key for key in graph.nodes)


def test_each_model_gets_its_own_bedrock_node():
    graph = build({**UNITS, "recommender-host": RECOMMENDER}, ACCOUNT)
    models = sorted(n.label for n in graph.nodes.values() if n.kind == BEDROCK)
    assert models == ["Bedrock\nus.amazon.nova-micro-v1:0", "Bedrock Mantle\nqwen.qwen3-32b"]


def test_users_reach_cloudfront_and_cognito(graph):
    assert edge(graph, USERS, "web/aws_cloudfront_distribution.site") == "HTTPS"
    assert edge(graph, USERS, "auth/aws_cognito_user_pool.this") == "sign in"


def test_hides_the_account_id(graph):
    assert graph.nodes["web/aws_s3_bucket.site"].label == "S3\nrag-web-<account>"


def test_nothing_deployed_draws_nothing():
    graph = build({"auth": {"values": {}}})
    assert (graph.nodes, graph.edges) == ({}, {})
