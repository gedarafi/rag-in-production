# /// script
# requires-python = ">=3.12"
# dependencies = ["diagrams>=0.25", "typer>=0.27"]
# ///
"""Draw the architecture diagram from what OpenTofu has deployed (`just aws-diagram`).

Every service on the diagram is a resource in a unit's state (`terragrunt ... show -json`), grouped
by unit. An arrow from A to B means A's state refers to B (an ARN, ID or domain name): CloudFront's
origins name the bucket, the API and the recommender instance, the API's integration names the chat
Lambda, and so on. Helper resources such as routes, integrations and authorizers are not drawn; they
only add arrows between the services they connect, and an Elastic IP's addresses stand for its
instance. Resource policies are ignored, since they point from a service back to whoever may call it.
Bedrock and the people using the page are not in any state: Bedrock is drawn from the chat Lambda's
MODEL_ID and MODEL_API and from the recommender instance's RAG_BEDROCK_MODEL (in its boot script),
the people from the Cognito app client.

Rendering needs Graphviz (`dot`); the diagrams library supplies the AWS icons.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

# Resource types drawn as a service, with the name shown under the icon.
SERVICES = {
    "aws_cloudfront_distribution": ("CloudFront", "comment"),
    "aws_s3_bucket": ("S3", "bucket"),
    "aws_apigatewayv2_api": ("API Gateway", "name"),
    "aws_lambda_function": ("Lambda", "function_name"),
    "aws_instance": ("EC2", "tags.Name"),
    "aws_cognito_user_pool": ("Cognito", "name"),
    "aws_sesv2_email_identity": ("SES", None),  # the identity is an email address: not shown
    "aws_budgets_budget": ("Budgets", "name"),
}

# Helper resources that connect the service named by `owner` to the services they refer to.
CONNECTORS = {
    "aws_apigatewayv2_integration": "api_id",
    "aws_apigatewayv2_authorizer": "api_id",
}

# Helper resources whose addresses stand for the service named by `owner`: CloudFront's origin is
# the Elastic IP's DNS name, not the instance's own.
ALIASES = {
    "aws_eip": ("instance", ("public_dns", "public_ip")),
}

EDGE_LABELS = {
    ("aws_cloudfront_distribution", "aws_s3_bucket"): "static page",
    ("aws_cloudfront_distribution", "aws_apigatewayv2_api"): "/api/*",
    ("aws_apigatewayv2_api", "aws_lambda_function"): "invokes",
    ("aws_apigatewayv2_api", "aws_cognito_user_pool"): "checks JWT",
    ("aws_cognito_user_pool", "aws_lambda_function"): "pre sign-up",
    ("aws_cognito_user_pool", "aws_sesv2_email_identity"): "sign-in codes",
    ("aws_cloudfront_distribution", "aws_instance"): "/recommend/*",
    ("aws_instance", "aws_s3_bucket"): "pulls library",
    ("aws_instance", "aws_cognito_user_pool"): "checks JWT",
}

API_LABELS = {"openai": "Chat Completions", "converse": "Converse"}

USERS = "users"
BEDROCK = "bedrock"


@dataclass(frozen=True)
class Node:
    key: str  # unit/address, or USERS / bedrock:<model> for the nodes that are not in any state
    kind: str  # resource type, or USERS / BEDROCK
    label: str
    unit: str | None = None


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str], str] = field(default_factory=dict)

    def connect(self, src: str, dst: str, label: str = ""):
        if src != dst and (src, dst) not in self.edges:
            self.edges[(src, dst)] = label


def resources(state: dict) -> list[dict]:
    """Managed resources of a `show -json` state, including those in child modules."""
    found, modules = [], [state.get("values", {}).get("root_module", {})]
    while modules:
        module = modules.pop()
        found += [r for r in module.get("resources", []) if r.get("mode") == "managed"]
        modules += module.get("child_modules", [])
    return found


def strings(value) -> list[str]:
    """Every string inside a resource's attributes."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def mentions(texts: list[str], ref: str) -> bool:
    """True if ref appears in texts as a whole token (so rag-aws-chat does not match rag-aws-chatbot)."""
    pattern = re.compile(rf"(?<![\w-]){re.escape(ref)}(?![\w-])")
    return any(pattern.search(t) for t in texts)


def attribute(values: dict, path: str):
    """values[a][b] for path "a.b"; None when a step is missing."""
    for step in path.split("."):
        values = values.get(step) if isinstance(values, dict) else None
    return values


def boot_env(user_data: str | None) -> dict[str, str]:
    """KEY=value lines of an instance's boot script, such as the env file it writes."""
    return dict(re.findall(r"^([A-Z][A-Z0-9_]*)=(\S+)$", user_data or "", re.MULTILINE))


def bedrock_key(model_id: str) -> str:
    return f"{BEDROCK}:{model_id}"


def build(units: dict[str, dict], account_id: str = "") -> Graph:
    """Turn {unit name: `show -json` state} into the services and the arrows between them."""
    graph, refs, texts, kinds = Graph(), {}, {}, {}
    connectors, aliases, models = [], [], {}

    for unit, state in sorted(units.items()):
        for r in resources(state):
            kind, values = r["type"], r.get("values", {})
            key = f"{unit}/{r['address']}"
            if kind in CONNECTORS:
                connectors.append((values.get(CONNECTORS[kind]), strings(values)))
                continue
            if kind in ALIASES:
                owner_attr, attrs = ALIASES[kind]
                aliases.append((values.get(owner_attr), [v for a in attrs if isinstance(v := values.get(a), str) and v]))
                continue
            if kind not in SERVICES:
                continue
            service, name_attr = SERVICES[kind]
            name = str(attribute(values, name_attr) or r["name"]) if name_attr else r["name"]
            if account_id:
                name = name.replace(account_id, "<account>")
            graph.nodes[key] = Node(key, kind, f"{service}\n{name}", unit)
            refs[key] = [v for v in (values.get("arn"), values.get("id")) if isinstance(v, str) and len(v) >= 6]
            # A resource policy names who may call this service, so it would draw the arrow backwards.
            texts[key] = strings({k: v for k, v in values.items() if k not in ("arn", "id", "policy")})
            kinds[key] = kind
            env = (values.get("environment") or [{}])[0].get("variables") or {}
            if kind == "aws_lambda_function" and env.get("MODEL_ID"):
                models[key] = (env["MODEL_ID"], env.get("MODEL_API", "converse"))
            boot = boot_env(values.get("user_data"))
            if kind == "aws_instance" and boot.get("RAG_PROVIDER") == "bedrock" and boot.get("RAG_BEDROCK_MODEL"):
                models[key] = (boot["RAG_BEDROCK_MODEL"], "openai")  # the recommender calls bedrock-mantle

    for owner_id, addresses in aliases:
        for key, ids in refs.items():
            if owner_id in ids:
                ids += addresses

    def label(src, dst):
        return EDGE_LABELS.get((kinds[src], kinds[dst]), "")

    # A service whose own attributes name another service.
    for src in graph.nodes:
        for dst, ids in refs.items():
            if src != dst and any(mentions(texts[src], i) for i in ids):
                graph.connect(src, dst, label(src, dst))

    # A connector links its owner (e.g. the API) to every other service it names.
    for owner_id, conn_texts in connectors:
        owners = [k for k, ids in refs.items() if owner_id in ids]
        for owner in owners:
            for dst, ids in refs.items():
                if dst != owner and any(mentions(conn_texts, i) for i in ids):
                    graph.connect(owner, dst, label(owner, dst))

    # Not in any state: the models the chat Lambda and the recommender call, and the people using the page.
    for key, (model_id, api) in models.items():
        service = "Bedrock Mantle" if api == "openai" else "Bedrock"
        graph.nodes[bedrock_key(model_id)] = Node(bedrock_key(model_id), BEDROCK, f"{service}\n{model_id}")
        graph.connect(key, bedrock_key(model_id), API_LABELS.get(api, "Converse"))

    has_client = any(r["type"] == "aws_cognito_user_pool_client" for s in units.values() for r in resources(s))
    for key, kind in kinds.items():
        if kind == "aws_cloudfront_distribution":
            graph.nodes[USERS] = Node(USERS, USERS, "Users\n(browser)")
            graph.connect(USERS, key, "HTTPS")
        if kind == "aws_cognito_user_pool" and has_client:
            graph.nodes[USERS] = Node(USERS, USERS, "Users\n(browser)")
            graph.connect(USERS, key, "sign in")
        if kind == "aws_sesv2_email_identity" and USERS in graph.nodes:
            graph.connect(key, USERS, "emails codes")

    return graph


def render(graph: Graph, out: Path, title: str):
    """Draw graph to out.png with the AWS icons, one cluster per unit."""
    from diagrams import Cluster, Diagram, Edge
    from diagrams.aws.compute import EC2, Lambda
    from diagrams.aws.cost import Budgets
    from diagrams.aws.engagement import SES
    from diagrams.aws.ml import Bedrock
    from diagrams.aws.network import APIGateway, CloudFront
    from diagrams.aws.security import Cognito
    from diagrams.aws.storage import S3
    from diagrams.onprem.client import Users

    icons = {
        "aws_cloudfront_distribution": CloudFront,
        "aws_s3_bucket": S3,
        "aws_apigatewayv2_api": APIGateway,
        "aws_lambda_function": Lambda,
        "aws_instance": EC2,
        "aws_cognito_user_pool": Cognito,
        "aws_sesv2_email_identity": SES,
        "aws_budgets_budget": Budgets,
        BEDROCK: Bedrock,
        USERS: Users,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    drawn = {}
    graph_attr = {"fontsize": "20", "pad": "0.5", "splines": "spline", "nodesep": "0.8", "ranksep": "1.1"}
    with Diagram(title, filename=str(out), outformat="png", show=False, direction="LR", graph_attr=graph_attr):
        if USERS in graph.nodes:
            drawn[USERS] = Users(graph.nodes[USERS].label)
        with Cluster("AWS account"):
            for unit in sorted({n.unit for n in graph.nodes.values() if n.unit}):
                with Cluster(f"unit: {unit}"):
                    for node in graph.nodes.values():
                        if node.unit == unit:
                            drawn[node.key] = icons[node.kind](node.label)
            for node in graph.nodes.values():
                if node.kind == BEDROCK:
                    drawn[node.key] = Bedrock(node.label)
        for (src, dst), text in graph.edges.items():
            style = "dashed" if USERS in (src, dst) else "solid"
            drawn[src] >> Edge(label=text, style=style) >> drawn[dst]


def load_states(live: Path, skip: list[str] = ()) -> dict[str, dict]:
    """`show -json` of every unit under live/ except those named in skip, read from its OpenTofu state."""
    units = {}
    for unit in sorted(p.parent for p in live.glob("*/terragrunt.hcl") if p.parent.name not in skip):
        cmd = ["terragrunt", "run", "--non-interactive", "--log-level", "error", "--working-dir", str(unit), "--", "show", "-json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"Could not read the state of {unit}:\n{result.stderr}")
        units[unit.name] = json.loads(result.stdout)
    return units


app = typer.Typer(add_completion=False)


@app.command()
def main(
    live: Annotated[Path, typer.Option(help="Directory of the Terragrunt units to draw")] = Path("live/aws"),
    out: Annotated[Path, typer.Option(help="Output file, without the .png")] = Path("../docs/infra/architecture"),
    title: Annotated[str, typer.Option(help="Diagram title")] = "RAG in Production on AWS",
    account_id: Annotated[str, typer.Option(envvar="RAG_AWS_ACCOUNT_ID", help="Replaced by <account> in names")] = "",
    skip: Annotated[list[str] | None, typer.Option(help="Unit to leave off the diagram (repeatable)")] = None,
):
    """Draw the services deployed by the units under LIVE and the links between them."""
    graph = build(load_states(live, skip or []), account_id)
    if not graph.nodes:
        raise SystemExit(f"No deployed services found under {live}: deploy first.")
    render(graph, out, title)
    typer.echo(f"Wrote {out}.png: {len(graph.nodes)} services, {len(graph.edges)} links")


if __name__ == "__main__":
    app()
