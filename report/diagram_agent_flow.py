"""Query flow inside the agent: one planning call, local tools, one answer call.

Run: python agent_flow.py
Writes report/figures/agent_flow.pdf.
"""

from diagram_style import draw_arrow, draw_box, draw_group, draw_label, new_figure, save

ROW_WIDTH, ROW_HEIGHT = 3.2, 1.5
TOOL_WIDTH, TOOL_HEIGHT = 2.8, 1.35
COLUMNS = [1.8, 5.9, 10.0, 14.1]
TOOL_COLUMNS = [1.8, 4.9, 8.0, 11.1, 14.2]
PLANNING_ROW, TOOLS_ROW, ANSWER_ROW = 7.6, 4.7, 2.0

figure, axes = new_figure(16.0, 8.8)

# Planning
user_message = draw_box(axes, COLUMNS[0], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                        "User message", "named titles and characters\npreloaded; up to 12 recent\nmessages kept")
routing = draw_box(axes, COLUMNS[1], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                   "Semantic routing", "top 4 tool guides and\ntop 10 genre/tag candidates\nby embedding similarity")
planner = draw_box(axes, COLUMNS[2], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                   "Planner LLM", "call 1: one JSON plan:\ncontext choice and\nup to 8 tool steps", kind="llm")
validation = draw_box(axes, COLUMNS[3], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                      "Schema validation", "invalid plan: tools skipped,\nreported as missing evidence")

# Local tools
tools = [
    ("resolve_entity", "names and aliases\nto IDs (SQL)"),
    ("get_anime\nget_character", "profiles and cast (SQL)"),
    ("recommend_anime\npopular", "SQL candidates,\nembedding re-rank"),
    ("recommend_by_\ncharacter", "biography embeddings"),
    ("search_passages", "BM25, vector fallback"),
]
for column, (name, detail) in zip(TOOL_COLUMNS, tools):
    draw_box(axes, column, TOOLS_ROW, TOOL_WIDTH, TOOL_HEIGHT, name, detail, kind="tool",
             title_family="monospace")
tools_bottom, tools_top = TOOLS_ROW - TOOL_HEIGHT / 2 - 0.2, TOOLS_ROW + TOOL_HEIGHT / 2 + 0.55
draw_group(axes, 0.15, tools_bottom, 15.85, tools_top,
           "Local tools, no LLM calls: later steps can reuse the results of earlier ones")

# Answering
evidence = draw_box(axes, COLUMNS[0], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                    "Evidence", "tool results, capped\nat 14,000 characters")
answer = draw_box(axes, COLUMNS[1], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                  "Answer LLM", "call 2: grounded reply,\nmissing information or\na clarifying question", kind="llm")
identity = draw_box(axes, COLUMNS[2], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                    "Identity check", "names in the reply match\nthe retrieved IDs")
reply = draw_box(axes, COLUMNS[3], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                 "Reply", "shown in the chat")

# Flow
draw_arrow(axes, user_message.right, routing.left)
draw_arrow(axes, routing.right, planner.left)
draw_arrow(axes, planner.right, validation.left)
draw_arrow(axes, validation.bottom, (validation.x, tools_top), "valid plan", label_position="right")
draw_arrow(axes, (evidence.x, tools_bottom), evidence.top)
draw_arrow(axes, evidence.right, answer.left)
draw_arrow(axes, answer.right, identity.left)
draw_arrow(axes, identity.right, reply.left, "pass")
draw_arrow(axes, identity.bottom, answer.bottom, dashed=True, curvature=-0.3)
draw_label(axes, (identity.x + answer.x) / 2, identity.bottom[1] - 0.66, "mismatch: one correction call\n(still wrong: error)", "below", 0)

save(figure, "agent_flow")
