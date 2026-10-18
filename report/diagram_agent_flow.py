"""Query flow inside the agent: LLM interpretation and planning, local tools, then an LLM answer
or a recommendation list rendered from catalogue records.

Run: python diagram_agent_flow.py
Writes figures/agent_flow.pdf (see FIGURES in diagram_style.py).
"""

from diagram_style import draw_arrow, draw_box, draw_group, draw_label, draw_path, new_figure, save

ROW_WIDTH, ROW_HEIGHT = 3.2, 1.5
TOOL_WIDTH, TOOL_HEIGHT = 2.8, 1.45
COLUMNS = [1.8, 5.9, 10.0, 14.1]
TOOL_COLUMNS = [1.8, 4.9, 8.0, 11.1, 14.2]
PLANNING_ROW, TOOLS_ROW, ANSWER_ROW, RENDER_ROW = 7.8, 5.175, 2.9, 0.8
RENDER_HEIGHT = 1.2

figure, axes = new_figure(16.0, 8.8)

# Interpretation and planning
user_message = draw_box(axes, COLUMNS[0], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                        "User message", "with up to 12 recent\nmessages (8,000 characters)")
interpretation = draw_box(axes, COLUMNS[1], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                          "Interpretation (LLM)", "names typed and matched to\ncatalogue IDs; criteria to\nfilters or preferences",
                          kind="llm")
planner = draw_box(axes, COLUMNS[2], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                   "Planning (LLM)", "one JSON plan, up to 8\ntool steps, from all tools\nand 10 nearest genres/tags",
                   kind="llm")
validation = draw_box(axes, COLUMNS[3], PLANNING_ROW, ROW_WIDTH, ROW_HEIGHT,
                      "Schema validation", "invalid plan: tools skipped,\nreported as missing evidence")

# Local tools: eight tools in five boxes
tools = [
    ("resolve_entity\nsearch_features", "names to IDs (SQL),\nthemes to tag IDs"),
    ("get_anime\nget_character", "profiles and cast (SQL)"),
    ("recommend_anime\npopular", "SQL filters and\nembedding re-rank"),
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
                  "Answer (LLM)", "facts for each subject,\nexpressions for numbers,\nor a clarifying question",
                  kind="llm")
checks = draw_box(axes, COLUMNS[2], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                  "Answer checks", "a fact for every subject,\narithmetic run by code,\nnames match retrieved IDs")
reply = draw_box(axes, COLUMNS[3], ANSWER_ROW, ROW_WIDTH, ROW_HEIGHT,
                 "Reply", "shown in the chat")
render_x = (COLUMNS[1] + COLUMNS[2]) / 2
rendered = draw_box(axes, render_x, RENDER_ROW, ROW_WIDTH, RENDER_HEIGHT,
                    "Recommendation list", "rendered from verified\ncatalogue records, no LLM")

# Flow
draw_arrow(axes, user_message.right, interpretation.left)
draw_arrow(axes, interpretation.right, planner.left)
draw_arrow(axes, planner.right, validation.left)
draw_arrow(axes, validation.bottom, (validation.x, tools_top), "valid plan", label_position="right")
draw_arrow(axes, (evidence.x, tools_bottom), evidence.top)
draw_arrow(axes, evidence.right, answer.left)
draw_arrow(axes, answer.right, checks.left)
draw_arrow(axes, checks.right, reply.left, "pass")
draw_arrow(axes, checks.bottom, answer.bottom, dashed=True, curvature=-0.12)
draw_label(axes, (checks.x + answer.x) / 2, answer.bottom[1] - 0.29,
           "fails: one correction call (fails again: error)", "below", 0)
draw_path(axes, [evidence.bottom, (evidence.x, rendered.y), rendered.left],
          "recommendations", label_at=(evidence.x + 0.12, (evidence.bottom[1] + rendered.y) / 2),
          label_position="right", dashed=False)
draw_path(axes, [rendered.right, (reply.x, rendered.y), reply.bottom], dashed=False)

save(figure, "agent_flow")
