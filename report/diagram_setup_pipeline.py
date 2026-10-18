"""Setup pipeline of the agent: how each AniList record becomes facts, text and labels for the agent.

Run: python diagram_setup_pipeline.py
Writes report/figures/setup_pipeline.pdf.
"""

from diagram_style import draw_arrow, draw_box, draw_group, draw_label, new_figure, save

INPUT_WIDTH, WIDTH, HEIGHT = 2.6, 3.2, 1.3
INPUT_COLUMN = 1.45
COLUMNS = [5.55, 9.65, 13.75]
FACTS_ROW, TEXT_ROW, LABELS_ROW = 7.0, 4.2, 1.4
LANE_LEFT, LANE_RIGHT = 3.6, 15.95

figure, axes = new_figure(16.0, 8.5)

for row, label in ((FACTS_ROW, "Catalogue: exact look-ups and keyword search"),
                   (TEXT_ROW, "Documents: search by meaning"),
                   (LABELS_ROW, "Labels: used to plan")):
    draw_group(axes, LANE_LEFT, row - 0.9, LANE_RIGHT, row + 1.15, label)

record = draw_box(axes, INPUT_COLUMN, 5.6, INPUT_WIDTH, HEIGHT, "AniList record", "one anime: text,\ngenres, tags, facts")
prepare = draw_box(axes, INPUT_COLUMN, 2.8, INPUT_WIDTH, HEIGHT, "Prepare", "splits each record\nby kind of content")

# Facts
fields = draw_box(axes, COLUMNS[0], FACTS_ROW, WIDTH, HEIGHT, "Structured fields", "dates, episodes,\nstudios, cast, aliases")
catalogue = draw_box(axes, COLUMNS[1], FACTS_ROW, WIDTH, HEIGHT, "SQLite catalogue", "exact look-ups")
full_text = draw_box(axes, COLUMNS[2], FACTS_ROW, WIDTH, HEIGHT, "Full-text index", "synopses and\nbiographies (BM25)")

# Text
descriptions = draw_box(axes, COLUMNS[0], TEXT_ROW, WIDTH, HEIGHT, "Anime documents", "synopsis, tags and\ncharacters as text")
chunks = draw_box(axes, COLUMNS[1], TEXT_ROW, WIDTH, HEIGHT, "Chunks and tokens", "overlapping chunks,\nWordPiece tokens")
text_vectors = draw_box(axes, COLUMNS[2], TEXT_ROW, WIDTH, HEIGHT, "Embeddings", "MiniLM vectors")

# Labels
labels = draw_box(axes, COLUMNS[0], LABELS_ROW, WIDTH, HEIGHT, "Labels and tools", "genre and tag names,\ntool descriptions")
label_vectors = draw_box(axes, COLUMNS[1], LABELS_ROW, WIDTH, HEIGHT, "Embeddings", "same encoder")
routing = draw_box(axes, COLUMNS[2], LABELS_ROW, WIDTH, HEIGHT, "Routing index", "closest tools and\ntags for a message")

draw_arrow(axes, record.bottom, prepare.top)
for target in (fields, descriptions, labels):
    draw_arrow(axes, prepare.right, target.left)
for first, second in ((fields, catalogue), (catalogue, full_text), (descriptions, chunks), (chunks, text_vectors),
                      (labels, label_vectors), (label_vectors, routing)):
    draw_arrow(axes, first.right, second.left)

draw_label(axes, 8.0, 0.35, "Built once at setup and cached; later launches reuse it.", "below", 0)

save(figure, "setup_pipeline")
