"""Convert an extracted VNDB COPY dump into joined, readable RAG documents.

No database server or third-party dependencies are required. Table headers are
read from the dump rather than assuming a particular column order.
"""
from __future__ import annotations

import json
import re
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from .datasets import download

VNDB_URL = "https://dl.vndb.org/dump/vndb-db-latest.tar.zst"


def decode_copy(value: str) -> str | None:
    if value == r"\N":
        return None
    escapes = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}
    return re.sub(r"\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)",
                  lambda m: chr(int(m[1][1:], 16)) if m[1].startswith("x") and len(m[1]) > 1 else
                  chr(int(m[1], 8)) if m[1][0] in "01234567" else escapes.get(m[1], m[1]), value)


def rows(source: Path, table: str):
    columns = (source / "db" / f"{table}.header").read_text(encoding="utf-8").strip().split("\t")
    with (source / "db" / table).open(encoding="utf-8") as file:
        for number, line in enumerate(file, 1):
            values = line.rstrip("\r\n").split("\t")
            if len(values) != len(columns):
                raise ValueError(f"{table}:{number}: expected {len(columns)} fields, got {len(values)}")
            yield dict(zip(columns, map(decode_copy, values)))


def plain(value: str | None) -> str:
    text = value or ""
    text = re.sub(r"\[url=([^\]]+)\](.*?)\[/url\]", r"\2 (\1)", text, flags=re.S | re.I)
    text = re.sub(r"\[spoiler\].*?\[/spoiler\]", "[spoiler omitted]", text, flags=re.S | re.I)
    return re.sub(r"\[/?(?:b|i|u|s|quote|code|raw|url)\]", "", text, flags=re.I).strip()


def release_date(value: str | None) -> str:
    if not value or value == "0":
        return "Unknown"
    if value == "99999999":
        return "TBA"
    year, month, day = value[:4], value[4:6], value[6:8]
    return year + (f"-{month}" if month != "99" else "") + (f"-{day}" if month != "99" and day != "99" else "")


def prepare_vndb(root: Path, source: Path | None = None, *, spoiler_level: int = 0, reviews: Path | None = None) -> int:
    if spoiler_level not in (0, 1, 2):
        raise ValueError("spoiler_level must be 0, 1 or 2")
    if source is None:
        candidates = (root / "data-raw/vndb", root / "data/raw/vndb")
        source = next((path for path in candidates if (path / "db/vn").is_file()), None)
        if source is None:
            source = root / "data/raw/vndb-db-latest.tar.zst"
            if not source.is_file():
                # Publish the cached download only after it completes.
                partial = source.with_name(source.name + ".part")
                download(VNDB_URL, partial)
                partial.replace(source)
    if source.is_file():
        with tempfile.TemporaryDirectory(prefix="vndb-") as folder:
            extracted = Path(folder)
            # Stream decompression; Python 3.14 supports Zstandard natively.
            # The data filter rejects traversal and links outside this directory.
            with tarfile.open(source, mode="r|*") as archive:
                archive.extractall(extracted, filter="data")
            candidates = [path.parent.parent for path in extracted.rglob("vn")
                          if path.is_file() and path.parent.name == "db"]
            if len(candidates) != 1:
                raise ValueError("VNDB archive must contain exactly one dump directory with db/vn.")
            return _prepare_vndb(root, candidates[0], spoiler_level=spoiler_level,
                                 reviews=reviews, source_origin=source)
    return _prepare_vndb(root, source, spoiler_level=spoiler_level, reviews=reviews)


def _prepare_vndb(root: Path, source: Path, *, spoiler_level: int,
                  reviews: Path | None, source_origin: Path | None = None) -> int:
    if not (source / "db/vn").is_file():
        raise ValueError("VNDB source must be a dump archive or an extracted directory containing db/vn and .header files.")
    novels = {r["id"]: r for r in rows(source, "vn")}
    if not novels or any(not re.fullmatch(r"v[0-9]+", identifier or "") for identifier in novels):
        raise ValueError("VNDB dump contains no novels or invalid VN identifiers.")
    titles = defaultdict(list)
    for row in rows(source, "vn_titles"):
        titles[row["id"]].append(row)

    def title(identifier):
        candidates = sorted(titles[identifier], key=lambda r: (r["lang"] != "en", r["lang"] != novels.get(identifier, {}).get("olang"), r["lang"]))
        return (candidates[0]["latin"] or candidates[0]["title"]) if candidates else identifier

    release_vns = defaultdict(list)
    for row in rows(source, "releases_vn"):
        if row["vid"] in novels:
            release_vns[row["id"]].append((row["vid"], row["rtype"]))
    releases = {r["id"]: r for r in rows(source, "releases") if r["id"] in release_vns}
    engines = {r["id"]: r for r in rows(source, "engines")} if (source / "db/engines").exists() else {}
    release_titles = defaultdict(list)
    platforms = defaultdict(set)
    producers = {r["id"]: r for r in rows(source, "producers")}
    release_producers = defaultdict(list)
    for r in rows(source, "releases_titles"):
        release_titles[r["id"]].append(r)
    for r in rows(source, "releases_platforms"):
        platforms[r["id"]].add(r["platform"])
    for r in rows(source, "releases_producers"):
        producer = producers.get(r["pid"])
        if producer:
            roles = [role for role in ("developer", "publisher") if r[role] == "t"]
            release_producers[r["id"]].append(f"{producer['latin'] or producer['name']} ({', '.join(roles)}; https://vndb.org/{r['pid']})")
    details = defaultdict(list)
    for rid, memberships in release_vns.items():
        r = releases.get(rid)
        if not r:
            continue
        names = sorted(release_titles[rid], key=lambda t: (t["lang"] != "en", t["lang"]))
        names_text = "; ".join(f"{t['latin'] or t['title'] or title(memberships[0][0])} [{t['lang']}, {'machine translated' if t['mtl'] == 't' else 'not marked machine translated'}]" for t in names)
        for vid, kind in memberships:
            details[vid].append(f"- {title(vid)} — release {rid}: {names_text or 'Title unspecified'}; date: {release_date(r['released'])}; type: {kind}; platforms: {', '.join(sorted(platforms[rid])) or 'Unknown'}; official: {r['official'] == 't'}; patch: {r['patch'] == 't'}; freeware: {r['freeware'] == 't'}; minimum age: {r['minage'] or 'Unknown'}; producers: {'; '.join(sorted(release_producers[rid])) or 'Unknown'}; source: https://vndb.org/{rid}")
            if r.get("notes"):
                details[vid].append(f"- {title(vid)} release {rid} notes: {plain(r['notes'])}")
            engine = engines.get(r.get("engine"))
            if engine:
                details[vid].append(f"- {title(vid)} release {rid} engine: {engine['name']}; {plain(engine['description'])}")
    aliases = {r["aid"]: r for r in rows(source, "staff_alias")}
    credits = defaultdict(list)
    for r in rows(source, "vn_staff"):
        person = aliases.get(r["aid"])
        if person:
            credits[r["id"]].append(f"- {title(r['id'])}: {person['latin'] or person['name']} — {r['role']}; edition: {r['eid'] or 'all/unspecified'}; {plain(r['note'])}; https://vndb.org/{person['id']}")
    relations = defaultdict(list)
    relation_names = {"seq": "Sequel", "preq": "Prequel", "set": "Same setting", "alt": "Alternative version",
                      "char": "Shares characters", "side": "Side story", "par": "Parent story", "ser": "Same series",
                      "fan": "Fan work", "orig": "Original work"}
    for r in rows(source, "vn_relations"):
        relations[r["id"]].append(f"- {title(r['id'])}: {relation_names.get(r['relation'], r['relation'])} — {title(r['vid'])} ({r['vid']}); official: {r['official'] == 't'}; https://vndb.org/{r['vid']}")
    timestamp = (source / "TIMESTAMP").read_text(encoding="utf-8").strip()
    from .vndb_context import load_context
    tags, adaptations, review_links, extra_documents, catalog = load_context(source, novels, title, spoiler_level, reviews)
    output = root / "data/documents/vndb"
    output.mkdir(parents=True, exist_ok=True)
    lengths = {"0": "Unknown", "1": "Very short (<2 hours)", "2": "Short (2–10 hours)", "3": "Medium (10–30 hours)", "4": "Long (30–50 hours)", "5": "Very long (>50 hours)"}
    for identifier, vn in novels.items():
        lines = [f"# {title(identifier)}", "", f"VNDB ID: {identifier}", f"Source: https://vndb.org/{identifier}", f"Dump timestamp: {timestamp}", f"Original language: {vn['olang']}", f"Development status: { {'0': 'Finished', '1': 'In development', '2': 'Cancelled'}.get(vn['devstatus'], vn['devstatus'])}", f"Aliases: {vn['alias'] or 'None listed'}", f"Length category: {lengths.get(vn['length'], vn['length'])}"]
        for label, key in (("Bayesian rating", "c_rating"), ("Average rating", "c_average")):
            if vn[key] is not None:
                lines.append(f"{label}: {int(vn[key]) / 100:.2f}/10 ({vn['c_votecount']} votes)")
        if vn["c_length"] is not None:
            lines.append(f"Reported play time: {int(vn['c_length']) / 60:.1f} hours ({vn['c_lengthnum']} reports)")
        lines += ["", "## Titles", *[f"- {t['title']} ({t['lang']}); romanization: {t['latin'] or 'None'}; official: {t['official'] == 't'}" for t in titles[identifier]], "", f"## Synopsis — {title(identifier)}", plain(vn["description"]) or "No description provided."]
        characters = [f"- {c['name']} ({c['id']}): {', '.join(c['roles'])}; details in character-{c['id']}-{identifier}.md; https://vndb.org/{c['id']}" for c in catalog[identifier]["characters"]]
        if characters:
            lines += ["", f"## Characters — {title(identifier)}", *characters]
        for heading, data in (("Tags and setting/theme descriptors", tags), ("Linked anime", adaptations),
                              ("Reviews", review_links), ("Releases", details), ("Staff credits", credits), ("Related visual novels", relations)):
            if data[identifier]:
                lines += ["", f"## {heading} — {title(identifier)}", *sorted(set(data[identifier]))]
        target = output / f"{identifier}.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for filename, content in extra_documents.items():
        (output / filename).write_text(f"Dump timestamp: {timestamp}\nExport spoiler threshold: {spoiler_level}\n\n" + content, encoding="utf-8")
    # Refresh owned auxiliary documents when links disappear or spoiler policy changes.
    for pattern in ("character-c*-v*.md", "review-*.md", "tag-g*.md", "trait-i*.md"):
        for path in output.glob(pattern):
            if path.name not in extra_documents:
                path.unlink()
    catalog_releases = defaultdict(list)
    for rid, memberships in release_vns.items():
        if rid not in releases:
            continue
        for vid, kind in memberships:
            catalog_releases[vid].append({"id": rid, "type": kind, "date": release_date(releases[rid]["released"]),
                                         "official": releases[rid]["official"] == "t", "platforms": sorted(platforms[rid]),
                                         "patch": releases[rid]["patch"] == "t", "freeware": releases[rid]["freeware"] == "t",
                                         "minimum_age": releases[rid]["minage"], "voiced_code": releases[rid].get("voiced"),
                                         "has_erotic_content": releases[rid].get("has_ero") == "t",
                                         "languages": sorted({t["lang"] for t in release_titles[rid]})})
    with (output / "catalog.jsonl").open("w", encoding="utf-8") as file:
        for vid, item in catalog.items():
            item.update({"vn_id": vid, "title": title(vid), "original_language": novels[vid]["olang"],
                         "rating": int(novels[vid]["c_rating"]) / 100 if novels[vid]["c_rating"] is not None else None,
                         "rating_votes": int(novels[vid]["c_votecount"]),
                         "play_time_minutes": int(novels[vid]["c_length"]) if novels[vid]["c_length"] is not None else None,
                         "development_status": novels[vid]["devstatus"],
                         "spoiler_threshold": spoiler_level, "timestamp": timestamp})
            item["releases"] = catalog_releases[vid]
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    # Only remove converter-owned VN documents absent from the current snapshot.
    for path in output.glob("v*.md"):
        if re.fullmatch(r"v[0-9]+", path.stem) and path.stem not in novels:
            path.unlink()
    count = len(novels) + len(extra_documents)
    manifest = {"timestamp": timestamp, "documents": count, "visual_novels": len(novels), "character_documents": sum(name.startswith("character-") for name in extra_documents),
                "review_documents": sum(name.startswith("review-") for name in extra_documents), "spoiler_threshold": spoiler_level,
                "definition_documents": sum(name.startswith(("tag-", "trait-")) for name in extra_documents),
                "reviews_source": str(reviews.resolve()) if reviews else None, "reviews_in_dump": False,
                "source": str((source_origin or source).resolve()), "excluded": ["user lists", "individual vote identities", "images"],
                "spoiler_caveat": "Only explicit markup and database spoiler flags are filtered. Unmarked descriptions can still reveal story details.",
                "tag_method": "Positive raw mean of nonignored, non-lie assertions; maximum reported/default spoiler flag; not official VNDB weighted scores.",
                "license": "See original dump README.txt and license files; descriptions may have separate terms. Anime metadata: AniDB CC-BY-NC-SA."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return count
