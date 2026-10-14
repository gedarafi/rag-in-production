"""Tags, characters, adaptations and provenance-bearing review imports."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .vndb import plain, rows


def optional_rows(source, table):
    if (source / "db" / table).exists():
        yield from rows(source, table)


def load_context(source: Path, novels: dict, title, spoiler_level: int, reviews: Path | None):
    sections = defaultdict(list)
    documents = {}
    metadata = {vid: {"tags": [], "characters": [], "anime": [], "reviews": []} for vid in novels}
    tags = {r["id"]: r for r in optional_rows(source, "tags")}
    parents = defaultdict(list)
    for r in optional_rows(source, "tags_parents"):
        if r["parent"] in tags:
            parents[r["id"]].append(tags[r["parent"]]["name"])
    # These are raw user assertions, not VNDB's official weighted tag scores.
    votes = {}
    for r in optional_rows(source, "tags_vn"):
        if r["vid"] not in novels or r["tag"] not in tags or r["ignore"] == "t" or r["lie"] == "t":
            continue
        tag = tags[r["tag"]]
        key = (r["vid"], r["tag"])
        stats = votes.setdefault(key, [0, 0, 0])
        stats[0] += int(r["vote"])
        stats[1] += 1
        stats[2] = max(stats[2], int(r["spoiler"] if r["spoiler"] is not None else tag["defaultspoil"]))
    for (vid, tid), (total, count, spoil) in sorted(votes.items()):
        if total <= 0 or spoil > spoiler_level:
            continue
        tag = tags[tid]
        item = {"id": tid, "name": tag["name"], "category": tag["cat"], "parents": sorted(parents[tid]),
                "spoiler_level": spoil, "raw_vote_mean": round(total / count, 3), "vote_count": count}
        metadata[vid]["tags"].append(item)
        sections[vid].append(f"- {title(vid)}: tag {tag['name']} ({tid}); category: {tag['cat']}; hierarchy: {', '.join(item['parents'])}; spoiler level: {spoil}; raw mean vote: {total / count:.2f} ({count} assertions, not official relevance); definition: tag-{tid}.md; https://vndb.org/{tid}")
        if f"tag-{tid}.md" not in documents:
            documents[f"tag-{tid}.md"] = (f"# VNDB tag: {tag['name']} ({tid})\nSource: https://vndb.org/{tid}\n"
                                          f"Category: {tag['cat']}\nParents: {', '.join(item['parents'])}\nAliases: {tag['alias']}\n\n{plain(tag['description'])}\n")

    anime = {r["id"]: r for r in optional_rows(source, "anime")}
    adaptations = defaultdict(list)
    for r in optional_rows(source, "vn_anime"):
        a = anime.get(r["aid"])
        if r["id"] not in novels or not a:
            continue
        item = {"anidb_id": a["id"], "title": a["title_romaji"], "original_title": a["title_kanji"], "type": a["type"], "year": a["year"]}
        metadata[r["id"]]["anime"].append(item)
        adaptations[r["id"]].append(f"- Anime linked to {title(r['id'])}: {a['title_romaji']} / {a['title_kanji'] or ''}; type: {a['type']}; year: {a['year'] or 'Unknown'}; metadata: AniDB (CC-BY-NC-SA 4.0); https://anidb.net/anime/{a['id']}")

    chars = {r["id"]: r for r in optional_rows(source, "chars")}
    names = defaultdict(list)
    for r in optional_rows(source, "chars_names"):
        names[r["id"]].append(r)
    aliases = defaultdict(list)
    for r in optional_rows(source, "chars_alias"):
        if int(r["spoil"]) <= spoiler_level:
            aliases[r["id"]].append(r["latin"] or r["name"])
    traits = {r["id"]: r for r in optional_rows(source, "traits")}
    char_traits = defaultdict(list)
    for r in optional_rows(source, "chars_traits"):
        trait = traits.get(r["tid"])
        if trait and r["lie"] != "t" and max(int(r["spoil"]), int(trait["defaultspoil"])) <= spoiler_level:
            group = traits.get(trait["gid"], {})
            char_traits[r["id"]].append(f"{trait['name']} ({r['tid']}; group: {group.get('name', 'unspecified')}; definition: trait-{r['tid']}.md)")
            if f"trait-{r['tid']}.md" not in documents:
                documents[f"trait-{r['tid']}.md"] = (f"# Character trait: {trait['name']} ({r['tid']})\nSource: https://vndb.org/{r['tid']}\n"
                                                   f"Group: {group.get('name', 'unspecified')}\nAliases: {trait.get('alias') or ''}\n\n{plain(trait['description'])}\n")
    links = defaultdict(list)
    for r in optional_rows(source, "chars_vns"):
        if r["vid"] in novels and r["id"] in chars and int(r["spoil"]) <= spoiler_level:
            links[(r["id"], r["vid"])].append(r)
    staff = {r["aid"]: r for r in optional_rows(source, "staff_alias")}
    voices = defaultdict(list)
    for r in optional_rows(source, "vn_seiyuu"):
        person = staff.get(r["aid"])
        if person:
            voices[(r["cid"], r["id"])].append(f"{person['latin'] or person['name']} ({plain(r['note'])}); https://vndb.org/{person['id']}")
    for (cid, vid), memberships in sorted(links.items()):
        char = chars[cid]
        candidates = sorted(names[cid], key=lambda r: (r["lang"] != "en", r["lang"]))
        name = (candidates[0]["latin"] or candidates[0]["name"]) if candidates else cid
        roles = sorted({r["role"] for r in memberships})
        metadata[vid]["characters"].append({"id": cid, "name": name, "roles": roles})
        lines = [f"# {name} — character in {title(vid)} ({vid})", f"Source: https://vndb.org/{cid}",
                 f"Visual novel: {title(vid)}; https://vndb.org/{vid}", f"Roles: {', '.join(roles)}",
                 f"Release IDs: {', '.join(sorted({r['rid'] for r in memberships if r['rid']})) or 'Unspecified'}",
                 f"Names: {'; '.join((r['latin'] or r['name']) + ' / ' + r['name'] + ' [' + r['lang'] + ']' for r in candidates)}",
                 f"Aliases: {'; '.join(sorted(set(aliases[cid]))) or 'None included'}"]
        for field in ("age", "height", "weight", "bloodt", "birthday"):
            if char.get(field) not in (None, "", "0", "unknown"):
                lines.append(f"{field}: {char[field]}")
        # Never expose alternate/spoiler identity fields such as main or spoil_sex.
        for field in ("sex", "gender"):
            if char.get(field) and char.get(f"spoil_{field}") in (None, ""):
                lines.append(f"{field}: {char[field]}")
        lines += ["", f"## Description — {name} in {title(vid)}", plain(char["description"]) or "No description available.",
                  "", f"## Traits — {name} in {title(vid)}", *[f"- {name} in {title(vid)}: {t}" for t in sorted(set(char_traits[cid]))]]
        if voices[(cid, vid)]:
            lines += ["", f"## Voice cast — {name} in {title(vid)}", *[f"- {name} in {title(vid)}: {voice}" for voice in sorted(set(voices[(cid, vid)]))]]
        documents[f"character-{cid}-{vid}.md"] = "\n".join(lines) + "\n"

    review_sections = defaultdict(list)
    if reviews:
        seen = set()
        with reviews.open(encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                required = ("id", "vn_id", "text", "source_url", "author", "date", "spoiler_level")
                if not isinstance(r, dict) or any(key not in r for key in required):
                    raise ValueError(f"Review line {number}: required fields are {required}")
                if any(not isinstance(r[key], str) or not r[key].strip() for key in required[:-1]):
                    raise ValueError(f"Review line {number}: required text fields must be nonempty strings")
                if type(r["spoiler_level"]) is not int or r["spoiler_level"] not in (0, 1, 2):
                    raise ValueError(f"Review line {number}: spoiler_level must be 0, 1 or 2")
                if not re_safe_id(r["id"]) or r["id"] in seen or r["vn_id"] not in novels:
                    raise ValueError(f"Review line {number}: invalid/duplicate ID or unknown vn_id")
                if urlparse(r["source_url"]).scheme not in ("https", "http") or not urlparse(r["source_url"]).netloc:
                    raise ValueError(f"Review line {number}: source_url must be an HTTP(S) URL")
                seen.add(r["id"])
                if r["spoiler_level"] > spoiler_level:
                    continue
                vid = r["vn_id"]
                metadata[vid]["reviews"].append({key: r[key] for key in required if key != "text"})
                review_sections[vid].append(f"- Review {r['id']} by {r['author']}: {r['source_url']}")
                documents[f"review-{r['id']}.md"] = (f"# Review of {title(vid)} ({vid})\n\n"
                    f"Reader opinion, not a canonical story fact.\nAuthor: {r['author']}\nDate: {r['date']}\n"
                    f"Source: {r['source_url']}\nSpoiler level: {r['spoiler_level']}\n\n{plain(r['text'])}\n")
    return sections, adaptations, review_sections, documents, metadata


def re_safe_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
