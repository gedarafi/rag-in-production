from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import requests

DATASET_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"
DOCUMENTS_MEMBER = "multidoc2dial/multidoc2dial_doc.json"
VALIDATION_MEMBER = "multidoc2dial/multidoc2dial_dial_validation.json"
EVALUATION_FILE = Path("data/evaluation/multidoc2dial_validation.jsonl")
QUESTION_DIALOGUE_ACTS = {"query_condition", "query_solution"}


def download_archive() -> zipfile.ZipFile:
    try:
        response = requests.get(DATASET_URL, timeout=120)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Could not download MultiDoc2Dial from {DATASET_URL}") from error
    return zipfile.ZipFile(io.BytesIO(response.content))


def document_filename(doc_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", doc_id.lower()).strip("-") + ".md"


def write_documents(documents_by_domain: dict, documents_folder: Path) -> dict[str, str]:
    source_by_doc_id: dict[str, str] = {}
    for domain, documents in documents_by_domain.items():
        domain_folder = documents_folder / "multidoc2dial" / domain
        domain_folder.mkdir(parents=True, exist_ok=True)
        for doc_id, document in documents.items():
            path = domain_folder / document_filename(doc_id)
            source = str(path.relative_to(documents_folder))
            if source in source_by_doc_id.values():
                raise RuntimeError(f"Two documents map to the same file: {source}")
            path.write_text(f"# {document['title']}\n\n{document['doc_text']}\n", encoding="utf-8")
            source_by_doc_id[doc_id] = source
    return source_by_doc_id


def evaluation_questions(dialogues_by_domain: dict, source_by_doc_id: dict[str, str]) -> list[dict]:
    questions: list[dict] = []
    for domain, dialogues in dialogues_by_domain.items():
        for dialogue in dialogues:
            history: list[str] = []
            for turn in dialogue["turns"]:
                if turn["role"] == "user" and turn["da"] in QUESTION_DIALOGUE_ACTS and turn["references"]:
                    gold_doc_id = turn["references"][0]["doc_id"]
                    questions.append(
                        {
                            "dialogue_id": dialogue["dial_id"],
                            "turn_id": turn["turn_id"],
                            "domain": domain,
                            "question": turn["utterance"],
                            "history": list(history),
                            "gold_doc_id": gold_doc_id,
                            "gold_source": source_by_doc_id[gold_doc_id],
                        }
                    )
                history.append(f"{turn['role']}: {turn['utterance']}")
    return questions


def prepare(documents_folder: Path, evaluation_file: Path) -> tuple[int, int]:
    archive = download_archive()
    documents_by_domain = json.loads(archive.read(DOCUMENTS_MEMBER))["doc_data"]
    dialogues_by_domain = json.loads(archive.read(VALIDATION_MEMBER))["dial_data"]

    source_by_doc_id = write_documents(documents_by_domain, documents_folder)
    questions = evaluation_questions(dialogues_by_domain, source_by_doc_id)
    evaluation_file.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_file.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(json.dumps(question, ensure_ascii=False) + "\n")
    return len(source_by_doc_id), len(questions)
