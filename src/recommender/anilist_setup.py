"""Resumable build and readiness checks of the AniList catalog, index and dictionaries.

`recommender build` runs it offline; artifacts.py then bundles the result for the worker Lambda.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from .chunk_index import DATABASE, VECTORS, ChunkIndex
from .config import Settings
from .datasets import ANILIST_FILENAME, ANILIST_URL, prepare_anilist
from .entity_agent import EntityAgent
from .rag import RagPipeline
from .sqlite_catalog import SCHEMA_VERSION

STEPS = (
    ('dataset', 'Dataset download'),
    ('documents', 'Prepare anime documents'), ('sql', 'SQL catalog and dictionaries'),
    ('embeddings', 'Document embeddings'), ('tools', 'Routing and feature embeddings'),
    ('verify', 'Verify workspace'),
)


class AniListWorkspace:
    def __init__(self, settings: Settings):
        if settings.documents != settings.root / 'data/documents/anilist':
            raise ValueError('The AniList workspace requires AniList settings.')
        self.settings = settings
        self.manifest = settings.index / 'setup.json'
        self.pipeline = None
        self.lock = threading.Lock()
        self.ready = self.is_prepared()
        self.steps = {key: {'label': label, 'status': 'ready' if self.ready else 'pending',
                            'message': '', 'elapsed_seconds': None} for key, label in STEPS}
        self.inspect_existing()
        self.message = 'Workspace prepared.' if self.ready else 'Run `recommender build` to prepare the AniList library.'

    def signature(self):
        s = self.settings
        paths = [s.documents / 'catalog.jsonl', s.index / DATABASE, s.index / VECTORS]
        return {'version': 1, 'sql_schema': SCHEMA_VERSION, 'embedding': s.embedding,
                'embedding_cache_version': s.embedding_cache_version,
                'chunk_size': s.chunk_size, 'chunk_overlap': s.chunk_overlap,
                'files': {str(p.relative_to(s.root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}

    def is_prepared(self):
        try:
            required = [self.settings.documents / 'catalog.sqlite3', self.settings.index / 'entity-tools.sqlite3',
                        self.settings.index / 'entity-features.sqlite3', self.settings.index / 'anime-texts.sqlite3']
            saved = json.loads(self.manifest.read_text(encoding='utf-8'))
            # Older manifests included generation settings. Neither changes the
            # local catalog or embedding index; connection errors surface at chat.
            if not isinstance(saved, dict):
                return False
            saved = {k: v for k, v in saved.items() if k not in ('llm', 'ollama_url')}
            return (all(p.is_file() and p.stat().st_size for p in required)
                    and saved == self.signature())
        except (OSError, ValueError, KeyError):
            return False

    def snapshot(self):
        return {'ready': self.ready, 'message': self.message, 'steps': deepcopy(self.steps)}

    def inspect_existing(self):
        """Inventory artifacts without treating file existence as verification."""
        s = self.settings
        groups = {
            'dataset': [s.root / 'data/raw' / ANILIST_FILENAME, s.root / 'data-raw' / ANILIST_FILENAME],
            'documents': [s.documents / 'catalog.jsonl'],
            'sql': [s.documents / 'catalog.sqlite3'],
            'embeddings': [s.index / DATABASE, s.index / VECTORS],
            'tools': [s.index / 'entity-tools.sqlite3', s.index / 'entity-features.sqlite3', s.index / 'anime-texts.sqlite3'],
        }
        for key, paths in groups.items():
            found = [p.name for p in paths if p.is_file() and p.stat().st_size]
            if found:
                self.steps[key]['message'] = 'Found ' + ', '.join(found) + '; checking before reuse.'

    def save_index_checkpoint(self):
        """Keep index provenance even if a later setup stage is interrupted."""
        target = self.settings.index / 'embedding-state.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.signature()), encoding='utf-8')
        temporary.replace(target)

    def reusable_index(self):
        """Validate the index independently of the final setup-ready marker.

        It is reused only with the signature it was built with: other chunks or another model need a new
        index, which the content-addressed embeddings.sqlite3 makes a replay of unchanged vectors.
        """
        try:
            s = self.settings
            if not ChunkIndex.exists(s.index):
                return False
            self.check_index()
            signature = self.signature()
            # A successful embedding stage is reusable even if tools/verification
            # never finished and the final ready marker was deliberately removed.
            for path in (s.index / 'embedding-state.json', self.manifest):
                try:
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    saved = {k: v for k, v in saved.items() if k not in ('llm', 'ollama_url')}
                    if saved == signature:
                        return True
                except (OSError, ValueError, AttributeError):
                    pass
            return False
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, sqlite3.Error, RuntimeError):
            return False

    def check_index(self):
        """The passage index's texts and vectors agree, match the embedding model and are finite."""
        try:
            index = ChunkIndex(self.settings.index)
        except (OSError, ValueError, sqlite3.Error) as error:
            raise RuntimeError(f'The embedding index is unreadable ({error}). Select rebuild and retry.') from error
        vectors = index.vectors
        probe = self.pipeline.models.embed(['AniList workspace verification'])
        if not len(vectors) or vectors.shape[1] != probe.shape[1]:
            raise RuntimeError('The embedding index does not match its chunks or embedding model. Select rebuild and retry.')
        for start in range(0, len(vectors), 8192):
            if not np.isfinite(vectors[start:start + 8192]).all():
                raise RuntimeError('The embedding index contains invalid values. Select rebuild and retry.')
        return len(vectors)

    def acquire_dataset(self, source, rebuild, progress):
        if source:
            path = Path(source)
            if not path.is_file() or path.suffix.lower() not in ('.csv', '.zip'):
                raise ValueError('Choose a local AniList CSV or ZIP file.')
            # Retain a managed raw copy for later retries and tag definitions.
            target = self.settings.root / 'data/raw' / ANILIST_FILENAME
            if path.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.csv.part')
                try:
                    shutil.copyfile(path, temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            return target, f'Using {path.name}; saved a raw copy for future setup runs.'
        if (self.settings.documents / 'catalog.jsonl').is_file() and not rebuild:
            return None, 'Reusing the prepared library; no download needed.'
        candidates = [self.settings.root / 'data/raw' / ANILIST_FILENAME,
                      self.settings.root / 'data-raw' / ANILIST_FILENAME]
        existing = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
        if existing:
            return existing, f'Reusing {existing.name}.'
        target = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.csv.part')
        try:
            with requests.get(ANILIST_URL, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                total, completed = int(response.headers.get('Content-Length', 0)), 0
                with temporary.open('wb') as file:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            file.write(block)
                            completed += len(block)
                            progress('Downloading AniList dataset', completed, total)
                if not completed or total and completed != total:
                    raise RuntimeError('Dataset download is incomplete. Retry setup or upload a local CSV/ZIP.')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target, f'Downloaded {target.name}.'

    def verify(self):
        p = self.pipeline
        catalog = p.anime_tools()
        count = catalog.query('SELECT COUNT(*) FROM anime')[0][0]
        if not count:
            raise RuntimeError('The SQL anime catalog is empty.')
        if not catalog.query('SELECT COUNT(*) FROM entity_dictionary')[0][0]:
            raise RuntimeError('The entity dictionary is empty.')
        return f'{count:,} anime and {self.check_index():,} embedded passages verified.'

    def setup(self, source=None, rebuild=False, progress=None):
        """Yield stage snapshots; long-running operations also report byte/batch progress."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Setup or a chat request is already running.')
        progress = progress or (lambda *args: None)
        self.ready = False
        self.pipeline = RagPipeline(self.settings)
        self.steps = {key: {'label': label, 'status': 'pending', 'message': '', 'elapsed_seconds': None}
                      for key, label in STEPS}
        self.inspect_existing()
        dataset = None
        prepared_now = False
        try:
            # Preserve proven index provenance before removing the overall
            # readiness marker, including installations from before checkpoints.
            if self.is_prepared():
                self.save_index_checkpoint()
            # A partial rebuild must never inherit an earlier ready marker.
            self.manifest.unlink(missing_ok=True)
            for key, _ in STEPS:
                step = self.steps[key]
                step['status'] = 'running'
                self.message = step['label'] + '…'
                yield self.snapshot()
                started = perf_counter()
                try:
                    if key == 'dataset':
                        dataset, message = self.acquire_dataset(source, rebuild, progress)
                    elif key == 'documents':
                        if source or rebuild or not (self.settings.documents / 'catalog.jsonl').is_file():
                            count = prepare_anilist(self.settings.root, dataset)
                            prepared_now = True
                            message = f'Prepared {count:,} anime documents and the structured catalog.'
                        else:
                            message = 'Reusing existing documents and structured catalog.'
                    elif key == 'sql':
                        catalog = self.pipeline.anime_tools()
                        if catalog is None:
                            raise RuntimeError('No AniList catalog was prepared.')
                        self.pipeline._agent = EntityAgent(self.settings, self.pipeline.models, catalog, self.pipeline.vector_search)
                        message = f"{catalog.query('SELECT COUNT(*) FROM anime')[0][0]:,} anime indexed in SQL; entity dictionaries ready."
                    elif key == 'embeddings':
                        if rebuild or prepared_now or not self.reusable_index():
                            count = self.pipeline.ingest(progress=progress)
                            message = f'{count:,} passages indexed; unchanged embeddings reused from cache.'
                        else:
                            message = 'Reusing the verified document embedding index.'
                        self.save_index_checkpoint()
                    elif key == 'tools':
                        agent = self.pipeline._agent
                        agent.guides('Recommend anime')
                        agent.feature_candidates('Anime genres and themes')
                        # Only the fields of EntityStore.anime_text, not the whole records.
                        records = [{'title': title, 'description': description or '',
                                    'genres': json.loads(genres or '[]'), 'tags': json.loads(tags or '[]')}
                                   for title, description, genres, tags in agent.catalog.query(
                                       "SELECT json_extract(data,'$.title'), json_extract(data,'$.description'), "
                                       "json_extract(data,'$.genres'), json_extract(data,'$.tags') FROM anime ORDER BY id")]
                        agent.anime_vectors(records, progress)
                        message = f'Tool guides, genre/tag and {len(records):,} anime embeddings are warm; SQL search is ready.'
                    else:
                        message = self.verify()
                        self.manifest.parent.mkdir(parents=True, exist_ok=True)
                        temporary = self.manifest.with_suffix('.json.tmp')
                        temporary.write_text(json.dumps(self.signature(), indent=2), encoding='utf-8')
                        temporary.replace(self.manifest)
                        if not self.is_prepared():
                            self.manifest.unlink(missing_ok=True)
                            raise RuntimeError('Setup artifacts are incomplete. Retry setup to finish building them.')
                        self.ready = True
                    step.update(status='completed', message=message, elapsed_seconds=round(perf_counter() - started, 3))
                    self.message = message
                    yield self.snapshot()
                except Exception as error:
                    step.update(status='error', message=str(error), elapsed_seconds=round(perf_counter() - started, 3))
                    self.message = f"{step['label']} failed: {error}"
                    yield self.snapshot()
                    return
        finally:
            self.lock.release()

    def ask(self, question, history=None):
        with self.lock:
            if not self.ready or not self.is_prepared():
                self.ready = False
                raise RuntimeError('Run `recommender build` to prepare or verify the AniList workspace first.')
            if self.pipeline is None:
                self.pipeline = RagPipeline(self.settings)
            return self.pipeline.ask(question, history)
