"""Resumable setup and readiness checks for the dedicated AniList app."""
from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
import threading
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from .config import Settings
from .datasets import ANILIST_FILENAME, ANILIST_URL, prepare_anilist
from .entity_agent import EntityAgent
from .documents import load_chunks
from .rag import RagPipeline
from .sqlite_catalog import SCHEMA_VERSION

STEPS = (
    ('model', 'Generation model'), ('dataset', 'Dataset download'),
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
        self.message = 'Workspace prepared. You can chat or run setup to verify it again.' if self.ready else 'Run setup to prepare your AniList library.'

    def signature(self):
        s = self.settings
        paths = [s.documents / 'catalog.jsonl', s.index / 'vectors.npy', s.index / 'chunks.json']
        return {'version': 1, 'sql_schema': SCHEMA_VERSION, 'embedding': s.embedding,
                'embedding_cache_version': s.embedding_cache_version,
                'chunk_size': s.chunk_size, 'chunk_overlap': s.chunk_overlap,
                'files': {str(p.relative_to(s.root)): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}

    def is_prepared(self):
        try:
            required = [self.settings.documents / 'catalog.sqlite3',
                        self.settings.index / 'entity-tools.sqlite3', self.settings.index / 'entity-features.sqlite3']
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
            'embeddings': [s.index / 'chunks.json', s.index / 'vectors.npy'],
            'tools': [s.index / 'entity-tools.sqlite3', s.index / 'entity-features.sqlite3'],
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
        """Validate the index independently of the final setup-ready marker."""
        vectors = None
        try:
            s = self.settings
            chunks = json.loads((s.index / 'chunks.json').read_text(encoding='utf-8'))
            vectors = np.load(s.index / 'vectors.npy', mmap_mode='r', allow_pickle=False)
            probe = self.pipeline.models.embed(['AniList workspace verification'])
            if (not isinstance(chunks, list) or not chunks or vectors.ndim != 2
                    or vectors.shape != (len(chunks), probe.shape[1])):
                return False
            if any(not np.isfinite(vectors[start:start + 8192]).all()
                   for start in range(0, len(vectors), 8192)):
                return False
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
            # Adopt older prepared libraries without trusting dimensions alone:
            # compare their chunks with today's documents/settings, then prove
            # each vector belongs to the current model's content-addressed cache.
            if not s.embedding_cache or not (s.index / 'embeddings.sqlite3').is_file():
                return False
            expected = load_chunks(s.documents, s.extensions, s.chunk_size, s.chunk_overlap)
            if len(expected) != len(chunks) or any(c.__dict__ != saved for c, saved in zip(expected, chunks)):
                return False
            models = self.pipeline.models
            namespace = json.dumps([s.embedding, s.embedding_cache_version,
                                    getattr(models, 'cache_revision', None),
                                    getattr(getattr(models, 'embedder', None), 'max_seq_length', None),
                                    'normalized-f32-v1'])
            with closing(sqlite3.connect(s.index / 'embeddings.sqlite3')) as db:
                for start in range(0, len(chunks), 256):
                    keys = [hashlib.sha256(c['text'].encode('utf-8')).hexdigest()
                            for c in chunks[start:start + 256]]
                    placeholders = ','.join('?' for _ in keys)
                    found = dict(db.execute(f'SELECT hash,vector FROM embeddings WHERE namespace=? '
                                            f'AND hash IN ({placeholders})', [namespace, *keys]))
                    if any(key not in found or not np.array_equal(
                            vectors[start + i], np.frombuffer(found[key], dtype=np.float32))
                           for i, key in enumerate(keys)):
                        return False
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, sqlite3.Error):
            return False
        finally:
            if vectors is not None and getattr(vectors, '_mmap', None) is not None:
                vectors._mmap.close()

    def ensure_model(self, progress):
        url = self.settings.ollama_url
        try:
            response = requests.get(f'{url}/api/tags', timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f'Cannot connect to Ollama at {url}. Start the Ollama app or run ollama serve, then retry setup.') from error
        model = self.settings.llm
        canonical = lambda name: name if ':' in name.rsplit('/', 1)[-1] else name + ':latest'
        names = [m.get('name', m.get('model', '')) for m in response.json().get('models', [])]
        if canonical(model) in {canonical(name) for name in names}:
            return f'{model} is already installed.'
        with requests.post(f'{url}/api/pull', json={'name': model, 'stream': True},
                           stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            complete = False
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                progress('Ollama: ' + event.get('status', 'Downloading model'),
                         int(event.get('completed', 0)), int(event.get('total', 0)))
                complete = event.get('status') == 'success'
            if not complete:
                raise RuntimeError('Model download ended before Ollama reported success. Retry setup.')
        return f'Downloaded {model}.'

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
        chunks = json.loads((self.settings.index / 'chunks.json').read_text(encoding='utf-8'))
        vectors = np.load(self.settings.index / 'vectors.npy', mmap_mode='r', allow_pickle=False)
        try:
            probe = p.models.embed(['AniList workspace verification'])
            if vectors.ndim != 2 or len(vectors) != len(chunks) or not len(chunks) or vectors.shape[1] != probe.shape[1]:
                raise RuntimeError('The embedding index does not match its chunks or embedding model. Select rebuild and retry.')
            for start in range(0, len(vectors), 8192):
                if not np.isfinite(vectors[start:start + 8192]).all():
                    raise RuntimeError('The embedding index contains invalid values. Select rebuild and retry.')
        finally:
            vectors._mmap.close()
        if not p.models.answer('Reply with the single word READY.', 0, 8).strip():
            raise RuntimeError('The generation model returned an empty response.')
        return f'{count:,} anime and {len(chunks):,} embedded passages verified. Ready to chat.'

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
                    if key == 'model':
                        message = self.ensure_model(progress)
                    elif key == 'dataset':
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
                        self.pipeline._agent.guides('Recommend anime')
                        self.pipeline._agent.feature_candidates('Anime genres and themes')
                        message = 'Tool guides and genre/tag embeddings are warm; SQL search is ready.'
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

    def ask(self, question, history=None, *, debug=None):
        with self.lock:
            if not self.ready or not self.is_prepared():
                self.ready = False
                raise RuntimeError('Run setup to prepare or verify the AniList workspace before chatting.')
            if self.pipeline is None:
                self.pipeline = RagPipeline(self.settings)
            return self.pipeline.ask(question, history, debug=debug)
