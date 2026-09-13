"""SBERT dense embeddings with chunking and a persistent cache.

Two things here matter more than they look:

**Chunking.** ``all-MiniLM-L6-v2`` has a 256-token limit. A real resume is
600-1500 tokens, so encoding one directly silently discards most of it - and the
discarded tail is usually the skills and education sections. Paper 4 solves this
with a recursive splitter, and we do the same: split into overlapping chunks,
encode each, then mean-pool into a single document vector. Ignoring this would
make the semantic score a measure of the resume's *first paragraph*.

**Normalisation.** Vectors are L2-normalised on the way out, so cosine
similarity is a plain dot product and the ranking hot path stays cheap.
"""

from __future__ import annotations

import atexit
import hashlib
import re
import threading
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL

#: Target chunk size in words. Chosen to sit comfortably under the model's
#: 256-token limit once sub-word tokenisation expands it (~1.3 tokens/word).
_CHUNK_WORDS = 170
#: Overlap between consecutive chunks, so a skill mentioned at a boundary is
#: not split away from the context that gives it meaning.
_CHUNK_OVERLAP = 30

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text: str, size: int = _CHUNK_WORDS, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word windows, respecting paragraphs first.

    Paragraphs are packed greedily so that a short section stays whole; only
    paragraphs that are themselves oversized get windowed.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(" ".join(buffer))
            buffer.clear()

    for paragraph in _PARAGRAPH_SPLIT.split(text):
        words = paragraph.split()
        if not words:
            continue
        if len(words) > size:
            flush()
            step = max(1, size - overlap)
            for start in range(0, len(words), step):
                window = words[start : start + size]
                if window:
                    chunks.append(" ".join(window))
                if start + size >= len(words):
                    break
            continue
        if len(buffer) + len(words) > size:
            flush()
        buffer.extend(words)
    flush()

    return chunks or [text]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class Embedder:
    """Lazy SBERT wrapper with an on-disk vector cache."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None
        self._lock = threading.Lock()
        self.use_cache = use_cache

        slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
        self._cache_path = (cache_dir or CACHE_DIR) / f"emb-{slug}.npz"
        self._cache: dict[str, np.ndarray] = {}
        self._cache_dirty = False
        if use_cache:
            self._load_cache()
            atexit.register(self.flush)

    # ------------------------------------------------------------- model

    @property
    def device(self) -> str:
        if self._device is None:
            try:
                import torch

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                self._device = "cpu"
        return self._device

    @property
    def model(self):
        """Load the transformer on first use.

        Deferred because importing sentence-transformers and pulling weights
        costs seconds, and several code paths (pure lexical baselines, unit
        tests of the scoring maths) never need it at all.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # ------------------------------------------------------------- cache

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with np.load(self._cache_path) as archive:
                self._cache = {key: archive[key] for key in archive.files}
        except Exception:
            # A corrupt cache must never block a run; regenerating is cheap.
            self._cache = {}

    def flush(self) -> None:
        """Persist newly computed vectors."""
        if not (self.use_cache and self._cache_dirty):
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._cache_path, **self._cache)
            self._cache_dirty = False
        except Exception:
            pass

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"k{digest[:32]}"

    # ------------------------------------------------------------ encode

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode raw strings (no chunking) into L2-normalised vectors."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        out: list[np.ndarray | None] = [None] * len(texts)
        pending: list[str] = []
        pending_at: list[int] = []

        for index, text in enumerate(texts):
            key = self._key(text)
            cached = self._cache.get(key) if self.use_cache else None
            if cached is not None:
                out[index] = cached
            else:
                pending.append(text)
                pending_at.append(index)

        if pending:
            vectors = self.model.encode(
                pending,
                batch_size=EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)
            for slot, vector in zip(pending_at, vectors):
                out[slot] = vector
                if self.use_cache:
                    self._cache[self._key(texts[slot])] = vector
                    self._cache_dirty = True

        return np.vstack([v for v in out if v is not None])

    def encode_document(self, text: str) -> np.ndarray:
        """Encode a long document by chunking and mean-pooling.

        Mean-pooling over chunks (rather than truncating) means a skill listed
        on page two still moves the document vector.
        """
        chunks = chunk_text(text)
        if not chunks:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vectors = self.encode(chunks)
        pooled = vectors.mean(axis=0)
        return _normalise(pooled.reshape(1, -1))[0].astype(np.float32)

    def encode_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Chunk-and-pool a batch of documents.

        All chunks across all documents are encoded in one batched call, which
        is markedly faster on GPU than encoding each document separately.
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        all_chunks: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            chunks = chunk_text(text)
            start = len(all_chunks)
            all_chunks.extend(chunks)
            spans.append((start, len(all_chunks)))

        vectors = self.encode(all_chunks, show_progress=show_progress)
        pooled = np.vstack(
            [
                vectors[start:end].mean(axis=0)
                if end > start
                else np.zeros(EMBEDDING_DIM, dtype=np.float32)
                for start, end in spans
            ]
        )
        return _normalise(pooled).astype(np.float32)


_EMBEDDER: Embedder | None = None


def get_embedder(**kwargs) -> Embedder:
    """Process-wide embedder, so the model is loaded at most once."""
    global _EMBEDDER
    if kwargs:
        return Embedder(**kwargs)
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


# --------------------------------------------------------------------------
# Similarity helpers
# --------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1].

    Clamping to zero is a deliberate scoring choice, not a maths error: a
    negative cosine and a zero cosine both mean "unrelated" for our purposes,
    and letting negatives through would make the weighted total dip below the
    range every other sub-score lives in.
    """
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(a, b)) / denominator)))


def cosine_matrix(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities between two normalised matrices."""
    if queries.size == 0 or documents.size == 0:
        return np.zeros((queries.shape[0], documents.shape[0]), dtype=np.float32)
    return np.clip(_normalise(queries) @ _normalise(documents).T, 0.0, 1.0)
