"""
rag/vector_store.py — the RAG layer: company documents -> chunks -> embeddings -> FAISS.

How it works
1. Every .md / .txt file in knowledge_base/ is split into small chunks
   (one chunk per "## " section, long sections split by paragraph).
2. Each chunk is turned into a vector (an "embedding").
     - EMBEDDING_PROVIDER=openai -> OpenAI embeddings (best quality)
     - EMBEDDING_PROVIDER=local  -> TF-IDF vectors (free, offline, no API key)
3. The vectors are stored in a FAISS index (cosine similarity).
4. search(query) embeds the question and returns the closest chunks, but only
   those above MIN_SIMILARITY, so unrelated questions return NOTHING instead
   of a weak, misleading match.

Error handling built in
- Empty / missing knowledge_base folder -> KnowledgeBaseError with a clear message.
- Unreadable files are skipped and reported in .warnings.
- The OpenAI index is cached on disk and rebuilt automatically when any
  document changes (fingerprint check). A corrupt cache is deleted and rebuilt.
- If OpenAI embeddings fail (bad key, no quota, no internet), the store falls
  back to local TF-IDF embeddings so the support desk keeps working.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}
MAX_CHUNK_CHARS = 1200


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base cannot be loaded or searched."""


@dataclass
class Chunk:
    source: str      # file name, e.g. refund_policy.md
    section: str     # heading, e.g. Refund timing
    text: str


# --------------------------------------------------------------------------
# 1. Load and chunk documents
# --------------------------------------------------------------------------
def load_chunks(kb_dir: Path, warnings: list[str]) -> list[Chunk]:
    if not kb_dir.exists():
        raise KnowledgeBaseError(f"Knowledge base folder not found: {kb_dir}")

    files = sorted(p for p in kb_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise KnowledgeBaseError(
            f"No .md or .txt documents found in {kb_dir}. Add at least one policy document."
        )

    chunks: list[Chunk] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            warnings.append(f"Skipped {path.name}: could not read file ({exc}).")
            continue
        if not text:
            warnings.append(f"Skipped {path.name}: file is empty.")
            continue
        chunks.extend(_split_document(path.name, text))

    if not chunks:
        raise KnowledgeBaseError("All knowledge base documents were empty or unreadable.")
    return chunks


def _split_document(source: str, text: str) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source

    # Split on level-2 headings ("## Heading")
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    for part in parts[1:] if len(parts) > 1 else parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        section = lines[0].strip() if len(parts) > 1 else title
        body = "\n".join(lines[1:]).strip() if len(parts) > 1 else part.strip()
        if not body:
            continue
        for piece in _split_long(body):
            chunks.append(Chunk(source=source, section=section,
                                text=f"{title} — {section}\n{piece}"))
    return chunks


def _split_long(body: str) -> list[str]:
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    pieces, current = [], ""
    for para in body.split("\n\n"):
        if current and len(current) + len(para) > MAX_CHUNK_CHARS:
            pieces.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def fingerprint(kb_dir: Path, provider: str, model: str) -> str:
    h = hashlib.sha256(f"{provider}|{model}".encode())
    for p in sorted(kb_dir.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# 2. Embedders
# --------------------------------------------------------------------------
class LocalEmbedder:
    """Free, offline TF-IDF embeddings (character n-grams). Good for keyword-style matching."""

    name = "local"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Character n-grams cope with typos and word forms (refund/refunds, broke/broken).
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                           sublinear_tf=True)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise KnowledgeBaseError("Local embedder used before fit().")
        return self._vectorizer.transform(texts).toarray().astype("float32")


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None,
                 timeout: int, max_retries: int) -> None:
        from openai import OpenAI
        if not api_key:
            raise KnowledgeBaseError("No API key for OpenAI embeddings.")
        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              timeout=timeout, max_retries=max_retries)
        self._model = model

    def fit(self, texts: list[str]) -> None:  # nothing to fit
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):          # batch to stay under limits
            batch = [t.replace("\n", " ")[:8000] for t in texts[start:start + 100]]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return np.array(vectors, dtype="float32")


# --------------------------------------------------------------------------
# 3. FAISS knowledge base
# --------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype="float32")
    faiss.normalize_L2(v)          # cosine similarity == inner product of unit vectors
    return v


class KnowledgeBase:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.warnings: list[str] = []
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None
        self.embedder = None
        self.backend = ""
        self.min_similarity = settings.min_similarity
        self.load_or_build()

    # ---------------- build / load ----------------
    def load_or_build(self, force_rebuild: bool = False) -> None:
        self.warnings = []
        self.chunks = load_chunks(self.settings.kb_dir, self.warnings)

        if self.settings.embedding_provider == "openai":
            try:
                self._load_or_build_openai(force_rebuild)
                return
            except Exception as exc:  # noqa: BLE001 — any failure -> safe fallback
                self.warnings.append(
                    f"OpenAI embeddings unavailable ({type(exc).__name__}: {exc}). "
                    "Falling back to local TF-IDF search."
                )
                logger.warning("Falling back to local embeddings: %s", exc)
        self._build_local()

    def _build_local(self) -> None:
        self.embedder = LocalEmbedder()
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        vectors = _normalize(self.embedder.embed(texts))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.backend = "local"
        # Local scores are lower than OpenAI scores; use the local threshold.
        if self.settings.embedding_provider != "local":
            self.min_similarity = 0.18

    def _load_or_build_openai(self, force_rebuild: bool) -> None:
        s = self.settings
        self.embedder = OpenAIEmbedder(s.api_key if s.has_api_key else "", s.embedding_model,
                                       s.base_url, s.request_timeout, s.max_retries)
        fp = fingerprint(s.kb_dir, "openai", s.embedding_model)
        index_path = s.index_dir / "faiss.index"
        meta_path = s.index_dir / "meta.json"

        if not force_rebuild and index_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("fingerprint") == fp:
                    self.index = faiss.read_index(str(index_path))
                    self.chunks = [Chunk(**c) for c in meta["chunks"]]
                    if self.index.ntotal != len(self.chunks):
                        raise KnowledgeBaseError("Index and metadata are out of sync.")
                    self.backend = "openai (cached)"
                    return
            except Exception as exc:  # corrupt cache -> delete and rebuild
                self.warnings.append(f"Vector store cache was unreadable ({exc}); rebuilt it.")
                shutil.rmtree(s.index_dir, ignore_errors=True)

        vectors = _normalize(self.embedder.embed([c.text for c in self.chunks]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.backend = "openai"
        try:
            s.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            meta_path.write_text(json.dumps({"fingerprint": fp,
                                             "chunks": [asdict(c) for c in self.chunks]}),
                                 encoding="utf-8")
        except OSError as exc:
            self.warnings.append(f"Could not save vector store to disk ({exc}); "
                                 "it will be rebuilt next start.")

    # ---------------- search ----------------
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        k = min(top_k or self.settings.top_k, len(self.chunks))
        try:
            qv = _normalize(self.embedder.embed([query]))
        except Exception as exc:  # e.g. internet drops in the middle of the day
            if self.backend.startswith("openai"):
                self.warnings.append(f"Embedding the query failed ({exc}); "
                                     "switched to local search.")
                self._build_local()
                qv = _normalize(self.embedder.embed([query]))
            else:
                raise KnowledgeBaseError(f"Search failed: {exc}") from exc

        scores, ids = self.index.search(qv, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or score < self.min_similarity:
                continue
            c = self.chunks[idx]
            results.append({"source": c.source, "section": c.section,
                            "text": c.text, "score": round(float(score), 3)})
        return results

    @property
    def document_names(self) -> list[str]:
        return sorted({c.source for c in self.chunks})
