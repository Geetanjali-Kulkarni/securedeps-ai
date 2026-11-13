"""Retrieval: chunk documents, turn chunks into vectors, find the chunks closest to a question.

This is the 'R' in RAG, written by hand so every step is visible:

  1. CHUNK   split each document into pieces of about 700 characters (an LLM prompt cannot
             hold everything, and small pieces make retrieval precise).
  2. EMBED   turn each piece into a vector of numbers so that similar meaning = similar vector.
  3. SEARCH  turn the question into a vector too, rank chunks by cosine similarity.

Two embedders, same interface:
  SentenceTransformerEmbedder  real semantic embeddings (pip install sentence-transformers)
  HashingEmbedder              pure Python word-overlap vectors; no install, weaker on synonyms

We hold only a few dozen chunks per question, so a plain list with cosine similarity is all
the 'vector store' we need. A database such as ChromaDB would only help for persistence or
millions of chunks.
"""
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .evidence import Doc

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to "
                  "was were will with you your not can may if when which who".split())


# ----------------------------------------------------------------- chunking
def chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    """Split on paragraphs, pack paragraphs up to max_chars; a single over-long paragraph is cut
    into windows that overlap, so a sentence cut at the boundary still appears whole once."""
    text = text.strip()
    if not text:
        return []
    chunks, current = [], ""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            step = max_chars - overlap
            pieces += [para[i:i + max_chars] for i in range(0, len(para), step)]
    for p in pieces:
        if not p:
            continue
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class Chunk:
    chunk_id: str    # "S1", "S2", ... the id the LLM must cite
    doc_id: str
    kind: str
    title: str
    url: str
    text: str


def chunk_docs(docs: Sequence[Doc], max_chars: int = 700) -> List[Chunk]:
    out, n = [], 0
    for d in docs:
        for piece in chunk_text(d.text, max_chars):
            n += 1
            out.append(Chunk(f"S{n}", d.doc_id, d.kind, d.title, d.url, piece))
    return out


# --------------------------------------------------------------- embedders
def tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed number of slots, with sublinear term weighting."""
    name = "hashing (word overlap)"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for t in texts:
            counts: Dict[int, int] = {}
            for w in tokens(t):
                slot = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                counts[slot] = counts.get(slot, 0) + 1
            vec = [0.0] * self.dim
            for slot, c in counts.items():
                vec[slot] = 1 + math.log(c)
            out.append(_normalize(vec))
        return out


class SentenceTransformerEmbedder:
    name = "sentence-transformers all-MiniLM-L6-v2"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer   # imported lazily: heavy
        self._model = SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in
                self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)]


def get_embedder():
    try:
        return SentenceTransformerEmbedder()
    except Exception:       # not installed, or the model could not be downloaded
        return HashingEmbedder()


def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))        # vectors are normalized, so dot = cosine


# --------------------------------------------------------------- retriever
class Retriever:
    def __init__(self, embedder, chunks: Sequence[Chunk]):
        self.embedder, self.chunks = embedder, list(chunks)
        self.vectors = embedder.embed([c.text for c in self.chunks]) if self.chunks else []

    def search(self, query: str, k: int = 6, min_per_kind: Optional[Dict[str, int]] = None
               ) -> List[Tuple[Chunk, float]]:
        """Top-k chunks by similarity. min_per_kind guarantees some release-note chunks make
        it in even when long advisories would otherwise fill every slot."""
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        ranked = sorted(((c, cosine(q, v)) for c, v in zip(self.chunks, self.vectors)),
                        key=lambda t: -t[1])
        picked = ranked[:k]
        for kind, need in (min_per_kind or {}).items():
            have = sum(1 for c, _ in picked if c.kind == kind)
            extras = [t for t in ranked[k:] if t[0].kind == kind][:max(0, need - have)]
            for extra in extras:
                for idx in range(len(picked) - 1, -1, -1):      # replace the weakest other-kind chunk
                    if picked[idx][0].kind != kind:
                        picked[idx] = extra
                        break
        return sorted(picked, key=lambda t: -t[1])
