"""Chunk index: dense (BGE-large-en-v1.5 + FAISS inner product on normalised vectors) or
sparse (TF-IDF cosine) fallback.  `auto` picks dense when the optional packages exist."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DENSE_MODEL = "BAAI/bge-large-en-v1.5"


def _dense_ok() -> bool:
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class ChunkIndex:
    def __init__(self, backend: str = "auto", model_name: str = DENSE_MODEL):
        self.backend = ("dense" if _dense_ok() else "sparse") if backend == "auto" else backend
        self.model_name, self.chunks, self.meta = model_name, [], []
        self._index = self._encoder = self._vec = self._mat = None

    def _enc(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def build(self, chunks: list[str], meta: list[dict]):
        self.chunks, self.meta = list(chunks), list(meta)
        if self.backend == "dense":
            import faiss
            emb = self._enc().encode(self.chunks, batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")
            self._index = faiss.IndexFlatIP(emb.shape[1]); self._index.add(emb)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
            self._mat = self._vec.fit_transform(self.chunks)
        return self

    def search(self, query: str, k: int = 20, min_score: float = 0.30) -> list[dict]:
        if not self.chunks:
            return []
        if self.backend == "dense":
            q = self._enc().encode([query], normalize_embeddings=True).astype("float32")
            scores, ids = self._index.search(q, min(k, len(self.chunks)))
            pairs = zip(scores[0], ids[0])
        else:
            s = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
            top = np.argsort(-s)[:k]
            pairs = ((s[i], i) for i in top)
        out = []
        for sc, i in pairs:
            if i < 0 or sc < min_score:
                continue
            m = self.meta[i]
            out.append(dict(score=float(sc), text=self.chunks[i], doc_id=m.get("doc_id", ""),
                            citation=m.get("citation", ""), doi=m.get("doi", ""), title=m.get("title", ""),
                            year=m.get("year"), abstract_only=bool(m.get("abstract_only", False))))
        return out

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        json.dump({"backend": self.backend, "model_name": self.model_name}, open(path / "index.json", "w"))
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c, m in zip(self.chunks, self.meta):
                fh.write(json.dumps({"text": c, "meta": m}, ensure_ascii=False) + "\n")
        if self.backend == "dense":
            import faiss
            faiss.write_index(self._index, str(path / "chunks.faiss"))

    @classmethod
    def load(cls, path):
        path = Path(path)
        path = path / "index" if (path / "index" / "index.json").exists() else path
        cfg = json.load(open(path / "index.json"))
        obj = cls(backend=cfg["backend"], model_name=cfg.get("model_name", DENSE_MODEL))
        rows = [json.loads(l) for l in open(path / "chunks.jsonl", encoding="utf-8")]
        obj.chunks, obj.meta = [r["text"] for r in rows], [r["meta"] for r in rows]
        if obj.backend == "dense":
            import faiss
            obj._index = faiss.read_index(str(path / "chunks.faiss"))
        else:
            obj.build(obj.chunks, obj.meta)
        return obj
