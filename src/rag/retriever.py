"""
Phase 2 - RAG: retrieval over the traffic-rule knowledge base.

Two access paths, on purpose:
  * by_class_id(cid)  -> exact entry for a detected sign. Deterministic, needs no
    model and no index. This is the anchor the perception stage feeds into.
  * retrieve(query)   -> semantic search for a free-text question. Uses the index
    built by embed_index.py.

Backends match whatever embed_index.py wrote (read from kb_index_meta.json):
  * 'st'    -> sentence-transformers encodes the query.
  * 'tfidf' -> a scikit-learn TF-IDF vectorizer, re-fitted here on the same KB
               texts (identical vocabulary), transforms the query. No pickle,
               no download, fully offline.

Search core: FAISS IndexFlatIP if faiss is installed, else a NumPy dot product.
Both are exact cosine search over the normalized matrix; with 43 rows the choice
is irrelevant. CPU only; everything loads lazily so importing is cheap.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb import load_kb, get_entry            # noqa: E402
from embed_index import entry_to_text, entries  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_KB = os.path.join(_ROOT, "data", "knowledge_base", "gtsrb_rules.json")
DEFAULT_EMB = os.path.join(_ROOT, "data", "knowledge_base", "kb_embeddings.npy")
DEFAULT_META = os.path.join(_ROOT, "data", "knowledge_base", "kb_index_meta.json")


class Retriever:
    def __init__(self, kb_path=DEFAULT_KB, emb_path=DEFAULT_EMB, meta_path=DEFAULT_META):
        self.kb = load_kb(kb_path)
        self._emb_path = emb_path
        self._meta_path = meta_path
        self._emb = None
        self._meta = None
        self._faiss_index = None
        self._encoder = None   # callable: query -> normalized float32 [1, D]

    # ---- exact lookup (no model, no index) --------------------------------

    def by_class_id(self, class_id: int):
        """Return the exact KB entry for a detected class, or None."""
        return get_entry(self.kb, int(class_id))

    # ---- semantic retrieval (needs the built index) -----------------------

    def _ensure_index(self):
        if self._emb is not None:
            return
        import numpy as np
        if not os.path.isfile(self._emb_path):
            raise FileNotFoundError(
                f"Embeddings not found: {self._emb_path}\n"
                "Run: python src/rag/embed_index.py  (builds the semantic index)."
            )
        self._emb = np.load(self._emb_path).astype("float32")
        with open(self._meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        try:
            import faiss
            idx = faiss.IndexFlatIP(self._emb.shape[1])
            idx.add(self._emb)
            self._faiss_index = idx
        except Exception:
            self._faiss_index = None

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        self._ensure_index()
        import numpy as np
        backend = self._meta.get("backend", "st")

        if backend == "st":
            from sentence_transformers import SentenceTransformer
            model_name = self._meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name, device="cpu")

            def encode(q):
                return model.encode([q], convert_to_numpy=True,
                                    normalize_embeddings=True).astype("float32")
            self._encoder = encode

        elif backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [entry_to_text(e) for e in entries(self.kb)]
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            vec.fit(texts)  # same texts as embed_index -> same vocabulary

            def encode(q):
                v = vec.transform([q]).toarray().astype("float32")
                n = np.linalg.norm(v)
                if n > 0:
                    v /= n
                return v
            self._encoder = encode
        else:
            raise ValueError(f"Unknown index backend: {backend!r}")

    def retrieve(self, query: str, k: int = 3):
        """Return up to k (entry, score) pairs most relevant to the query."""
        import numpy as np
        self._ensure_encoder()
        q = self._encoder(query)

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = self._emb @ q[0]
            idxs = np.argsort(-sims)[:k]
            scores = sims[idxs]

        out = []
        for score, row in zip(scores, idxs):
            if row < 0:
                continue
            cid = self._meta["rows"][int(row)]["class_id"]
            entry = get_entry(self.kb, cid)
            if entry is not None:
                out.append((entry, float(score)))
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quick retriever smoke test.")
    ap.add_argument("--query", default="what should I do when children are near the road?")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--class-id", type=int, default=None,
                    help="If set, just print the exact entry for this class.")
    args = ap.parse_args()

    r = Retriever()
    if args.class_id is not None:
        print(json.dumps(r.by_class_id(args.class_id), indent=2, ensure_ascii=False))
    else:
        for entry, score in r.retrieve(args.query, args.k):
            print(f"[{score:.3f}] class {entry['class_id']:>2} - {entry['name']}")
            print(f"        {entry['rule']}")
