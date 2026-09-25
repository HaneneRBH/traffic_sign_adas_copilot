"""
Phase 2 - RAG: build the semantic index over the traffic-rule knowledge base.

Each KB entry (one per GTSRB class) is turned into a short text document and
embedded into a vector. That matrix powers *free-text* retrieval in retriever.py
(e.g. "what do I do near a school?"). The exact class_id -> rule lookup used by
explain() does NOT need this index; even if you never build it, the core ADAS
explanation still works.

Two embedding backends, chosen automatically (--backend auto, the default):
  * 'st'    -> sentence-transformers (better semantics, needs a working install
               + a one-time ~80 MB model download from Hugging Face).
  * 'tfidf' -> scikit-learn TF-IDF (already in Anaconda, no download, fully
               offline). Used automatically if sentence-transformers can't load.

With only 43 short rule cards, TF-IDF retrieval is perfectly usable; 'st' just
generalizes better to paraphrases. The retriever reads the backend from the meta
file and matches it, so build and query always agree.

Run once, and re-run whenever gtsrb_rules.json changes:

    python src/rag/embed_index.py --backend auto

Outputs (written next to the KB):
    kb_embeddings.npy    float32 [N, D], L2-normalized (cosine == dot product)
    kb_index_meta.json   {backend, model?, dim, rows:[{class_id,name,category}]}

CPU only. Guarded by `if __name__ == "__main__":` for Windows / Spyder.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb import load_kb, entries  # noqa: E402

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def entry_to_text(e: dict) -> str:
    """Flatten one KB entry into a single embeddable document string.

    Shared with retriever.py so the TF-IDF vocabulary is identical on both sides.
    """
    return (
        f"{e['name']} (category: {e['category']}). "
        f"Rule: {e['rule']} "
        f"Context: {e['context']} "
        f"Danger: {e['typical_danger']} "
        f"Driver action: {e['driver_action']}"
    )


def _l2_normalize(mat):
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


def _embed_st(texts, model_name):
    """sentence-transformers backend. Raises if it can't import/load."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(texts, batch_size=16, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.astype("float32"), {"backend": "st", "model": model_name}


def _embed_tfidf(texts):
    """scikit-learn TF-IDF backend. No download, offline, already in Anaconda."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(texts).toarray()
    return _l2_normalize(mat), {"backend": "tfidf"}


def build(kb_path: str, out_dir: str, backend: str = "auto",
          model_name: str = DEFAULT_MODEL) -> None:
    import numpy as np

    kb = load_kb(kb_path)
    rows = entries(kb)
    if not rows:
        raise SystemExit("KB has no entries - nothing to index.")

    texts = [entry_to_text(e) for e in rows]
    meta_rows = [
        {"class_id": int(e["class_id"]), "name": e["name"], "category": e["category"]}
        for e in rows
    ]

    if backend in ("auto", "st"):
        try:
            print(f"Trying sentence-transformers backend ('{model_name}', CPU)...")
            emb, info = _embed_st(texts, model_name)
        except Exception as exc:
            if backend == "st":
                raise
            print(f"[warn] sentence-transformers unavailable ({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0]}).")
            print("[info] falling back to TF-IDF backend (offline, no download).")
            emb, info = _embed_tfidf(texts)
    else:  # backend == "tfidf"
        print("Using TF-IDF backend (offline)...")
        emb, info = _embed_tfidf(texts)

    info["dim"] = int(emb.shape[1])
    info["rows"] = meta_rows

    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, "kb_embeddings.npy")
    meta_path = os.path.join(out_dir, "kb_index_meta.json")
    np.save(emb_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[ok] backend    : {info['backend']}")
    print(f"[ok] {emb.shape[0]} vectors x {emb.shape[1]} dims")
    print(f"[ok] embeddings -> {emb_path}")
    print(f"[ok] meta       -> {meta_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Build the RAG semantic index over the KB.")
    p.add_argument("--kb", default="data/knowledge_base/gtsrb_rules.json")
    p.add_argument("--out-dir", default="data/knowledge_base")
    p.add_argument("--backend", choices=["auto", "st", "tfidf"], default="auto")
    p.add_argument("--model", default=DEFAULT_MODEL)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.kb, args.out_dir, args.backend, args.model)
