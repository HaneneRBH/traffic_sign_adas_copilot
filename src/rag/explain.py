"""
Phase 2 - RAG: turn a detected sign into a grounded, natural-language explanation.

Design goal: the explanation must be *grounded* — every fact comes from the KB
entry, never invented by the model. Two layers guarantee this:

  1. The prompt hands the LLM one "rule card" and forbids adding anything not in it.
  2. If no local LLM is reachable (Ollama not installed or not running), we fall
     back to a deterministic template built straight from the KB fields. That
     fallback cannot hallucinate at all — it only reorders the stored text.

Public functions:
  explain(class_id)      -> explanation for a sign the perception stage detected.
  answer(question)       -> free-text RAG: retrieve relevant rules, then explain.

The LLM is optional and local (Ollama, https://ollama.com). Default model
'llama3.2'. Everything is CPU-friendly; the LLM runs in Ollama's own process.

    # after building the index once (embed_index.py):
    python src/rag/explain.py --class-id 28
    python src/rag/explain.py --ask "what do I do near a school?"
    python src/rag/explain.py --class-id 28 --no-llm     # template only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perception"))
from retriever import Retriever          # noqa: E402
from class_names import SPEED_LIMIT_KMH  # noqa: E402

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("ADAS_LLM_MODEL", "llama3.2")

SYSTEM = (
    "You are a calm in-car driving assistant. In 2-3 short sentences, explain the "
    "traffic sign to the driver in plain language. Use ONLY the facts in the rule "
    "card below. Do not invent any rule, speed, distance, fine, or legal detail that "
    "is not written in the card. If the card does not say something, do not mention it."
)


def _card(entry: dict) -> str:
    """One rule card, the only ground truth the LLM is allowed to use."""
    lines = [
        f"Sign: {entry['name']} (GTSRB class {entry['class_id']}, category {entry['category']})",
        f"Rule: {entry['rule']}",
        f"Context: {entry['context']}",
        f"Danger: {entry['typical_danger']}",
        f"Recommended action: {entry['driver_action']}",
    ]
    kmh = SPEED_LIMIT_KMH.get(int(entry["class_id"]))
    if kmh is not None:
        lines.append(f"Numeric speed limit: {kmh} km/h")
    return "\n".join(lines)


def _prompt(cards: str, question: str = None) -> str:
    ask = (f'\n\nDriver question: "{question}"\nAnswer using only the card(s) above.'
           if question else "\n\nExplain this sign to the driver now.")
    return f"{SYSTEM}\n\n--- rule card ---\n{cards}\n--- end card ---{ask}"


def _call_ollama(prompt: str, model: str, host: str, timeout: float = 60.0):
    """Return the LLM text, or None if Ollama is unreachable / errors out."""
    try:
        import requests
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception:
        return None


def _template(entry: dict) -> str:
    """Deterministic, hallucination-proof explanation from KB fields only."""
    kmh = SPEED_LIMIT_KMH.get(int(entry["class_id"]))
    speed = f" The limit is {kmh} km/h." if kmh is not None else ""
    return f"{entry['name']}. {entry['rule']}{speed} {entry['driver_action']}"


# --- public API -----------------------------------------------------------

def explain(class_id: int, use_llm: bool = True, model: str = DEFAULT_LLM,
            host: str = OLLAMA_HOST, retriever: Retriever = None):
    """Grounded explanation for a detected sign class. Falls back to a template."""
    r = retriever or Retriever()
    entry = r.by_class_id(class_id)
    if entry is None:
        return {"class_id": class_id, "found": False,
                "explanation": f"No knowledge-base entry for class {class_id}."}

    text, source = None, "template"
    if use_llm:
        text = _call_ollama(_prompt(_card(entry)), model, host)
        source = "llm" if text else "template"
    if text is None:
        text = _template(entry)

    return {"class_id": int(class_id), "found": True, "name": entry["name"],
            "category": entry["category"], "source": source, "explanation": text}


def answer(question: str, k: int = 3, use_llm: bool = True, model: str = DEFAULT_LLM,
           host: str = OLLAMA_HOST, retriever: Retriever = None):
    """Free-text RAG: retrieve the most relevant rules, then explain from them."""
    r = retriever or Retriever()
    hits = r.retrieve(question, k=k)
    if not hits:
        return {"question": question, "explanation": "No relevant rule found.", "used": []}

    cards = "\n\n".join(_card(e) for e, _ in hits)
    used = [{"class_id": e["class_id"], "name": e["name"], "score": round(s, 3)}
            for e, s in hits]

    text, source = None, "template"
    if use_llm:
        text = _call_ollama(_prompt(cards, question), model, host)
        source = "llm" if text else "template"
    if text is None:
        top = hits[0][0]
        text = _template(top)

    return {"question": question, "source": source, "explanation": text, "used": used}


def parse_args():
    p = argparse.ArgumentParser(description="Grounded explanation for a detected sign.")
    p.add_argument("--class-id", type=int, default=None)
    p.add_argument("--ask", default=None, help="Free-text question (semantic retrieval).")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--no-llm", action="store_true", help="Skip Ollama, use template only.")
    p.add_argument("--model", default=DEFAULT_LLM)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_llm = not args.no_llm
    if args.ask is not None:
        out = answer(args.ask, k=args.k, use_llm=use_llm, model=args.model)
    elif args.class_id is not None:
        out = explain(args.class_id, use_llm=use_llm, model=args.model)
    else:
        raise SystemExit("Give --class-id N or --ask \"your question\".")
    print(json.dumps(out, indent=2, ensure_ascii=False))
