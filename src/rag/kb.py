"""
Knowledge-base loader and validator for the Traffic-Sign ADAS Copilot.

Usage as a script:
    python src/rag/kb.py --path data/knowledge_base/gtsrb_rules.json --validate

Usage as a library:
    from rag.kb import load_kb, get_entry
    kb = load_kb("data/knowledge_base/gtsrb_rules.json")
    entry = get_entry(kb, class_id=14)   # -> the "Stop" entry

The validator is intentionally strict: missing fields or malformed class_ids
are caught here rather than surfacing later in the retrieval pipeline.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Import the canonical class list so we can cross-check coverage.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "perception"))
try:
    from class_names import CLASS_NAMES  # type: ignore
except ImportError:
    CLASS_NAMES = {i: f"class_{i}" for i in range(43)}  # fallback if run standalone

REQUIRED_FIELDS = ("class_id", "name", "category", "rule",
                   "context", "typical_danger", "driver_action")
VALID_CATEGORIES = {"speed_limit", "prohibition", "mandatory",
                    "warning", "priority", "other"}
NUM_CLASSES = 43


def load_kb(path: str) -> Dict:
    """Load the KB JSON from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"KB file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entries(kb: Dict) -> List[Dict]:
    return kb.get("entries", [])


def get_entry(kb: Dict, class_id: int) -> Optional[Dict]:
    """Return the entry for a given GTSRB class_id, or None if absent."""
    for e in entries(kb):
        if int(e.get("class_id", -1)) == int(class_id):
            return e
    return None


def validate(kb: Dict, verbose: bool = True) -> bool:
    """Return True if the KB passes all checks. Prints a coverage report."""
    ok = True
    seen_ids = set()

    for i, e in enumerate(entries(kb)):
        # Required fields present and non-empty (except class_id which is int).
        for f in REQUIRED_FIELDS:
            if f not in e:
                print(f"[error] entry #{i}: missing field '{f}'")
                ok = False
            elif f != "class_id" and not str(e[f]).strip():
                print(f"[error] entry #{i}: field '{f}' is empty")
                ok = False

        # class_id must be int in 0..42 and unique.
        cid = e.get("class_id")
        if not isinstance(cid, int) or cid < 0 or cid >= NUM_CLASSES:
            print(f"[error] entry #{i}: class_id={cid!r} must be int in [0, {NUM_CLASSES - 1}]")
            ok = False
        elif cid in seen_ids:
            print(f"[error] entry #{i}: duplicate class_id={cid}")
            ok = False
        else:
            seen_ids.add(cid)

        # Category must be in the allowed set.
        cat = e.get("category")
        if cat not in VALID_CATEGORIES:
            print(f"[error] entry #{i} (class_id={cid}): category={cat!r} not in {sorted(VALID_CATEGORIES)}")
            ok = False

        # Optional: warn if 'name' disagrees with CLASS_NAMES (typo safeguard).
        if isinstance(cid, int) and 0 <= cid < NUM_CLASSES:
            expected = CLASS_NAMES.get(cid, "")
            if expected and e.get("name") and e["name"].strip().lower() != expected.strip().lower():
                print(f"[warn]  class_id={cid}: KB name '{e['name']}' differs from CLASS_NAMES '{expected}'")

    # Coverage report.
    missing = sorted(set(range(NUM_CLASSES)) - seen_ids)
    if verbose:
        print(f"\nCoverage: {len(seen_ids)}/{NUM_CLASSES} classes with entries.")
        if missing:
            preview = ", ".join(f"{c} ({CLASS_NAMES.get(c, '?')})" for c in missing[:6])
            more = "" if len(missing) <= 6 else f", ... (+{len(missing) - 6} more)"
            print(f"Missing class_ids: {preview}{more}")

    return ok


def main():
    ap = argparse.ArgumentParser(description="Load and validate the traffic-rule knowledge base.")
    ap.add_argument("--path", default="data/knowledge_base/gtsrb_rules.json")
    ap.add_argument("--validate", action="store_true", help="Run schema + coverage checks.")
    ap.add_argument("--show", type=int, default=None,
                    help="Print the entry for a given class_id.")
    args = ap.parse_args()

    kb = load_kb(args.path)
    print(f"Loaded KB with {len(entries(kb))} entries from {args.path}")

    if args.show is not None:
        e = get_entry(kb, args.show)
        if e is None:
            print(f"No entry for class_id={args.show}")
        else:
            print(json.dumps(e, indent=2, ensure_ascii=False))

    if args.validate:
        ok = validate(kb)
        print("\nRESULT:", "OK" if ok else "FAILED (see errors above)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
