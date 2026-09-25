"""
Bridge between Phase 1 (perception) and Phase 2 (RAG).

Takes a single traffic-sign image, runs the fine-tuned MobileNetV3 to get a
class_id, then asks the RAG layer for a grounded explanation:

        image  ->  class_id (+ confidence)  ->  grounded explanation

This is the whole pipeline in one call, and the core of the future Streamlit demo.

    python src/pipeline/predict_and_explain.py --image data/Test/00001.png
    python src/pipeline/predict_and_explain.py --image data/Test/00001.png --no-llm
    python src/pipeline/predict_and_explain.py --image path.png --ckpt checkpoints/perception_mobilenetv3_v2.pt --topk 3

Preprocessing is taken straight from dataset.py's eval transform (Resize ->
ToTensor -> ImageNet Normalize), so it matches training exactly. GTSRB test images
are already tightly cropped to the sign, so no ROI crop is applied here — that is
what a real detected sign patch would look like.

CPU only. Heavy imports (torch) are deferred into the functions that need them, so
importing this module stays cheap and the RAG side can be reused on its own.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "perception"))
sys.path.insert(0, os.path.join(_HERE, "..", "rag"))

# Light import (no torch): the RAG entry points.
from explain import explain, Retriever  # noqa: E402

_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "perception_mobilenetv3_v2.pt")


def load_model(ckpt_path=DEFAULT_CKPT):
    """Load the perception model + its training image size from a checkpoint."""
    import torch
    from model import build_model, NUM_CLASSES

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # weights_only=False is safe here: this is your own trained checkpoint, and
    # it stores plain metadata (img_size, num_classes) alongside the weights.
    # Setting it explicitly also silences PyTorch's FutureWarning.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        img_size = int(ckpt.get("img_size", 64))
        num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    else:  # a bare state_dict was saved
        state, img_size, num_classes = ckpt, 96, NUM_CLASSES

    model = build_model(num_classes, pretrained=False, freeze="none")
    model.load_state_dict(state)
    model.eval()
    return model, img_size


def predict(model, img_size, image_path, topk=3):
    """Return (class_id, confidence, [(class_id, prob), ...]) for one image."""
    import torch
    from PIL import Image
    from dataset import build_transforms

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    tf = build_transforms(img_size, train=False)      # same as eval
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)                            # [1, 3, H, W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    k = min(topk, probs.numel())
    top_p, top_i = torch.topk(probs, k)
    top = [(int(i), float(p)) for p, i in zip(top_p, top_i)]
    class_id, confidence = top[0]
    return class_id, confidence, top


def predict_and_explain(image_path, ckpt_path=DEFAULT_CKPT, use_llm=True,
                        topk=3, retriever=None):
    """Full image -> class -> grounded explanation."""
    model, img_size = load_model(ckpt_path)
    class_id, confidence, top = predict(model, img_size, image_path, topk)

    r = retriever or Retriever()
    exp = explain(class_id, use_llm=use_llm, retriever=r)

    return {
        "image": os.path.basename(image_path),
        "class_id": class_id,
        "name": exp.get("name"),
        "confidence": round(confidence, 4),
        "topk": [{"class_id": c, "name": r.by_class_id(c)["name"] if r.by_class_id(c) else None,
                  "prob": round(p, 4)} for c, p in top],
        "explanation_source": exp.get("source"),
        "explanation": exp.get("explanation"),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Image -> sign class -> grounded explanation.")
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--no-llm", action="store_true", help="Skip Ollama, use template only.")
    return p.parse_args()


if __name__ == "__main__":
    import json
    args = parse_args()
    out = predict_and_explain(args.image, ckpt_path=args.ckpt,
                              use_llm=not args.no_llm, topk=args.topk)
    print(json.dumps(out, indent=2, ensure_ascii=False))
