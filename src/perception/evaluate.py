"""
Phase 1 - Perception: evaluate a trained checkpoint.

Reports test accuracy, parameter count, on-disk size, and per-image CPU latency.
Optionally applies INT8 dynamic quantization and re-measures, and/or exports ONNX
for the deployment / serving bridge.

    python src/perception/evaluate.py \
        --data-root /path/to/GTSRB \
        --ckpt checkpoints/perception_mobilenetv3.pt \
        --quantize --onnx checkpoints/perception_mobilenetv3.onnx

Honest note (documented, not hidden): dynamic quantization only quantizes Linear
layers. MobileNetV3 is convolution-heavy, so INT8 dynamic quant mainly reduces
model size; the latency change depends on the CPU/backend and is often small or
even negative on desktop x86. The reliable, reportable win is the size reduction.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_dataloaders, NUM_CLASSES          # noqa: E402
from model import build_model, count_parameters, model_size_mb  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a GTSRB perception checkpoint.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-csv", default="Test.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-roi", action="store_true")
    p.add_argument("--quantize", action="store_true", help="Also evaluate INT8 dynamic quant.")
    p.add_argument("--onnx", default=None, help="Path to export ONNX (optional).")
    p.add_argument("--latency-iters", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def per_image_latency_ms(model, img_size, iters=200, warmup=20):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters * 1000.0


def file_size_kb(path):
    return os.path.getsize(path) / 1024.0 if os.path.isfile(path) else float("nan")


def main():
    args = parse_args()
    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = ckpt.get("img_size", 64)

    model = build_model(ckpt.get("num_classes", NUM_CLASSES), pretrained=False,
                        freeze="none")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = build_dataloaders(
        root=args.data_root, img_size=img_size, batch_size=args.batch_size,
        val_split=0.0, num_workers=args.num_workers, use_roi=not args.no_roi,
        test_csv=args.test_csv,
    )

    total, _ = count_parameters(model)
    acc = test_accuracy(model, test_loader, device)
    lat = per_image_latency_ms(model, img_size, args.latency_iters)
    print("=== FP32 ===")
    print(f"params        : {total:,}")
    print(f"size (in-mem) : {model_size_mb(model):.2f} MB")
    print(f"test accuracy : {acc*100:.2f}%")
    print(f"CPU latency   : {lat:.3f} ms/image")

    if args.quantize:
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8)
        qacc = test_accuracy(qmodel, test_loader, device)
        qlat = per_image_latency_ms(qmodel, img_size, args.latency_iters)
        tmp = "checkpoints/_int8_tmp.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(qmodel.state_dict(), tmp)
        print("=== INT8 (dynamic, Linear only) ===")
        print(f"state size    : {file_size_kb(tmp):.0f} KB "
              f"(fp32 state {file_size_kb(args.ckpt):.0f} KB)")
        print(f"test accuracy : {qacc*100:.2f}%  (delta {100*(qacc-acc):+.2f} pt)")
        print(f"CPU latency   : {qlat:.3f} ms/image")
        print("note: MobileNet is conv-heavy; dynamic INT8 mainly shrinks model size; "
              "latency change depends on the CPU/backend.")
        os.remove(tmp)

    if args.onnx:
        dummy = torch.randn(1, 3, img_size, img_size)
        torch.onnx.export(
            model, dummy, args.onnx, opset_version=13,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"[onnx] exported to {args.onnx} ({file_size_kb(args.onnx):.0f} KB)")


if __name__ == "__main__":
    main()
