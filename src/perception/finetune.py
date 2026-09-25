"""
Phase 1 - Perception: fine-tune MobileNetV3-Small on GTSRB (CPU-friendly).

Example (from the repo root, on your machine):

    python src/perception/finetune.py \
        --data-root /path/to/GTSRB \
        --img-size 64 --epochs 12 --batch-size 64 \
        --freeze last_block --out checkpoints/perception_mobilenetv3.pt

Windows / Spyder: run as a script (the __main__ guard below makes num_workers
safe). Default num_workers=0 avoids the Windows DataLoader multiprocessing pitfalls.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Allow "python src/perception/finetune.py" and "python -m src.perception.finetune".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_dataloaders, NUM_CLASSES          # noqa: E402
from model import build_model, count_parameters, model_size_mb  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MobileNetV3-Small on GTSRB.")
    p.add_argument("--data-root", required=True, help="GTSRB root (contains Train.csv/Test.csv).")
    p.add_argument("--train-csv", default="Train.csv")
    p.add_argument("--test-csv", default="Test.csv")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze", choices=["all", "last_block", "none"], default="last_block")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-roi", action="store_true", help="Disable ROI cropping.")
    p.add_argument("--no-pretrained", action="store_true", help="Random init (ablation).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/perception_mobilenetv3.pt")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # this project targets CPU-only training
    print(f"[setup] device={device} | torch={torch.__version__}")

    train_loader, val_loader, test_loader = build_dataloaders(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_roi=not args.no_roi,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        seed=args.seed,
    )
    print(f"[data] train batches={len(train_loader)} "
          f"val batches={len(val_loader)} test batches={len(test_loader)}")

    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrained, freeze=args.freeze).to(device)
    total, trainable = count_parameters(model)
    print(f"[model] MobileNetV3-Small | params={total:,} trainable={trainable:,} "
          f"({100*trainable/total:.1f}%) | fp32 size={model_size_mb(model):.2f} MB")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, seen, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        scheduler.step()
        train_loss, train_acc = run_loss / seen, correct / seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        dt = time.perf_counter() - t0
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, seconds=round(dt, 1)))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "img_size": args.img_size,
                "arch": "mobilenet_v3_small",
                "freeze": args.freeze,
                "val_acc": val_acc,
            }, args.out)
            print(f"           -> saved best checkpoint (val_acc={val_acc:.4f}) to {args.out}")

    # Final test-set report using the best checkpoint.
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"[done] best_val_acc={best_val_acc:.4f} | test_acc={test_acc:.4f}")

    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dict(best_val_acc=best_val_acc, test_acc=test_acc,
                       args=vars(args), history=history), f, indent=2)
    print(f"[done] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
