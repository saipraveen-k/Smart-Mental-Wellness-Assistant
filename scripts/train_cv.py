#!/usr/bin/env python
"""Train the facial-expression CNN on FER-2013 (dynamic Hugging Face loading).

Pipeline:
    Hugging Face (CV_DATASET_ID: Jeneral/fer-2013)
        -> DatasetManager.load_cv_dataset()  (dynamic + validated)
        -> augmentation + normalisation
        -> CNN training (class-weighted CE)
        -> evaluation on test split (accuracy, P/R/F1, confusion matrix)
        -> artifacts/cv_emotion/{model.pt,labels.json,metrics.json}

Usage:
    python scripts/train_cv.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train facial-expression CNN on FER-2013")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = full split")
    p.add_argument("--limit-steps", type=int, default=0, help="debug: stop epoch after N steps")
    p.add_argument("--output", type=str, default="ml/artifacts/cv_emotion")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def load_data(args):
    from app.cv.augmentation import FERAugmenter
    from app.ml.dataset_manager import dataset_manager

    print("[1/6] Loading FER dataset dynamically from the Hugging Face Hub ...")
    t0 = time.time()
    data = dataset_manager.load_cv_dataset()
    prov = dataset_manager.get_status()["datasets"]["cv"]
    print(f"      dataset={prov['dataset_id']} revision={prov['revision']} "
          f"source={prov['source']} loader={prov['loader']} splits={prov['splits']}")
    print(f"      loaded in {time.time() - t0:.1f}s")

    print("[2/6] Validating ...")
    dataset_manager.validate_cv_dataset(data)
    return data, prov


def build_loaders(data, args, augmenter):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_x = data["train"]["images"]
    train_y = data["train"]["labels"]
    if args.max_train_samples:
        idx = np.random.RandomState(args.seed).permutation(len(train_y))[: args.max_train_samples]
        train_x, train_y = train_x[idx], train_y[idx]

    # Stratified 90/10 train/validation carve-out (model selection never
    # touches the held-out test split).
    from sklearn.model_selection import train_test_split

    tr_idx, va_idx = train_test_split(
        np.arange(len(train_y)), test_size=0.1, random_state=args.seed, stratify=train_y
    )
    va_x, va_y = train_x[va_idx], train_y[va_idx]
    train_x, train_y = train_x[tr_idx], train_y[tr_idx]

    # Normalised float tensors in [0, 1]
    xt = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1) / 255.0
    yt = torch.tensor(train_y, dtype=torch.long)
    xv = torch.tensor(va_x, dtype=torch.float32).unsqueeze(1) / 255.0
    yv = torch.tensor(va_y, dtype=torch.long)
    xe = torch.tensor(data["test"]["images"], dtype=torch.float32).unsqueeze(1) / 255.0
    ye = torch.tensor(data["test"]["labels"], dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(xt, yt)

    if augmenter is not None:
        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                img, label = self.base[i]
                arr = (img[0].numpy() * 255.0).astype(np.uint8)
                arr = augmenter(arr)
                return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0), label

        train_ds = AugmentedDataset(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(TensorDataset(xv, yv), batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(TensorDataset(xe, ye), batch_size=args.batch_size * 2, shuffle=False)
    return train_loader, val_loader, test_loader, len(train_y), len(va_y)


def evaluate(model, loader, device) -> Tuple[List[int], List[int], float]:
    import torch

    model.eval()
    preds: List[int] = []
    true: List[int] = []
    total_loss = 0.0
    n = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y).item())
            n += y.size(0)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            true.extend(y.cpu().tolist())
    return preds, true, total_loss / max(n, 1)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    import torch

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    from app.config import settings
    from app.cv.augmentation import FERAugmenter
    from app.cv.model import FacialEmotionCNN, build_class_weights
    from app.ml.emotion_mapping import FER2013_LABELS
    from app.ml.evaluation import compute_classification_metrics

    device = torch.device("cuda" if torch.cuda.is_available() and settings.model_device != "cpu" else "cpu")
    print(f"device={device} | torch threads={torch.get_num_threads()}")

    data, prov = load_data(args)
    augmenter = None if args.no_augment else FERAugmenter()

    print("[3/6] Building loaders (stratified 90/10 train/val carve-out) ...")
    train_loader, val_loader, test_loader, n_train, n_val = build_loaders(data, args, augmenter)
    print(f"      train={n_train} val={n_val} test={len(data['test']['labels'])} batch={args.batch_size}")

    print("[4/6] Building model ...")
    class_weights = build_class_weights(data["train"]["labels"], len(FER2013_LABELS))
    weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    model = FacialEmotionCNN(num_classes=len(FER2013_LABELS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      FacialEmotionCNN params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )
    criterion_base = torch.nn.CrossEntropyLoss(weight=weights_t)

    print(f"[5/6] Training: {args.epochs} epoch(s), lr={args.lr}, augment={'off' if args.no_augment else 'on'}")
    history = []
    best_test_f1 = -1.0
    best_state = None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        t_ep = time.time()
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion_base(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * y.size(0)
            seen += y.size(0)
            if step % 40 == 0 or step == len(train_loader):
                print(f"      epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / seen:.4f}", flush=True)
            if args.limit_steps and step >= args.limit_steps:
                break
        preds, true, val_loss = evaluate(model, val_loader, device)
        metrics = compute_classification_metrics(true, preds, FER2013_LABELS)
        scheduler.step(metrics["macro_f1"])
        entry = {
            "epoch": epoch,
            "train_loss": round(running / max(seen, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "duration_s": round(time.time() - t_ep, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        print(f"      epoch {epoch}: {entry}", flush=True)
        if metrics["macro_f1"] > best_test_f1:
            best_test_f1 = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[6/6] Final evaluation ...")
    preds, true, test_loss = evaluate(model, test_loader, device)
    test_metrics = compute_classification_metrics(true, preds, FER2013_LABELS)

    # --- Save artifacts -------------------------------------------------
    out_dir = REPO_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": model.arch_config(),
            "num_classes": len(FER2013_LABELS),
        },
        out_dir / "model.pt",
    )

    labels_doc = {
        "classes": list(FER2013_LABELS),
        "id2label": {str(i): n for i, n in enumerate(FER2013_LABELS)},
        "label2id": {n: i for i, n in enumerate(FER2013_LABELS)},
        "wellness_mapping": {
            "happy": "Joy", "sad": "Sadness", "angry": "Anger", "fear": "Anxiety",
            "neutral": "Neutral", "disgust": "Neutral/Other", "surprise": "Neutral/Other",
        },
        "note": (
            "Facial-expression analysis only: these labels describe visible "
            "expressions, not a person's true internal emotional state."
        ),
        "dataset": prov,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels_doc, indent=2), encoding="utf-8")

    metrics = {
        "model": "FacialEmotionCNN",
        "parameters": n_params,
        "fine_tuned_on": prov["dataset_id"],
        "dataset_revision": prov["revision"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "augmentation": not args.no_augment,
            "class_weights": [round(w, 4) for w in class_weights],
            "seed": args.seed,
            "device": str(device),
            "train_seconds": round(train_time, 1),
            "history": history,
        },
        "test_metrics": test_metrics,
        "best_epoch_by_macro_f1": int(np.argmax([h["val_macro_f1"] for h in history]) + 1)
        if history
        else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tm = test_metrics
    print("\n===== CV MODEL RESULTS (test split) =====")
    print(f"accuracy     : {tm['accuracy']:.4f}")
    print(f"macro F1     : {tm['macro_f1']:.4f}")
    print(f"weighted F1  : {tm['weighted_f1']:.4f}")
    for name, m in tm["per_class"].items():
        print(f"  {name:<9} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
