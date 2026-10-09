#!/usr/bin/env python
"""Train the DistilBERT text-emotion model.

Pipeline:
    Hugging Face (TEXT_DATASET_ID)
        -> DatasetManager.load_text_dataset()  (dynamic + validated)
        -> text_preprocessing.prepare_split()   (documented label mapping)
        -> DistilBERT fine-tune (class-weighted CE)
        -> evaluation on validation/test (accuracy, P/R/F1, confusion matrix)
        -> artifacts/text_emotion/{model,tokenizer,labels.json,metrics.json}

Usage:
    python scripts/train_text.py --epochs 2 --batch-size 32
    python scripts/train_text.py --help
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DistilBERT on GoEmotions -> wellness taxonomy")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 = use full training split")
    p.add_argument("--limit-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--output", type=str, default="ml/artifacts/text_emotion")
    p.add_argument("--model-name", type=str, default=None, help="defaults to TEXT_MODEL_NAME env/config")
    p.add_argument("--num-threads", type=int, default=0, help="0 = torch default")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def build_datasets(args):
    from app.ml.dataset_manager import dataset_manager
    from app.ml.text_preprocessing import class_distribution, compute_class_weights, prepare_split

    print("[1/6] Loading dataset dynamically from the Hugging Face Hub ...")
    t0 = time.time()
    raw = dataset_manager.load_text_dataset()
    prov = dataset_manager.get_status()["datasets"]["text"]
    print(f"      dataset={prov['dataset_id']} revision={prov['revision']} "
          f"source={prov['source']} loader={prov['loader']} splits={prov['splits']}")
    print(f"      loaded in {time.time() - t0:.1f}s (cached locally after this)")

    print("[2/6] Validating + preprocessing ...")
    dataset_manager.validate_text_dataset(raw)
    prepared = {}
    report: Dict[str, Dict] = {}
    for split in ("train", "validation", "test"):
        samples, drops = prepare_split(raw[split], split)
        prepared[split] = samples
        report[split] = {"kept": len(samples), "dropped": drops,
                         "distribution": class_distribution(samples)}
        print(f"      {split:>10}: kept={len(samples):>6} dist={report[split]['distribution']} drops={drops}")

    weights = compute_class_weights(prepared["train"])
    print(f"      class weights (Joy..Neutral): {weights}")
    return prepared, weights, prov, report


def tokenize_splits(prepared, tokenizer, max_length: int) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for split, samples in prepared.items():
        texts = [s["text"] for s in samples]
        labels = np.array([s["label"] for s in samples], dtype=np.int64)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
        out[split] = {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
            "labels": labels,
        }
        print(f"      tokenized {split}: {out[split]['input_ids'].shape}")
    return out


def evaluate(model, loader, device, desc: str = "eval") -> Dict:
    import torch

    model.eval()
    all_preds: List[int] = []
    all_true: List[int] = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(out.loss.item()) * labels.size(0)
            n += labels.size(0)
            preds = out.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())
    return {"loss": round(total_loss / max(n, 1), 4), "y_true": all_true, "y_pred": all_preds}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    from app.config import settings
    from app.ml.emotion_mapping import WELLNESS_LABELS, WELLNESS_ID2LABEL, EMOTION_MAPPING
    from app.ml.evaluation import compute_classification_metrics

    device = torch.device("cuda" if torch.cuda.is_available() and settings.model_device != "cpu" else "cpu")
    print(f"device={device} | torch threads={torch.get_num_threads()}")

    prepared, class_weights, prov, prep_report = build_datasets(args)

    print("[3/6] Loading tokenizer + model ...")
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    model_name = args.model_name or settings.text_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2id = {name: i for i, name in enumerate(WELLNESS_LABELS)}
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(WELLNESS_LABELS),
        id2label={i: n for i, n in enumerate(WELLNESS_LABELS)},
        label2id=label2id,
    )
    model.to(device)

    print("[4/6] Tokenizing ...")
    tokenized = tokenize_splits(prepared, tokenizer, args.max_length)

    def make_loader(split: str, batch_size: int, shuffle: bool):
        arr = tokenized[split]
        ids = torch.tensor(arr["input_ids"], dtype=torch.long)
        mask = torch.tensor(arr["attention_mask"], dtype=torch.long)
        lbls = torch.tensor(arr["labels"], dtype=torch.long)
        ds = TensorDataset(ids, mask, lbls)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader("train", args.batch_size, True)
    val_loader = make_loader("validation", args.eval_batch_size, False)

    total_steps = len(train_loader) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if args.max_train_samples:
        print(f"      (debug) limiting train split to {args.max_train_samples} samples")
    print(f"[5/6] Training: {args.epochs} epoch(s), ~{total_steps} steps, "
          f"batch={args.batch_size}, lr={args.lr}")

    history = []
    best_val_macro_f1 = -1.0
    best_state = None
    t_start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        t_epoch = time.time()
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            if step % 50 == 0 or step == len(train_loader):
                lr_now = scheduler.get_last_lr()[0]
                print(f"      epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={epoch_loss / seen:.4f} lr={lr_now:.2e}", flush=True)
            if args.limit_steps and step >= args.limit_steps:
                break
        val = evaluate(model, val_loader, device)
        from app.ml.evaluation import compute_classification_metrics as ccm

        val_metrics = ccm(val["y_true"], val["y_pred"], WELLNESS_LABELS)
        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(seen, 1), 4),
            "val_loss": val["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "steps": global_step,
            "duration_s": round(time.time() - t_epoch, 1),
        }
        history.append(entry)
        print(f"      epoch {epoch} done: {entry}", flush=True)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"      new best val macro_f1={best_val_macro_f1:.4f} (checkpoint kept in memory)")

    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    print("[6/6] Final evaluation on test split ...")
    test = evaluate(model, make_loader("test", args.eval_batch_size, False), device)
    test_metrics = compute_classification_metrics(test["y_true"], test["y_pred"], WELLNESS_LABELS)
    val_final = evaluate(model, val_loader, device)
    val_metrics = compute_classification_metrics(val_final["y_true"], val_final["y_pred"], WELLNESS_LABELS)

    # --- Save artifacts -------------------------------------------------
    out_dir = REPO_ROOT / args.output
    model_dir = out_dir / "model"
    tok_dir = out_dir / "tokenizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tok_dir)

    labels_doc = {
        "label2id": {n: i for i, n in enumerate(WELLNESS_LABELS)},
        "id2label": {str(i): n for i, n in enumerate(WELLNESS_LABELS)},
        "source_labels": "GoEmotions (28 labels)",
        "emotion_mapping": EMOTION_MAPPING,
        "ambiguity_policy": (
            "Rows whose active GoEmotions labels resolve to >=2 wellness classes are "
            "excluded from training (ambiguous), never silently relabelled."
        ),
        "dataset": prov,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels_doc, indent=2), encoding="utf-8")

    metrics = {
        "model": model_name,
        "base_model": model_name,
        "fine_tuned_on": prov["dataset_id"],
        "dataset_revision": prov["revision"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "class_weights": dict(zip(WELLNESS_LABELS, class_weights)),
            "seed": args.seed,
            "device": str(device),
            "total_steps": global_step,
            "train_seconds": round(train_time, 1),
            "history": history,
        },
        "preprocessing_report": prep_report,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tm = test_metrics
    print("\n===== TEXT MODEL RESULTS (test split) =====")
    print(f"accuracy     : {tm['accuracy']:.4f}")
    print(f"macro F1     : {tm['macro_f1']:.4f}")
    print(f"weighted F1  : {tm['weighted_f1']:.4f}")
    for name, m in tm["per_class"].items():
        print(f"  {name:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
