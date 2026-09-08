"""
Part 1, section 6: self-supervised masked payment event modeling.

Usage:
    python -m train.pretrain --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from vulcan.common.config import load_config, config_hash, file_sha256, git_commit
from vulcan.common.checkpoint_naming import ckpt_name
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, assert_no_leaked_columns
from vulcan.data.chronological_split import chronological_split
from vulcan.data.windowing import build_windows, encode_windows
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.masking import apply_masking
from vulcan.models.pretraining_heads import (
    MaskedPretrainingHeads,
    masked_categorical_loss,
    masked_continuous_loss,
    next_event_type_loss,
)


def records_hash(records) -> str:
    canonical = json.dumps(records[:50] + records[-50:], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run(config_path: str, quick: bool = False):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_global_seed(seed)

    print(f"[1/6] Generating training data from IndiaPaymentSim (seed={seed})...")
    n_txn = cfg["data"]["n_transactions"] if not quick else 2000
    records = generate_training_data(cfg, seed=seed, n_transactions=n_txn)
    assert_no_leaked_columns(records)
    print(f"      generated {len(records)} transactions")

    print("[2/6] Chronological train/val/test split...")
    train, val, test = chronological_split(
        records,
        cfg["data"]["train_frac"],
        cfg["data"]["val_frac"],
        cfg["data"]["test_frac"],
    )
    print(f"      train={len(train)} val={len(val)} test={len(test)}")

    print("[3/6] Fitting tokenizer on train split...")
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train)

    history_len = cfg["model"]["history_len"]
    print(f"[4/6] Building sliding windows (history_len={history_len})...")
    train_windows = build_windows(train, history_len)
    val_windows = build_windows(val, history_len)
    print(f"      train_windows={len(train_windows)} val_windows={len(val_windows)}")

    train_cat, train_cont = encode_windows(tokenizer, train_windows)
    val_cat, val_cont = encode_windows(tokenizer, val_windows)

    field_names = list(tokenizer.categorical_fields.keys())

    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    heads = MaskedPretrainingHeads(tokenizer, model_cfg.d_model)
    n_params = backbone.num_parameters() + sum(p.numel() for p in heads.parameters())
    print(f"[5/6] Model built: {n_params:,} trainable parameters (size={cfg['model']['size']})")

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(heads.parameters()),
        lr=cfg["pretrain"]["lr"],
        weight_decay=cfg["pretrain"]["weight_decay"],
    )

    batch_size = cfg["pretrain"]["batch_size"]
    epochs = 1 if quick else cfg["pretrain"]["epochs"]
    mask_field_prob = cfg["pretrain"]["mask_field_prob"]
    mask_event_prob = cfg["pretrain"]["mask_event_prob"]
    next_event_weight = cfg["pretrain"]["next_event_loss_weight"]

    train_indices = TensorDataset(torch.arange(train_cont.shape[0]))
    loader = DataLoader(train_indices, batch_size=batch_size, shuffle=True)

    history = []
    print("[6/6] Training...")
    t0 = time.time()
    for epoch in range(epochs):
        backbone.train()
        heads.train()
        epoch_losses = []
        epoch_cat_acc = []
        epoch_next_acc = []
        for (idx_batch,) in loader:
            idx_batch = idx_batch
            batch_cat = {name: train_cat[name][idx_batch] for name in field_names}
            batch_cont = train_cont[idx_batch]

            masked_cat, masked_cont, cat_mask, cont_mask = apply_masking(
                batch_cat, batch_cont, tokenizer, mask_field_prob, mask_event_prob
            )

            z = backbone(masked_cat, masked_cont)
            cat_logits, cont_pred, next_event_logits = heads(z)

            loss_cat, accs = masked_categorical_loss(cat_logits, batch_cat, cat_mask)
            loss_cont = masked_continuous_loss(cont_pred, batch_cont, cont_mask)
            loss_next, next_acc = next_event_type_loss(next_event_logits, batch_cat["outcome_error_type"])

            loss = loss_cat + loss_cont + next_event_weight * loss_next

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(heads.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            if accs:
                epoch_cat_acc.append(sum(accs.values()) / len(accs))
            epoch_next_acc.append(next_acc)

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        mean_cat_acc = sum(epoch_cat_acc) / max(1, len(epoch_cat_acc)) if epoch_cat_acc else 0.0
        mean_next_acc = sum(epoch_next_acc) / max(1, len(epoch_next_acc))
        print(
            f"      epoch {epoch+1}/{epochs}  loss={mean_loss:.4f}  "
            f"masked_cat_acc={mean_cat_acc:.4f}  next_event_acc={mean_next_acc:.4f}"
        )
        history.append({"epoch": epoch + 1, "loss": mean_loss, "masked_cat_acc": mean_cat_acc, "next_event_acc": mean_next_acc})

    train_duration = time.time() - t0

    # ---- validation pass (masking applied once, no gradient) ----
    backbone.eval()
    heads.eval()
    with torch.no_grad():
        masked_cat, masked_cont, cat_mask, cont_mask = apply_masking(
            val_cat, val_cont, tokenizer, mask_field_prob, mask_event_prob
        )
        z = backbone(masked_cat, masked_cont)
        cat_logits, cont_pred, next_event_logits = heads(z)
        val_loss_cat, val_accs = masked_categorical_loss(cat_logits, val_cat, cat_mask)
        val_loss_cont = masked_continuous_loss(cont_pred, val_cont, cont_mask)
        val_loss_next, val_next_acc = next_event_type_loss(next_event_logits, val_cat["outcome_error_type"])
        val_cat_acc = sum(val_accs.values()) / len(val_accs) if val_accs else 0.0

    print(
        f"      VAL  masked_cat_acc={val_cat_acc:.4f}  "
        f"cont_mse={float(val_loss_cont):.4f}  next_event_acc={val_next_acc:.4f}"
    )

    # ---- persist checkpoint + manifest ----
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name("mini_vulcan_pretrained", cfg["model"]["size"])
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "model_cfg": cfg["model"],
        },
        ckpt_path,
    )
    ckpt_hash = file_sha256(ckpt_path)

    manifest = {
        "run_id": f"pretrain_{int(time.time())}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "seed": seed,
        "config_path": config_path,
        "config_hash": config_hash(cfg),
        "dataset_hash": records_hash(records),
        "model_checkpoint_sha256": ckpt_hash,
        "environment": {"torch_version": torch.__version__},
        "n_transactions": len(records),
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "n_parameters": n_params,
        "train_duration_sec": train_duration,
        "history": history,
        "final_metrics": {
            "train_loss": history[-1]["loss"] if history else None,
            "train_masked_cat_acc": history[-1]["masked_cat_acc"] if history else None,
            "val_masked_cat_acc": val_cat_acc,
            "val_continuous_mse": float(val_loss_cont),
            "val_next_event_acc": val_next_acc,
        },
    }

    artifacts_dir = Path("artifacts/runs")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / f"{manifest['run_id']}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCheckpoint saved to {ckpt_path} (sha256={ckpt_hash[:12]}...)")
    print(f"Manifest saved to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test run")
    args = parser.parse_args()
    run(args.config, quick=args.quick)
