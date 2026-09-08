"""
Part 2, section 7: downstream Mini-Vulcan heads (Baseline B).

Loads the pretrained backbone from Part 1, freezes it, and trains
per-route outcome heads using bandit feedback (only the taken route's
label is available per example). Persists a manifest + checkpoint, and
also a small "reference window" artifact of embeddings/predictions from a
stable period, needed to bootstrap DriftSafe (Part 2, sections 11-12).

Usage:
    python -m train.train_baseline_heads --config configs/tiny.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from vulcan.common.config import load_config, config_hash, file_sha256, git_commit
from vulcan.common.checkpoint_naming import ckpt_name
from vulcan.common.batched_inference import batched_current_state
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data, assert_no_leaked_columns
from vulcan.data.chronological_split import chronological_split
from vulcan.data.windowing import build_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import (
    DownstreamHeads,
    masked_bandit_bce_loss,
    masked_latency_mse_loss,
    greedy_utility_select,
)


def _labels_from_windows(windows, key):
    return torch.tensor([float(w[-1][key]) for w in windows], dtype=torch.float32)


def _bool_labels_from_windows(windows, key):
    return torch.tensor([float(bool(w[-1][key])) for w in windows], dtype=torch.float32)


def run(config_path: str, quick: bool = False):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_global_seed(seed)

    n_txn = cfg["data"]["n_transactions"] if not quick else 2000
    records = generate_training_data(cfg, seed=seed, n_transactions=n_txn)
    assert_no_leaked_columns(records)
    train, val, test = chronological_split(records, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["data"]["test_frac"])

    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train)
    history_len = cfg["model"]["history_len"]

    train_windows = build_windows(train, history_len, stride=max(1, history_len // 8))
    val_windows = build_windows(val, history_len, stride=max(1, history_len // 4))

    # Masked encoding, so the champion is trained on decision-time
    # information only -- consistent with every downstream evaluation and
    # adaptation path that later measures and repairs it.
    train_cat, train_cont = encode_windows_for_world_model(tokenizer, train_windows)
    val_cat, val_cont = encode_windows_for_world_model(tokenizer, val_windows)

    # zero out mask flags (no masking at fine-tune time): concat zeros
    def add_mask_channel(cont):
        return torch.cat([cont, torch.zeros_like(cont)], dim=-1)

    train_cont_full = add_mask_channel(train_cont)
    val_cont_full = add_mask_channel(val_cont)

    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    ckpt_path = Path("checkpoints") / ckpt_name("mini_vulcan_pretrained", cfg["model"]["size"])
    if ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        backbone.load_state_dict(state["backbone_state_dict"])
        print(f"Loaded pretrained backbone from {ckpt_path}")
    else:
        print("WARNING: no pretrained checkpoint found, training backbone from random init")

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    num_routes = cfg["simulator"]["num_routes"]
    heads = DownstreamHeads(model_cfg.d_model, num_routes)

    taken_route_train = torch.tensor([w[-1]["action_route_id"] for w in train_windows], dtype=torch.long)
    taken_route_val = torch.tensor([w[-1]["action_route_id"] for w in val_windows], dtype=torch.long)

    success_train = _bool_labels_from_windows(train_windows, "outcome_success")
    fraud_train = torch.tensor([float(w[-1]["outcome_fraud_loss"] > 0) for w in train_windows], dtype=torch.float32)
    abandon_train = _bool_labels_from_windows(train_windows, "outcome_abandoned")
    log_latency_train = torch.tensor([np.log1p(w[-1]["outcome_latency_ms"]) for w in train_windows], dtype=torch.float32)

    success_val = _bool_labels_from_windows(val_windows, "outcome_success")
    fraud_val = torch.tensor([float(w[-1]["outcome_fraud_loss"] > 0) for w in val_windows], dtype=torch.float32)
    abandon_val = _bool_labels_from_windows(val_windows, "outcome_abandoned")
    log_latency_val = torch.tensor([np.log1p(w[-1]["outcome_latency_ms"]) for w in val_windows], dtype=torch.float32)

    with torch.no_grad():
        z_train_all = batched_current_state(backbone, train_cat, train_cont_full)
        z_val_all = batched_current_state(backbone, val_cat, val_cont_full)

    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = cfg["pretrain"]["batch_size"]
    epochs = 1 if quick else 8

    ds = TensorDataset(torch.arange(z_train_all.shape[0]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        heads.train()
        losses = []
        for (idx,) in loader:
            z = z_train_all[idx]
            out = heads(z)
            loss = (
                masked_bandit_bce_loss(out["success_logit"], success_train[idx], taken_route_train[idx])
                + masked_bandit_bce_loss(out["fraud_logit"], fraud_train[idx], taken_route_train[idx])
                + masked_bandit_bce_loss(out["abandon_logit"], abandon_train[idx], taken_route_train[idx])
                + 0.1 * masked_latency_mse_loss(out["log_latency"], log_latency_train[idx], taken_route_train[idx])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch {epoch+1}/{epochs} loss={mean_loss:.4f}")

    heads.eval()
    with torch.no_grad():
        val_out = heads(z_val_all)
        idx = torch.arange(z_val_all.shape[0])
        p_success_taken = torch.sigmoid(val_out["success_logit"])[idx, taken_route_val]
        val_bce = torch.nn.functional.binary_cross_entropy(p_success_taken, success_val).item()
        val_acc = ((p_success_taken > 0.5).float() == success_val).float().mean().item()

    print(f"VAL success BCE={val_bce:.4f} acc={val_acc:.4f}")

    # ---- save reference window artifact for DriftSafe bootstrapping ----
    ref_dir = Path("artifacts/reference_window")
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_n = min(500, z_val_all.shape[0])
    np.savez(
        ref_dir / "reference.npz",
        features=np.array([w[-1]["amount"] for w in val_windows[:ref_n]]),
        z=z_val_all[:ref_n].numpy(),
        probs=p_success_taken[:ref_n].numpy(),
        labels=success_val[:ref_n].numpy(),
    )

    ckpt_out = Path("checkpoints") / ckpt_name("baseline_heads", cfg["model"]["size"])
    torch.save({"heads_state_dict": heads.state_dict(), "model_cfg": cfg["model"], "num_routes": num_routes}, ckpt_out)
    ckpt_hash = file_sha256(ckpt_out)

    manifest = {
        "run_id": f"baseline_heads_{int(time.time())}",
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "checkpoint_sha256": ckpt_hash,
        "history": history,
        "final_metrics": {"val_success_bce": val_bce, "val_success_acc": val_acc},
    }
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    with open(f"artifacts/runs/{manifest['run_id']}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {ckpt_out} and manifest.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.config, quick=args.quick)
