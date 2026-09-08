"""
Temporal target leakage, specifically in the Forge pipeline's downstream
head/adapter training and evaluation -- vulcan/adaptation/candidate_trainer.py,
vulcan/adaptation/champion_challenger.py, and vulcan/forge/forge_loop.py.

This is a DIFFERENT (but mechanistically identical) property from
tests/data/test_no_temporal_leakage.py, which only ever tested the world
model's encoding path. This file exists because an external review
correctly identified that the SAME leak (the last window position's own
action_route_id/outcome_success/outcome_latency_ms/outcome_fraud_loss/
outcome_abandoned fields were visible to the model at exactly the position
it was being trained/evaluated to predict them from) was present throughout
the entire Forge pipeline -- diagnosis-window construction, candidate
training, champion/challenger evaluation, blind-spot mining, red-team
scoring, and failure-memory checks -- and had NO test coverage of its own.

The bug was confirmed empirically before the fix: a challenger trained via
the (then-unmasked) candidate_trainer.py scored 86.11% accuracy on a
held-out slice; the identical challenger, evaluated with masked encoding,
scored 72.22% -- a 13.9-point gap driven entirely by encoding, not by any
real improvement in the model. Fixed by routing all three modules through
`encode_windows_for_world_model` instead of the unmasked `encode_windows`.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import torch

from vulcan.common.config import load_config
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data
from vulcan.data.chronological_split import chronological_split
from vulcan.data.windowing import build_windows, encode_windows, encode_windows_for_world_model
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.adaptation.candidate_trainer import train_candidate
from vulcan.adaptation.champion_challenger import evaluate_model_on_slice


def _imported_names(module) -> set:
    tree = ast.parse(open(inspect.getsourcefile(module)).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_candidate_trainer_does_not_import_leaky_encode_windows():
    import vulcan.adaptation.candidate_trainer as m
    names = _imported_names(m)
    assert "encode_windows" not in names, (
        "candidate_trainer.py must not import the unmasked encode_windows (temporal leakage)"
    )
    assert "encode_windows_for_world_model" in names


def test_champion_challenger_does_not_import_leaky_encode_windows():
    import vulcan.adaptation.champion_challenger as m
    names = _imported_names(m)
    assert "encode_windows" not in names
    assert "encode_windows_for_world_model" in names


def test_forge_loop_does_not_import_leaky_encode_windows():
    import vulcan.forge.forge_loop as m
    names = _imported_names(m)
    assert "encode_windows" not in names, (
        "forge_loop.py must not import the unmasked encode_windows "
        "(used by make_predict_fn for blind-spot mining, red-team scoring, "
        "and failure-memory checks)"
    )
    assert "encode_windows_for_world_model" in names


def test_fresh_challenger_accuracy_is_encoding_invariant():
    """A challenger trained via the FIXED candidate_trainer.py should show
    a negligible accuracy gap between masked and unmasked evaluation
    encoding, since it was never trained with access to the leaked fields.
    Before the fix this same setup produced a 13.9-point gap (86.11% vs
    72.22%). A 5-point tolerance is used; a reintroduced leak would blow
    well past it."""
    cfg = load_config("configs/tiny.yaml")
    set_global_seed(cfg["seed"])
    records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=2000)
    train, val, test = chronological_split(records, 0.7, 0.15, 0.15)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    heads.eval()

    history_len = cfg["model"]["history_len"]
    train_windows = build_windows(train, history_len, stride=4)[:100]
    replay_windows = build_windows(train, history_len, stride=4)[100:200]

    candidate = train_candidate(
        backbone, tokenizer, heads.state_dict(), drift_windows=train_windows, replay_windows=replay_windows,
        num_routes=cfg["simulator"]["num_routes"], d_model=model_cfg.d_model, epochs=15, seed=1,
    )

    eval_windows = build_windows(val, history_len, stride=max(1, history_len // 4))[:200]
    masked_acc = evaluate_model_on_slice(backbone, candidate.adapter, candidate.heads, tokenizer, eval_windows)["accuracy"]

    cat, cont = encode_windows(tokenizer, eval_windows)
    cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
    with torch.no_grad():
        z = backbone.current_state(cat, cont_full)
        z = candidate.adapter(z)
        out = candidate.heads(z)
        idx = torch.arange(z.shape[0])
        taken = torch.tensor([w[-1]["action_route_id"] for w in eval_windows], dtype=torch.long)
        leaky_probs = torch.sigmoid(out["success_logit"])[idx, taken].numpy()
    labels = np.array([float(bool(w[-1]["outcome_success"])) for w in eval_windows])
    leaky_acc = float(((leaky_probs > 0.5).astype(float) == labels).mean())

    gap = abs(masked_acc - leaky_acc)
    assert gap < 0.05, (
        f"Accuracy gap between masked (production) and leaky encoding is {gap:.3f}, "
        f"expected < 0.05 for a challenger that was never trained with leak access. "
        f"masked_acc={masked_acc:.4f} leaky_acc={leaky_acc:.4f}. This suggests the "
        f"leak has been reintroduced somewhere in the training or evaluation path."
    )
