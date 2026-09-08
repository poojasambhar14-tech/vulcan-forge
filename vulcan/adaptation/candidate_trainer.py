"""
Candidate (challenger) trainer (master spec section 13).

Freezes the pretrained backbone entirely. Trains only a BottleneckAdapter
applied to z_t, plus a fresh copy of the downstream heads, on a training set
built from:
    recent drift buffer (post-shift windows)
    + optionally a historical replay buffer (pre-shift windows)

Omitting the replay buffer is intentionally supported (and used by the demo
to construct a deliberately BAD candidate that overfits to the recent
window and forgets the historical regime -- see scripts/demo_part3.py).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vulcan.models.mini_vulcan import MiniVulcanBackbone
from vulcan.models.downstream_heads import DownstreamHeads, masked_bandit_bce_loss, masked_latency_mse_loss
from vulcan.adaptation.adapter import BottleneckAdapter
from vulcan.common.config import config_hash


@dataclass
class CandidateArtifacts:
    adapter: BottleneckAdapter
    heads: DownstreamHeads
    metrics: Dict[str, Any]
    manifest: Dict[str, Any]


def _labels(windows, key, is_bool=True):
    if is_bool:
        return torch.tensor([float(bool(w[-1][key])) for w in windows], dtype=torch.float32)
    return torch.tensor([float(w[-1][key]) for w in windows], dtype=torch.float32)


def _prepare_batch(backbone: MiniVulcanBackbone, tokenizer, windows):
    from vulcan.data.windowing import encode_windows_for_world_model

    # INVARIANT: the model must never see a transaction's own outcome at
    # the position it is being trained to predict from. Masked encoding
    # strips action_route_id / outcome_success / outcome_latency_ms /
    # outcome_fraud_loss / outcome_abandoned from the final window position.
    # Labels below still read from the ORIGINAL windows -- only the encoded
    # representation the model sees is masked.
    # Guarded by tests/forge/test_no_forge_leakage.py.
    cat, cont = encode_windows_for_world_model(tokenizer, windows)
    cont_full = torch.cat([cont, torch.zeros_like(cont)], dim=-1)
    with torch.no_grad():
        z = backbone.current_state(cat, cont_full)
    taken_route = torch.tensor([w[-1]["action_route_id"] for w in windows], dtype=torch.long)
    success = _labels(windows, "outcome_success")
    fraud = torch.tensor([float(w[-1]["outcome_fraud_loss"] > 0) for w in windows], dtype=torch.float32)
    abandon = _labels(windows, "outcome_abandoned")
    log_latency = torch.tensor([np.log1p(w[-1]["outcome_latency_ms"]) for w in windows], dtype=torch.float32)
    return z, taken_route, success, fraud, abandon, log_latency


def train_candidate(
    backbone: MiniVulcanBackbone,
    tokenizer,
    base_heads_state: dict,
    drift_windows: List[list],
    replay_windows: Optional[List[list]],
    num_routes: int,
    d_model: int,
    epochs: int = 15,
    lr: float = 2e-3,
    bottleneck_dim: int = 16,
    weight_decay: float = 1e-4,
    seed: int = 0,
    parent_checkpoint_hash: str = "",
    base_adapter_state: Optional[dict] = None,
    route_anchors: Optional[List[tuple]] = None,
    route_anchor_weight: float = 1.0,
) -> CandidateArtifacts:
    torch.manual_seed(seed)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    adapter = BottleneckAdapter(d_model, bottleneck_dim)
    # ADAPTER LINEAGE. Previously this was always a fresh, randomly-initialized
    # adapter, so every healing cycle threw away the previous generation's
    # repair and re-derived one from scratch. That is clean-slate REPLACEMENT,
    # not incremental healing: if generation 1's adapter fixed an HDFC blind
    # spot and generation 2 was promoted for fixing something unrelated,
    # generation 2 was never built on top of generation 1's repair. Worse, the
    # heads DID carry forward (below) -- so heads trained jointly with adapter
    # A1 were being placed behind a different, randomly-initialized A2, a
    # genuine representation mismatch.
    #
    # Warm-starting from the champion's installed adapter is standard practice
    # in the continual/multi-round adapter literature: the new repair begins as
    # a small perturbation of the one already certified, rather than being
    # re-derived from replay data alone. Deep-copied so training can never
    # mutate the live champion's weights in place.
    if base_adapter_state is not None:
        adapter.load_state_dict(copy.deepcopy(base_adapter_state))
    heads = DownstreamHeads(d_model, num_routes)
    heads.load_state_dict(copy.deepcopy(base_heads_state))  # start from champion heads, then specialize

    training_windows = list(drift_windows)
    if replay_windows:
        training_windows = training_windows + list(replay_windows)

    z, taken_route, success, fraud, abandon, log_latency = _prepare_batch(backbone, tokenizer, training_windows)

    # ---- ROUTE-PREFERENCE ANCHORS (training/certification alignment fix) ----
    # PROBLEM THIS SOLVES: failure-memory scenarios were replayed into the
    # bandit losses above, which train "did the taken route succeed?" -- a
    # BINARY OUTCOME signal. But certification tests a ROUTING decision: does
    # the challenger still send this payment to a materially worse route than
    # the certified champion did? Those are different properties, so a
    # challenger could be trained on every remembered failure and still fail
    # 8/8 of them at the gate (observed exactly, seed 2024 round 3). The model
    # was structurally unable to learn its way past the criterion it was
    # judged on.
    #
    # WHY NOT JUST TRAIN ON THE ORACLE: the route invariant stores the
    # simulator's true per-route success probabilities. Training on those
    # would be a NEW leak -- privileged information that does not exist at
    # deployment time -- and would invalidate every number the same way the
    # encoding leak did. Deliberately not done.
    #
    # WHAT THIS DOES INSTEAD: pure self-distillation from the CERTIFIED
    # CHAMPION's own past decision. Each anchor is (window, champion_route,
    # bad_route) -- both observed, neither privileged. A pairwise margin loss
    # pushes the challenger's score for the champion's route above its score
    # for the route the rejected challenger picked. This is exactly the
    # invariant certification checks, expressed as a differentiable objective,
    # using only information the system legitimately has.
    anchor_z = anchor_good = anchor_bad = None
    if route_anchors:
        anchor_windows = [w for (w, _g, _b) in route_anchors]
        anchor_z, *_ = _prepare_batch(backbone, tokenizer, anchor_windows)
        anchor_good = torch.tensor([g for (_w, g, _b) in route_anchors], dtype=torch.long)
        anchor_bad = torch.tensor([b for (_w, _g, b) in route_anchors], dtype=torch.long)

    trainable_params = list(adapter.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    loss_history = []
    for epoch in range(epochs):
        adapter.train()
        heads.train()
        z_adapted = adapter(z)
        out = heads(z_adapted)
        loss = (
            masked_bandit_bce_loss(out["success_logit"], success, taken_route)
            + masked_bandit_bce_loss(out["fraud_logit"], fraud, taken_route)
            + masked_bandit_bce_loss(out["abandon_logit"], abandon, taken_route)
            + 0.1 * masked_latency_mse_loss(out["log_latency"], log_latency, taken_route)
        )
        if anchor_z is not None:
            a_out = heads(adapter(anchor_z))
            a_logits = a_out["success_logit"]
            idx = torch.arange(a_logits.shape[0])
            good_score = a_logits[idx, anchor_good]
            bad_score = a_logits[idx, anchor_bad]
            # margin ranking: champion's route must outscore the bad route by
            # at least `margin`; zero loss once that already holds.
            margin = 0.5
            loss = loss + route_anchor_weight * torch.clamp(margin - (good_score - bad_score), min=0.0).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    n_trainable = adapter.num_trainable_parameters() + sum(p.numel() for p in heads.parameters())
    n_backbone = sum(p.numel() for p in backbone.parameters())

    manifest = {
        "candidate_id": f"candidate_{int(time.time()*1000)}",
        "parent_checkpoint_sha256": parent_checkpoint_hash,
        "seed": seed,
        "n_drift_windows": len(drift_windows),
        "n_replay_windows": len(replay_windows) if replay_windows else 0,
        "used_replay_buffer": bool(replay_windows),
        "adapter_config": {"bottleneck_dim": bottleneck_dim},
        "n_trainable_parameters": n_trainable,
        "n_backbone_parameters": n_backbone,
        "pct_trainable_of_backbone": round(100.0 * n_trainable / max(1, n_backbone), 3),
        "loss_history": loss_history,
        "final_train_loss": loss_history[-1] if loss_history else None,
    }

    return CandidateArtifacts(adapter=adapter, heads=heads, metrics={"final_train_loss": loss_history[-1]}, manifest=manifest)
