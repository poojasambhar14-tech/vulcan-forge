"""
Training/certification alignment.

THE BUG THIS LOCKS OUT: failure-memory scenarios were replayed into the
bandit losses, which train "did the taken route succeed?" -- a BINARY
OUTCOME signal. But certification tests a ROUTING decision: does the
challenger still route this payment materially worse than the certified
champion did? Different properties. Measured consequence (seed 2024,
round 3, before the fix): a challenger trained on all 8 remembered
failures still failed 8/8 of them at the gate -- pass rate 0.000. The
model was structurally unable to learn its way past the criterion it was
being judged on, so the self-healing loop could never converge.

THE FIX: route-preference anchors. Each remembered failure contributes a
(window, champion_route, bad_route) triple, and a pairwise margin loss
pushes the challenger's score for the champion's route above its score
for the route the rejected challenger chose.

WHY NOT TRAIN ON THE ORACLE: the stored route invariant also contains the
simulator's true per-route success probabilities. Training on those would
be privileged information unavailable at deployment -- a new leak of
exactly the kind this project already had to fix once. The anchors use
only OBSERVED decisions (what the champion chose, what the challenger
chose), which is legitimate self-distillation.

After the fix, the same seed goes 0.000 -> 1.000 memory pass rate and the
loop converges to a promoted champion.
"""
from __future__ import annotations

import inspect

import torch

from vulcan.common.config import load_config
from vulcan.common.seeding import set_global_seed
from vulcan.data.generate import generate_training_data
from vulcan.data.windowing import build_windows
from vulcan.tokenization.tokenizer import FieldTokenizer
from vulcan.models.mini_vulcan import MiniVulcanBackbone, MiniVulcanConfig
from vulcan.models.downstream_heads import DownstreamHeads
from vulcan.adaptation.candidate_trainer import train_candidate


def test_route_anchors_are_not_trained_on_oracle_probabilities():
    """The anchor path must consume only observed route CHOICES, never the
    oracle success probabilities stored alongside them."""
    src = inspect.getsource(train_candidate)
    for forbidden in ("oracle_route_p_success", "_oracle_p_success", "oracle_best_route"):
        assert forbidden not in src, (
            f"train_candidate references {forbidden} -- training on evaluator-only "
            f"oracle data would be a privileged-information leak"
        )


def test_route_anchor_loss_moves_preference_toward_champion_route():
    """The core behavioural claim: given an anchor saying 'champion chose
    route G, rejected challenger chose route B', training must increase the
    challenger's preference for G over B on that scenario. If this fails,
    remembered failures cannot be learned and the loop cannot converge."""
    cfg = load_config("configs/tiny.yaml")
    set_global_seed(cfg["seed"])
    records = generate_training_data(cfg, seed=cfg["seed"], n_transactions=1200)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(records)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])

    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    heads = DownstreamHeads(model_cfg.d_model, cfg["simulator"]["num_routes"])
    heads.eval()

    history_len = cfg["model"]["history_len"]
    windows = build_windows(records, history_len, stride=4)
    train_w = windows[:60]
    anchor_w = windows[60:70]

    good_route, bad_route = 0, 1
    anchors = [(w, good_route, bad_route) for w in anchor_w]

    def preference_gap(hs, ad):
        """score(good) - score(bad) on the anchor scenarios."""
        from vulcan.adaptation.candidate_trainer import _prepare_batch
        z, *_ = _prepare_batch(backbone, tokenizer, anchor_w)
        with torch.no_grad():
            out = hs(ad(z)) if ad is not None else hs(z)
            lg = out["success_logit"]
            return float((lg[:, good_route] - lg[:, bad_route]).mean())

    without = train_candidate(
        backbone, tokenizer, heads.state_dict(), drift_windows=train_w,
        replay_windows=None, num_routes=cfg["simulator"]["num_routes"],
        d_model=model_cfg.d_model, epochs=30, seed=5,
    )
    with_anchors = train_candidate(
        backbone, tokenizer, heads.state_dict(), drift_windows=train_w,
        replay_windows=None, num_routes=cfg["simulator"]["num_routes"],
        d_model=model_cfg.d_model, epochs=30, seed=5,
        route_anchors=anchors,
    )

    gap_without = preference_gap(without.heads, without.adapter)
    gap_with = preference_gap(with_anchors.heads, with_anchors.adapter)

    assert gap_with > gap_without, (
        f"Route anchors did not increase preference for the champion's route "
        f"(with={gap_with:.4f} vs without={gap_without:.4f}). Remembered failures "
        f"would be unlearnable and the self-healing loop could not converge."
    )
