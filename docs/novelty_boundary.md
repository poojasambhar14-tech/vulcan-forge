# Novelty Boundary

**This project's current central contribution is Vulcan Forge**: a
diagnosis-driven, verifier-gated lifecycle layer around a payment
foundation model — not the payment foundation model itself. The
action-conditioned world model described in earlier sections of this
project's history is retained only as experimental research
(`experimental/world_model/`) and is not part of Forge's decision path.
See `README.md` for why that demotion happened.

## What Forge claims

A closed loop of:

```
degradation -> diagnose (measured signals only, never a hidden shift label)
-> mine the specific blind spot -> build a small targeted curriculum
(real examples + simulator perturbations + boundary search), every
scenario passing an independent deterministic validator -> route to the
appropriate healing strategy (not always retrain) -> train a
parameter-efficient challenger -> independently red-team it (a search
process separate from the curriculum generator, objective is to find
regressions, not to improve the challenger) -> check against permanently
stored failure memory -> multi-criterion promotion gate -> promote/reject
-> rollback if later degradation
```

The specific claims, evidenced in `reports/forge_final_report.md`:

- A dedicated blind-spot miner finds a correctly-named, model-specific
  weakness that a generic drift-only baseline misses entirely (measured:
  0/18 triggers for the pre-Forge DriftSafe baseline vs. 3/3 correct
  identification for the injected local blind spot).
- Independent red-team certification catches a regression that passes
  every conventional validation criterion (demonstrated concretely in
  `scripts/demo_forge.py`).
- Failure memory prevents recurrence of a previously discovered failure in
  a subsequent challenger.
- A real promotion gate refuses inappropriate challengers; a naive
  retrain-and-deploy baseline with no such gate does not, and was
  measurably worse than doing nothing in at least one benchmarked case.

## What is explicitly NOT claimed

- This project does NOT claim to reproduce Razorpay's proprietary Vulcan
  architecture, training data, training scale, or any internal system —
  for either the retained Mini-Vulcan backbone or the Forge lifecycle
  layer built around it.
- This project does NOT claim Razorpay's actual production system lacks
  an equivalent lifecycle/self-healing mechanism internally. Razorpay's
  public material does not specify whether one exists; Forge is offered as
  an independently-designed exploration of what such a layer could look
  like, not a claim about what Razorpay has or hasn't built.
- Forge does NOT claim to have replaced or improved on Razorpay Vulcan's
  actual architecture in any respect. The exact tokenizer, backbone
  architecture, training objective, and inference-time feature-fusion
  mechanism Razorpay uses remain publicly unspecified beyond the AWS/
  Razorpay press disclosure summarized in `docs/research_basis.md`. Any
  more specific architectural claim about Razorpay's system (e.g. a
  particular transformer variant or masking scheme) should be treated as
  unverified until traced to a primary, checkable source — see
  `docs/research_basis.md`'s note on source verification standards.
- This project does NOT claim "fewer training examples for comparable
  recovery" versus naive retraining — the current benchmark compares Forge
  against baselines at equal budget, not smaller. See
  `reports/forge_final_report.md` section 3, "What was NOT demonstrated."
- This project does NOT claim high diagnosis accuracy uniformly — measured
  at 44% in the current benchmark, with two specific, identified failure
  modes (see the same report).
- Synthetic simulator results are not claimed to be production results.
- No benchmark numbers in this repository are hand-written; all are
  generated from persisted run manifests under `artifacts/runs/`.

## Prior claim (superseded, kept for history)

Earlier in this project's history, before Forge existed, the central
claim was an action-conditioned world model evaluated on decision-quality
metrics. That work is preserved in `experimental/world_model/` and its
negative results are preserved in `reports/failures.md`,
`reports/base_scale_result.md`, and `reports/leakage_fix_result.md`. It is
superseded by, not a component of, Forge's current contribution.
