# Vulcan Forge

### Adversarially Certified Self-Healing for Payment Foundation Models

**Detect → Diagnose → Repair → Attack → Remember → Certify → Install → Continue**

A payment foundation model does not fail loudly. It fails on one issuer, one
corridor, one amount band — while the dashboard average still looks fine.

Vulcan Forge is the reliability layer that notices, works out *what* broke,
builds the exact data needed to fix it, then **attacks its own repair** to find
the damage the repair caused, remembers every regression it finds forever, and
only replaces the live model once the fix has been independently certified.

> Mini-Vulcan is a research-scale analogue inspired by publicly disclosed
> payment-foundation-model ideas. It is not Razorpay's proprietary Vulcan
> implementation and uses no Razorpay data, weights, or internal systems.
> Forge is backbone-agnostic — it wraps whatever foundation model you have.

---

## The idea in one paragraph

Traditional MLOps asks: *did the model drift?* Then it retrains and ships.

That question is not sufficient, and there is literature proving it. Yan et al.,
*Positive-Congruent Training* (CVPR 2021), showed that when you update a
production model, **the negative flip rate is always positive** — there is always
a set of cases the old model handled correctly that the new one does not, even
when average accuracy improves. Average metrics are structurally blind to this.
It is why teams update models sporadically despite a steady stream of
improvements: any regression risks breaking what depends on it.

So Forge asks four questions instead of one:

**What broke? What does the model need to learn? Is retraining even the right
response? And did my fix quietly break something else?**

---

## Architecture

```
                        LIVE PAYMENT TRAFFIC
                      (routed by the champion)
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   DRIFTSAFE MONITOR    │   4 independent signals
                    │  consensus + persistence│  NORMAL → SHIFT_DETECTED
                    └───────────┬────────────┘  → PERSISTENCE_CHECK
                                │               → ADAPTATION_CANDIDATE
                    incident?  ─┴─  no → keep serving
                                │ yes
                                ▼
                      FAILURE DIAGNOSER          measured signals only
                                │                (never a hidden label)
                                ▼
                      BLIND-SPOT MINER           business-weighted slices
                                │
                                ▼
                      HEALING POLICY             retraining is not automatic
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              recalibrate   targeted    monitor
                            adapter      only
                    └───────────┼───────────┘
                                ▼
                     TARGETED CURRICULUM         + independent validator
                                │
                                ▼
                     PEFT CHALLENGER             backbone frozen
                                │                lineage-inherited adapter
                                ▼
                     STANDARD VALIDATION         recent / historical / slice
                                │
                                ▼
                  ⚔ INDEPENDENT RED TEAM         route-level negative flips
                                │
                                ▼
                  FAILURE-MEMORY REPLAY          every past regression
                                │
                                ▼
                     CERTIFICATION GATE
                    ┌───────────┴───────────┐
                 REJECT                  PROMOTE
                    │                       │
              remember &              persist → hash →
              repair again            register → swap
                    │                       │
                    │                       ▼
                    │              FRESH CHAMPION-ROUTED
                    │                    TRAFFIC
                    │                       │
                    │              guardrail breach? ──► AUTO ROLLBACK
                    │                       │
                    └───────────────────────┴──► MONITOR CONTINUES
```

---

## What makes this different

### 1. The system decides when to heal

Forge is not invoked on a schedule or by a script. Live traffic — routed by the
champion model itself — is scored window-by-window by four independent drift
signals. Healing runs only if the state machine escalates on its own through
consensus and persistence. **A transient shift returns to NORMAL and nothing
happens.** The trigger can say no, which is what makes it a trigger.

### 2. Diagnosis decides the treatment

Ten failure types, classified from measured signals alone — never a hidden shift
label. The diagnosis is consequential: a label shift routes to **recalibration**
with no weight update at all; a transient shock routes to monitoring; only a
genuine learnable blind spot gets an adapter.

### 3. Repairs are surgical and inheritable

~4,300 parameters. Roughly 1.2% of the backbone, which stays frozen — the blast
radius is bounded and every repair is cheap and reversible. Each new adapter
**warm-starts from the champion's installed adapter**, so generation N+1 builds
on N's repair rather than re-deriving it from scratch.

### 4. Every repair is attacked before it ships

An independent red team searches for **negative flips**: payments the champion
routes correctly that the challenger routes materially worse, scored as regret
against per-route ground truth. It never sees the training curriculum, and
train/certification scenarios are hash-separated.

This is the part average accuracy cannot see. A challenger can improve every
conventional metric and still carry a coherent set of routing regressions.

### 5. Failures become permanent tests — that the model can actually learn

Every discovered regression is stored with a **route invariant**: the champion's
route, the bad route, and the maximum regret any future challenger may incur.
Future models are re-tested on exactly that property.

Crucially, those failures also become a **route-preference training signal** —
a margin loss pushing the challenger toward the champion's route over the
rejected one. This uses only observed decisions, never the evaluator's oracle,
so the model can genuinely learn its way past a gate it previously failed.

### 6. Promotion is transactional, rollback is real

Persist → hash → register → **then** swap the active pointer. A failed write can
never leave an unrecorded model serving. Rollback re-hashes the checkpoint bytes
on disk, refuses corrupted artifacts, and restores authoritative runtime state —
including correctly uninstalling an adapter when reverting to a champion that
never had one.

### 7. The loop continues

After promotion the monitor resets into cooldown, the new champion serves fresh
traffic it routes itself, and an entirely independent incident can open a new
healing cycle against it. If a promoted champion's live success rate falls below
the baseline it was meant to improve, it is **automatically rolled back**.

---

## Does adversarial certification actually add value?

The question a technical reviewer will ask. Answered with a dedicated
multi-seed benchmark rather than one lucky demo seed.

`scripts/silent_regression_benchmark.py` deliberately constructs adaptations
that improve a target slice while quietly damaging routing elsewhere. Ground
truth is measured independently of both gates, on a separate probe trajectory.
**The red team is told nothing** about which arm it is examining.

| Metric | Result |
|---|---|
| Silent regressions created (risky arm) | **60%** |
| Standard gates **missed** them | **43%** |
| **Red-team recall on what standard gates missed** | **67%** |
| Red-team detection recall (all real regressions) | 57% |
| Red-team false-positive rate | 33% |
| **Regressions caught that ordinary validation would have shipped** | **2** |

The false-positive rate is on the same table as the recall, on purpose. A
certification system whose error rate you cannot see is not a certification
system.

---

## Quickstart

```bash
pip install -e ".[dev,api]"

# Train the champion
python -m train.pretrain --config configs/tiny.yaml
python -m train.train_baseline_heads --config configs/tiny.yaml

# Autonomous self-healing demo
python -m scripts.demo_forge --seed 2024

# Adversarial certification benchmark
python -m scripts.silent_regression_benchmark --seeds 1,2,3,4,5

# Control Room UI
uvicorn api.main:app --port 8000
# → http://127.0.0.1:8000/forge_control_room.html

pytest tests/ -q
```

---

## Control Room

A live operational view, not a slideshow. Every value streams from a real
backend event or a persisted benchmark manifest — nothing is hardcoded.

- **Autonomous monitor** — all four drift signals as live bars with threshold
  markers, a per-window state track, and a sparkline of consensus over time
- **Ten-stage pipeline** — colour-coded by each stage's real outcome
- **Stage evidence** — per-issuer degradation tables, curriculum composition,
  frozen-vs-adapted parameter split, adversarial transaction cards
- **Model lineage** — builds live as attempts are rejected and champions promoted
- **Certification evidence tab** — the multi-seed benchmark, read from its manifest

---

## Engineering guarantees

Each is enforced by a test, and each runs as its own named CI check.

| Guarantee | Enforced by |
|---|---|
| No temporal leakage anywhere in the pipeline | `test_no_forge_leakage.py`, `test_no_temporal_leakage.py` |
| Diagnosis never reads hidden simulator state | AST-level import check |
| Training never touches evaluator-only oracle data | `test_training_certification_alignment.py` |
| Certification tests the property the red team found | `test_training_certification_alignment.py` |
| Cooldown suppresses, and expires | `test_lifecycle_safety.py` |
| Rollback restores behaviour, refuses corrupted checkpoints | `test_runtime_rollback.py` |
| Promotion is atomic; registration precedes install | `test_lifecycle_safety.py` |
| A single anomalous window cannot trigger healing | `test_autonomous_trigger.py` |
| Monitoring is champion-aware across generations | `test_continuous_lifecycle.py` |

**77 tests.** Integrity gates run as separate CI jobs so a failure is
unmistakable rather than buried in a summary.

---

## Repository

```
vulcan/
├── forge/
│   ├── failure_diagnoser.py      10 failure types from measured signals
│   ├── blindspot_miner.py        business-weighted slice discovery
│   ├── curriculum_generator.py   real + perturbed + boundary scenarios
│   ├── scenario_validator.py     independent; train/cert separation
│   ├── healing_policy.py         non-retrain-by-default routing
│   ├── redteam_examiner.py       route-level negative-flip search
│   ├── failure_memory.py         permanent regression suite
│   ├── recalibration.py          held-out temperature scaling
│   └── forge_loop.py             orchestration
├── drift/driftsafe.py            multi-signal incident state machine
├── adaptation/                   PEFT challengers, certification gate
├── registry/model_registry.py    hash-verified promotion + rollback
└── models/, simulator/, data/

scripts/
├── demo_forge.py                     autonomous end-to-end lifecycle
├── silent_regression_benchmark.py    adversarial certification benchmark
└── forge_benchmark.py                five-arm adaptation comparison
```

---

## Positioning

Forge does not claim to invent adapters, red teaming, or drift detection. The
contribution is their composition into a specific lifecycle:

> **failure diagnosis → targeted model repair → adversarial negative-flip
> discovery → cross-generation regression memory → certification → continual
> model evolution**

Because the red team searches a finite scenario space, the honest claim is that
Forge **adversarially certifies repairs against discovered and remembered
regressions** — not that it proves the absence of all regressions. That
distinction is deliberate, and it is the claim that survives scrutiny.

---

## Vulcan Forge

### **Diagnose. Repair. Attack. Remember. Certify.**

**Forge does not just retrain the model. It earns the right to replace it.**
