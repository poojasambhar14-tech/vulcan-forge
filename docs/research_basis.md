# Research Basis

This project is a research-scale exploration inspired by publicly disclosed
payment-foundation-model systems. It is **not** a reproduction of any
proprietary system's architecture, training data, or weights.

## Public Vulcan material (Razorpay, Aug 18 2026)

Razorpay/AWS publicly disclosed that Vulcan is a proprietary transformer-based
payments foundation model trained on roughly 3 trillion data points across
4 billion payments with ~3,000 signals per transaction, used for shared
intelligence across routing, fraud/risk, RTO, and checkout personalization,
and described as continuously learning. The exact tokenizer, architecture,
pretraining objective, model size, calibration mechanism, and continual-learning
algorithm are **not** publicly specified. This project does not claim to
reproduce any of those undisclosed details.

## Transaction foundation model literature

1. **PRAGMA: Revolut Foundation Model** (arXiv:2604.08649) — heterogeneous
   irregular event sequences, masked self-supervised modelling, explicit
   time-delta encoding, hierarchical profile/event/history architecture,
   10M/100M/1B variants, LoRA adaptation of ~2-4% of parameters.
2. **TransactionGPT** (arXiv:2511.08939) — a 3D Transformer for
   multi-modal-temporal-tabular data that autoregressively generates future
   transaction trajectories. Future-transaction generation is therefore
   **prior art**, not a novel contribution of this project.
3. **nuFormer** (arXiv:2507.23267) — large-scale self-supervised transaction
   representation learning.
4. **TabFormer** (ICASSP 2021 / arXiv:2011.01843) — hierarchical tabular
   transformers, public synthetic transaction dataset, masked pretraining.
5. **LLM-based sentence embeddings for transaction understanding**
   (arXiv:2601.05271) — semantic treatment of high-cardinality merchant
   fields. This project does not place a runtime LLM in the payment
   decision critical path.

## Public industry reference systems

- Stripe Payments Foundation Model (product/engineering disclosure, not a paper)
- Mastercard Large Transaction Model (product/research disclosure)
- Plaid sequential foundation model (order, timing, cadence, CPC, Replaced
  Token Detection, Temporal Contrastive Learning)
- NVIDIA Transaction Foundation Model blueprint (~29M decoder-only model,
  temporal splits, XGBoost baseline, AP/AUPRC for imbalanced fraud)
- Adyen/NVIDIA 2026 material on RL across the acquiring lifecycle (reference
  only; RL is explicitly out of scope for this MVP)

## World model literature

- **World Action Planner** (arXiv:2607.27599) — action-conditioned world
  models with imagined rollouts for comparing actions.
- **How Should World Models Be Evaluated?** (arXiv:2606.15032) — argues
  evaluation should focus on decision quality, not merely realism of
  generated futures.
- 2026 world-model surveys highlight action conditioning as central to
  planning use cases, and compounding rollout error as a major limitation.

## Drift / continual learning literature

- **RACLA** (Expert Systems with Applications, 2026) — role-aware continual
  learning for AML, addressing concept drift and catastrophic forgetting via
  replay.
- 2026 work on drift-aware financial fraud detection — prequential
  evaluation, distribution-shift monitoring, latent-representation drift,
  delayed labels, replay / stability-plasticity tradeoffs.

See `novelty_boundary.md` for exactly what this project claims as new, and
`limitations.md` for what it explicitly does not claim.

## Source verification standard for claims about Razorpay Vulcan specifically

The Vulcan material above (proprietary transformer, ~3T data points, ~4B
payments, ~3,000 signals/payment, continuous learning) traces to the
AWS/Razorpay press disclosure — a primary, checkable source. This project
holds a higher bar for any MORE specific architectural claim about
Razorpay's actual system (e.g. a particular transformer variant, a masking
scheme, or an inference-time feature-fusion mechanism): such a claim is
not added to this document unless it traces to a primary, dated, directly
checkable source (an official disclosure, a specific dated technical post
with a stable URL, or equivalent) — not a general profile page, a
paraphrase of an unlinked post, or a secondhand summary. As of this
writing, no source meeting that bar has been found for any claim beyond
what the AWS/Razorpay press material above states; the exact tokenizer,
architecture, pretraining objective, model size, calibration mechanism,
and continual-learning algorithm remain, per that material, publicly
unspecified. This note exists because a specific, more detailed
architectural claim was proposed for addition to this document and
deliberately not added after independent verification did not find it
corroborated by any of several press sources covering the same launch.

