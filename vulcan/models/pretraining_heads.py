"""Reconstruction heads for Masked Payment Event Modeling, plus an optional
secondary next-event-type prediction head."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from vulcan.tokenization.tokenizer import FieldTokenizer


class MaskedPretrainingHeads(nn.Module):
    def __init__(self, tokenizer: FieldTokenizer, d_model: int):
        super().__init__()
        vocab_sizes = tokenizer.vocab_sizes()
        self.field_names = list(tokenizer.categorical_fields.keys())
        self.categorical_heads = nn.ModuleDict(
            {name: nn.Linear(d_model, vocab_sizes[name]) for name in self.field_names}
        )
        self.continuous_head = nn.Linear(d_model, tokenizer.n_continuous)
        # secondary objective: predict next event's error-type category
        self.next_event_type_head = nn.Linear(d_model, vocab_sizes["outcome_error_type"])

    def forward(self, z: torch.Tensor):
        """z: [N, K, d_model] contextualized representations.
        Returns dict of logits/predictions, evaluated at every position;
        caller applies masks to select which positions count for loss."""
        cat_logits = {name: head(z) for name, head in self.categorical_heads.items()}
        continuous_pred = self.continuous_head(z)
        next_event_logits = self.next_event_type_head(z)
        return cat_logits, continuous_pred, next_event_logits


def masked_categorical_loss(
    cat_logits: Dict[str, torch.Tensor],
    categorical_ids_target: Dict[str, torch.Tensor],
    cat_mask: Dict[str, torch.Tensor],
):
    """Cross-entropy over masked positions only, averaged across fields.
    Also returns per-field accuracy on masked positions for monitoring."""
    total_loss = 0.0
    n_terms = 0
    accuracies = {}
    for name, logits in cat_logits.items():
        mask = cat_mask[name]
        if mask.sum() == 0:
            continue
        target = categorical_ids_target[name][mask]
        pred_logits = logits[mask]
        loss = torch.nn.functional.cross_entropy(pred_logits, target)
        total_loss = total_loss + loss
        n_terms += 1
        with torch.no_grad():
            pred = pred_logits.argmax(dim=-1)
            accuracies[name] = float((pred == target).float().mean().item())
    if n_terms == 0:
        return torch.tensor(0.0, requires_grad=True), accuracies
    return total_loss / n_terms, accuracies


def masked_continuous_loss(continuous_pred: torch.Tensor, continuous_target: torch.Tensor, cont_mask: torch.Tensor):
    """MSE over masked continuous entries only."""
    if cont_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    pred = continuous_pred[cont_mask]
    target = continuous_target[cont_mask]
    return torch.nn.functional.mse_loss(pred, target)


def next_event_type_loss(next_event_logits: torch.Tensor, error_type_ids: torch.Tensor):
    """Predict event t+1's error-type from position t's contextual repr.
    error_type_ids: [N, K] true (unmasked) target ids."""
    N, K, _ = next_event_logits.shape
    if K < 2:
        return torch.tensor(0.0, requires_grad=True), 0.0
    logits_t = next_event_logits[:, :-1, :].reshape(-1, next_event_logits.shape[-1])
    targets_t1 = error_type_ids[:, 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(logits_t, targets_t1)
    with torch.no_grad():
        acc = float((logits_t.argmax(dim=-1) == targets_t1).float().mean().item())
    return loss, acc
