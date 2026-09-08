"""Masked Payment Event Modeling (inspired by PRAGMA / TabFormer masked
pretraining, not a reproduction of either)."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from vulcan.tokenization.tokenizer import FieldTokenizer


def apply_masking(
    categorical_ids: Dict[str, torch.Tensor],
    continuous: torch.Tensor,
    tokenizer: FieldTokenizer,
    mask_field_prob: float,
    mask_event_prob: float,
    generator: torch.Generator | None = None,
):
    """Randomly masks individual categorical fields, individual continuous
    fields, and whole events.

    Returns:
      masked_categorical_ids: dict[name] -> LongTensor [N, K] (MASK id where masked)
      masked_continuous: FloatTensor [N, K, 2*C]  (value=0 where masked, plus mask-flag channel)
      cat_mask: dict[name] -> BoolTensor [N, K]   (True where that field was masked -> compute loss there)
      cont_mask: BoolTensor [N, K, C]             (True where that continuous field was masked)
    """
    field_names = list(categorical_ids.keys())
    N, K = next(iter(categorical_ids.values())).shape
    C = continuous.shape[-1]
    kwargs = {"generator": generator} if generator is not None else {}

    # whole-event mask: [N, K]
    event_mask = torch.rand(N, K, **kwargs) < mask_event_prob

    masked_categorical_ids = {}
    cat_mask = {}
    for name in field_names:
        field_mask = torch.rand(N, K, **kwargs) < mask_field_prob
        full_mask = field_mask | event_mask
        ids = categorical_ids[name].clone()
        mask_id = tokenizer.categorical_fields[name].mask_id
        ids[full_mask] = mask_id
        masked_categorical_ids[name] = ids
        cat_mask[name] = full_mask

    cont_field_mask = torch.rand(N, K, C, **kwargs) < mask_field_prob
    cont_full_mask = cont_field_mask | event_mask.unsqueeze(-1).expand(-1, -1, C)
    values = continuous.clone()
    values[cont_full_mask] = 0.0
    mask_flags = cont_full_mask.float()
    masked_continuous = torch.cat([values, mask_flags], dim=-1)

    return masked_categorical_ids, masked_continuous, cat_mask, cont_full_mask
