import torch
import pytest

from vulcan.common.config import load_config
from vulcan.data.generate import generate_training_data
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


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("configs/tiny.yaml")
    records = generate_training_data(cfg, seed=42, n_transactions=600)
    train, val, test = chronological_split(records, 0.7, 0.15, 0.15)
    tokenizer = FieldTokenizer(cfg)
    tokenizer.fit(train)
    windows = build_windows(train, history_len=cfg["model"]["history_len"], stride=8)
    cat_ids, cont = encode_windows(tokenizer, windows)
    model_cfg = MiniVulcanConfig.from_dict(cfg["model"])
    backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    heads = MaskedPretrainingHeads(tokenizer, model_cfg.d_model)
    return cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg


def test_output_shapes(setup):
    cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg = setup
    masked_cat, masked_cont, cat_mask, cont_mask = apply_masking(cat_ids, cont, tokenizer, 0.15, 0.1)
    z = backbone(masked_cat, masked_cont)
    N, K = cont.shape[0], cont.shape[1]
    assert z.shape == (N, K, model_cfg.d_model)

    cat_logits, cont_pred, next_event_logits = heads(z)
    assert cont_pred.shape == (N, K, tokenizer.n_continuous)
    for name, logits in cat_logits.items():
        assert logits.shape[:2] == (N, K)


def test_no_nans_in_forward_pass(setup):
    cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg = setup
    masked_cat, masked_cont, cat_mask, cont_mask = apply_masking(cat_ids, cont, tokenizer, 0.15, 0.1)
    z = backbone(masked_cat, masked_cont)
    assert not torch.isnan(z).any()
    cat_logits, cont_pred, next_event_logits = heads(z)
    assert not torch.isnan(cont_pred).any()
    for logits in cat_logits.values():
        assert not torch.isnan(logits).any()


def test_gradients_flow(setup):
    cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg = setup
    masked_cat, masked_cont, cat_mask, cont_mask = apply_masking(cat_ids, cont, tokenizer, 0.15, 0.1)
    z = backbone(masked_cat, masked_cont)
    cat_logits, cont_pred, next_event_logits = heads(z)

    loss_cat, _ = masked_categorical_loss(cat_logits, cat_ids, cat_mask)
    loss_cont = masked_continuous_loss(cont_pred, cont, cont_mask)
    loss_next, _ = next_event_type_loss(next_event_logits, cat_ids["outcome_error_type"])
    loss = loss_cat + loss_cont + 0.2 * loss_next

    loss.backward()
    total_grad_norm = sum(
        p.grad.abs().sum().item() for p in backbone.parameters() if p.grad is not None
    )
    assert total_grad_norm > 0


def test_masked_loss_is_only_computed_on_masked_positions(setup):
    cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg = setup
    # mask nothing -> masked_categorical_loss should return 0 loss / empty accs
    zero_mask = {name: torch.zeros_like(ids, dtype=torch.bool) for name, ids in cat_ids.items()}
    cat_logits = {name: torch.zeros(ids.shape[0], ids.shape[1], 5) for name, ids in cat_ids.items()}
    loss, accs = masked_categorical_loss(cat_logits, cat_ids, zero_mask)
    assert float(loss) == 0.0
    assert accs == {}


def test_checkpoint_save_and_load(setup, tmp_path):
    cfg, tokenizer, cat_ids, cont, backbone, heads, model_cfg = setup
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"backbone_state_dict": backbone.state_dict()}, ckpt_path)

    new_backbone = MiniVulcanBackbone(tokenizer, model_cfg)
    state = torch.load(ckpt_path, weights_only=False)
    new_backbone.load_state_dict(state["backbone_state_dict"])

    backbone.eval()
    new_backbone.eval()
    masked_cat, masked_cont, _, _ = apply_masking(cat_ids, cont, tokenizer, 0.0, 0.0)
    with torch.no_grad():
        z1 = backbone(masked_cat, masked_cont)
        z2 = new_backbone(masked_cat, masked_cont)
    assert torch.allclose(z1, z2, atol=1e-6)
