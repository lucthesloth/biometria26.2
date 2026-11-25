import pytest
import torch

from model import build_model, freeze_until


def test_freeze_until_rejects_unknown_stage():
    cfg = {"model": {"backbone": "convnext_tiny", "weights": None, "in_channels": 1, "embedding_dim": 32}}
    with pytest.raises(ValueError):
        freeze_until(build_model(cfg), "features.99")


def test_convnext_keeps_pooled_layernorm_and_freezes_by_feature_block():
    cfg = {"model": {"backbone": "convnext_tiny", "weights": None, "in_channels": 1, "embedding_dim": 32}}
    model = build_model(cfg)
    head = model.backbone.classifier
    assert type(head[0]).__name__ == "LayerNorm2d" and isinstance(head[-1], torch.nn.Identity)
    frozen = freeze_until(model, "features.3")
    assert frozen == [model.backbone.features[i] for i in range(4)]
    assert not any(p.requires_grad for p in model.backbone.features[3].parameters())
    assert all(p.requires_grad for p in model.backbone.features[4].parameters())
    assert model.eval()(torch.randn(2, 1, 64, 64)).shape == (2, 32)
