import pytest
import torch

from model import build_model, freeze_until, train_mode


def small_cfg():
    return {"model": {"backbone": "resnet18", "weights": None, "in_channels": 1, "embedding_dim": 32}}


def bn_stats(module):
    return [m.running_mean.clone() for m in module.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def test_frozen_stages_keep_batchnorm_statistics():
    torch.manual_seed(0)
    model = build_model(small_cfg())
    frozen = freeze_until(model, "layer2")
    assert [m for m in frozen] == [model.backbone.conv1, model.backbone.bn1, model.backbone.relu,
                                   model.backbone.maxpool, model.backbone.layer1, model.backbone.layer2]
    before_frozen = bn_stats(model.backbone.layer2) + bn_stats(model.backbone.bn1)
    before_trained = bn_stats(model.backbone.layer3)

    train_mode(model, frozen)
    model(torch.randn(4, 1, 64, 64) * 3 + 1)

    for a, b in zip(before_frozen, bn_stats(model.backbone.layer2) + bn_stats(model.backbone.bn1)):
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(before_trained, bn_stats(model.backbone.layer3)))
    assert not any(p.requires_grad for p in model.backbone.layer2.parameters())
    assert all(p.requires_grad for p in model.backbone.layer3.parameters())


def test_freeze_until_rejects_unknown_stage():
    with pytest.raises(ValueError):
        freeze_until(build_model(small_cfg()), "layer9")


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
