from __future__ import annotations

import math
import os

import gin

from mycorrhiza.cultivation.grower import Grower
from mycorrhiza.network import ImmunoFounder
from mycorrhiza.substrate.corpus import synthetic_corpus


def test_two_step_training_runs_on_assay_config(in_root: str, tmp_path) -> None:
    gin.parse_config_file(os.path.join("conf", "_assay.gin"))
    batch = synthetic_corpus()
    model = ImmunoFounder(batch.expr.shape[1])
    grower = Grower(model, ckpt_dir=str(tmp_path))
    history = grower.fit([batch], epochs=2)
    assert len(history) == 2
    assert all(math.isfinite(value) for value in history)


def test_checkpoint_roundtrip_restores_seed(in_root: str, tmp_path) -> None:
    gin.parse_config_file(os.path.join("conf", "_assay.gin"))
    batch = synthetic_corpus()
    model = ImmunoFounder(batch.expr.shape[1])
    grower = Grower(model, ckpt_dir=str(tmp_path))
    grower.fit([batch], epochs=1)
    path = grower.save("probe.pt")
    assert os.path.exists(path)
    fresh = ImmunoFounder(batch.expr.shape[1])
    reloaded = Grower(fresh, ckpt_dir=str(tmp_path))
    reloaded.load(path)
    assert reloaded.seed == grower.seed
    assert reloaded.step == grower.step


def test_ablation_config_parses(in_root: str) -> None:
    gin.parse_config_file(os.path.join("conf", "ablation_no_crossmodal.gin"))
    batch = synthetic_corpus()
    model = ImmunoFounder(batch.expr.shape[1])
    assert model.encoder.blocks[0].attn.use_cross is False
