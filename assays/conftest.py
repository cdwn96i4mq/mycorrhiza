from __future__ import annotations

import os
from typing import Iterator

import gin
import pytest

from mycorrhiza.substrate.corpus import CrossModalBatch, synthetic_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def reset_gin() -> Iterator[None]:
    gin.clear_config()
    yield
    gin.clear_config()


@pytest.fixture
def in_root() -> Iterator[str]:
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        yield ROOT
    finally:
        os.chdir(previous)


@pytest.fixture
def small_batch() -> CrossModalBatch:
    return synthetic_corpus(
        n_sc=16, n_sp=16, n_genes=24, n_types=3, neighbours=4, r_min=0.3, seed=1
    )
