from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from mycorrhiza.field.io import atomic_save, load_checkpoint


@dataclass
class TrainState:
    step: int
    epoch: int
    seed: int
    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "seed": self.seed,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TrainState":
        return cls(
            step=int(payload["step"]),
            epoch=int(payload["epoch"]),
            seed=int(payload["seed"]),
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload["scheduler"],
        )


def save_state(state: TrainState, path: str) -> None:
    atomic_save(state.to_payload(), path)


def restore_state(path: str) -> TrainState:
    return TrainState.from_payload(load_checkpoint(path))
