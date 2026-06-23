from __future__ import annotations

import os
from typing import Iterable, List

import gin
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from mycorrhiza.cultivation.schedule import build_optimizer, build_scheduler
from mycorrhiza.cultivation.state import TrainState, restore_state, save_state
from mycorrhiza.field.loupe import get_logger
from mycorrhiza.field.seed import set_seed
from mycorrhiza.network import ImmunoFounder
from mycorrhiza.substrate.corpus import CrossModalBatch
from mycorrhiza.tropism.composite import LossTerms, composite_loss

_LOG = get_logger("mycorrhiza.cultivation")


@gin.configurable
class Grower:
    def __init__(
        self,
        model: ImmunoFounder,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        grad_accum: int = 8,
        amp: bool = False,
        seed: int = 0,
        device: str = "cpu",
        ckpt_dir: str = "runs",
    ) -> None:
        set_seed(seed)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
        self.scheduler = build_scheduler(
            self.optimizer, warmup_steps=warmup_steps, max_steps=max_steps
        )
        self.scaler = GradScaler(enabled=amp)
        self.grad_accum = grad_accum
        self.amp = amp
        self.seed = seed
        self.ckpt_dir = ckpt_dir
        self.step = 0
        self.epoch = 0

    def run_step(self, batch: CrossModalBatch, accumulate_index: int) -> LossTerms:
        self.model.train()
        resident = batch.to(self.device)
        with autocast(enabled=self.amp):
            output = self.model(resident)
            terms = composite_loss(output, resident)
        (self.scaler.scale(terms.total / self.grad_accum)).backward()
        if (accumulate_index + 1) % self.grad_accum == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.step += 1
        return terms

    def fit(self, batches: Iterable[CrossModalBatch], epochs: int) -> List[float]:
        history: List[float] = []
        for _ in range(epochs):
            for index, batch in enumerate(batches):
                terms = self.run_step(batch, index)
                history.append(float(terms.total.item()))
            self.epoch += 1
        return history

    def save(self, name: str = "latest.pt") -> str:
        path = os.path.join(self.ckpt_dir, name)
        state = TrainState(
            step=self.step,
            epoch=self.epoch,
            seed=self.seed,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
        )
        save_state(state, path)
        _LOG.info("checkpoint written to %s at step %d", path, self.step)
        return path

    def load(self, path: str) -> None:
        state = restore_state(path)
        set_seed(state.seed)
        self.model.load_state_dict(state.model)
        self.optimizer.load_state_dict(state.optimizer)
        self.scheduler.load_state_dict(state.scheduler)
        self.step = state.step
        self.epoch = state.epoch
        self.seed = state.seed
