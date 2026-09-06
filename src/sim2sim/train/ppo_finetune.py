"""PPO subclass that records mean KL for logging (rsl_rl 5.0.1 does not return it)."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.algorithms import PPO


class PPOFinetune(PPO):
    """PPO with the last mini-batch KL stashed on ``last_kl`` and in ``loss_dict``."""

    last_kl: float = 0.0

    def update(self) -> dict[str, float]:  # type: ignore[override]
        orig = self.actor.get_kl_divergence

        def _hook(old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
            kl = orig(old_params, new_params)
            self.last_kl = float(torch.mean(kl).item())
            return kl

        self.actor.get_kl_divergence = _hook  # type: ignore[method-assign]
        try:
            loss_dict: dict[str, Any] = super().update()
        finally:
            self.actor.get_kl_divergence = orig  # type: ignore[method-assign]
        loss_dict["kl"] = self.last_kl
        return loss_dict
