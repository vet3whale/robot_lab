# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
No stock rsl_rl 5.0.1 model combines CNN + RNN, has the per-image FC stage, or the Figure-3 head
wiring, so this small model does it by subclassing ``CNNModel`` and reusing ``rsl_rl.modules.RNN`` and
``rsl_rl.modules.MLP``. 
Per depth image: the inherited CNN encoder (3 conv + max-pool) then 2 FC layers to a 64-dim latent. 
The per-image latents are concatenated with proprioception and fed to a 2-layer LSTM. 
The head MLP reads ``[LSTM out, proprio, commands]`` so proprio re-enters at the head
and the velocity commands bypass the LSTM entirely (they only appear at the head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models import CNNModel
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import unpad_trajectories


class CNNRNNModel(CNNModel):
    """CNN encoders + 2-layer LSTM student with the paper's command-bypass head wiring."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        cnn_fc_dims: tuple[int, ...] | list[int] = (128, 64),
        command_obs_groups: tuple[str, ...] | list[str] = ("student_commands",),
    ) -> None:
        """Build the CNN + LSTM student.

        Args:
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
            distribution_cfg, cnn_cfg, cnns: forwarded to ``CNNModel`` (``hidden_dims`` sizes the head).
            rnn_type: "lstm" or "gru".
            rnn_hidden_dim: LSTM hidden size (paper leaves this unspecified; the repo keeps 256).
            rnn_num_layers: number of LSTM layers (paper Fig. 3 uses 2).
            cnn_fc_dims: per-image FC stack after the conv encoder; the last value is the latent size
                (paper: 2 FC layers to 64).
            command_obs_groups: 1D groups that bypass the LSTM and join only at the head.
        """
        if obs_normalization:
            # The head splits the 1D observations into proprio and commands, but EmpiricalNormalization
            # is a single monolithic module sized to their sum, so it cannot be applied per-part.
            # These obs already carry per-term scales, so normalization is unused here.
            raise NotImplementedError("CNNRNNModel does not support obs_normalization; scale via the obs terms.")

        self.rnn_hidden_dim = rnn_hidden_dim
        self.cnn_fc_dims = tuple(cnn_fc_dims)
        self.command_obs_groups = list(command_obs_groups)

        # CNNModel.__init__ calls _get_obs_dim (our override sets proprio_dim / cmd_dim), builds the
        # per-camera conv encoders, then MLPModel.__init__ sizes the head from our _get_latent_dim.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            cnn_cfg,
            cnns,
        )

        # Per-image FC stack: conv features -> cnn_fc_dims[-1] latent (paper Fig. 3: 2 FC to 64).
        latent_dim = self.cnn_fc_dims[-1]
        self.cnn_fcs = nn.ModuleDict(
            {
                group: MLP(int(self.cnns[group].output_dim), latent_dim, list(self.cnn_fc_dims[:-1]), activation)
                for group in self.obs_groups_2d
            }
        )

        # LSTM input: proprio + per-image latents. Commands are NOT in the LSTM input (Fig. 3).
        rnn_input_dim = self.proprio_dim + latent_dim * len(self.obs_groups_2d)
        self.rnn = RNN(rnn_input_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode depth + proprio through the LSTM, then rejoin proprio and commands at the head."""
        # Per-image CNN -> FC -> latent, concatenated over cameras.
        z_img = torch.cat(
            [self.cnn_fcs[group](self.cnns[group](obs[group])) for group in self.obs_groups_2d], dim=-1
        )
        proprio = torch.cat([obs[group] for group in self.proprio_obs_groups], dim=-1)
        commands = torch.cat([obs[group] for group in self.command_obs_groups_active], dim=-1)

        # LSTM over [proprio, depth latents]; commands bypass it.
        z_rnn = self.rnn(torch.cat([proprio, z_img], dim=-1), masks, hidden_state).squeeze(0)

        # In batch/update mode the RNN returns unpadded outputs; align proprio and commands to it so
        # the head concat stays row-matched. (Distillation runs masks=None, so this is inert there.)
        if masks is not None:
            proprio = unpad_trajectories(proprio, masks)
            commands = unpad_trajectories(commands, masks)

        # Head input (Fig. 3): [LSTM out, proprio, commands].
        return torch.cat([z_rnn, proprio, commands], dim=-1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str):
        """Split the 1D groups into proprio (into the LSTM + head) and commands (head only)."""
        obs_groups_1d, obs_dim_1d = super()._get_obs_dim(obs, obs_groups, obs_set)
        self.proprio_obs_groups = [g for g in obs_groups_1d if g not in self.command_obs_groups]
        self.command_obs_groups_active = [g for g in obs_groups_1d if g in self.command_obs_groups]
        self.proprio_dim = sum(int(obs[g].shape[-1]) for g in self.proprio_obs_groups)
        self.cmd_dim = sum(int(obs[g].shape[-1]) for g in self.command_obs_groups_active)
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Head input = LSTM output + proprio (re-entered) + commands (bypass)."""
        return self.rnn_hidden_dim + self.proprio_dim + self.cmd_dim

    # --- recurrent state plumbing (delegate to self.rnn, like RNNModel) ---

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the LSTM hidden state (all envs, or only the done ones)."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the LSTM hidden state."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the LSTM hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Deploy export is a separate CNN+LSTM wrapper; not implemented yet."""
        raise NotImplementedError(
            "CNNRNNModel needs a dedicated CNN+LSTM export wrapper; "
            "the stock CNN/RNN exporters each handle only one modality."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Deploy export is a separate CNN+LSTM wrapper; not implemented yet."""
        raise NotImplementedError(
            "CNNRNNModel needs a dedicated CNN+LSTM ONNX wrapper; "
            "the stock CNN/RNN exporters each handle only one modality."
        )