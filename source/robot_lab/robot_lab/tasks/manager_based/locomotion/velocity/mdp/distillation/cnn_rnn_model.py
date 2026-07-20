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

import copy

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
        # Per-image CNN -> FC -> latent, concatenated over cameras. In the recurrent PPO update the
        # image obs arrive as [T, N, C, H, W]; conv2d needs 4D, so fold the leading (time, env) dims
        # for the CNN and restore them on the latent. Single-step obs (rollout / distillation) are 4D
        # and skip the reshape.
        # Different loaders produce different input dimensions (4D vs 5D) -> both feed the one CNN
        # function -> that function was written to accept only 4D -> PPO's 5D input crashes it.
        img_latents = []
        for group in self.obs_groups_2d:
            image = obs[group]
            if image.dim() == 5:
                lead = image.shape[:2]
                flat = image.reshape(-1, *image.shape[2:])
                latent = self.cnn_fcs[group](self.cnns[group](flat)).reshape(*lead, -1)
            else:
                latent = self.cnn_fcs[group](self.cnns[group](image))
            img_latents.append(latent)
        z_img = torch.cat(img_latents, dim=-1)
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
        """Return a version of the model compatible with Torch JIT export."""
        if not isinstance(self.rnn.rnn, nn.LSTM):
            raise NotImplementedError(f"Unsupported RNN type for export: {type(self.rnn.rnn)}")
        return _TorchCNNRNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        if not isinstance(self.rnn.rnn, nn.LSTM):
            raise NotImplementedError(f"Unsupported RNN type for export: {type(self.rnn.rnn)}")
        return _OnnxCNNRNNModel(self, verbose)


def _copy_encoders(model: CNNRNNModel) -> nn.ModuleList:
    """Fuse each camera's conv encoder and FC stage into one module, ordered by ``obs_groups_2d``.

    The fusion keeps the JIT forward loop to a single ``ModuleList`` iteration: TorchScript can
    iterate a ``ModuleList`` but cannot index a second one with the loop variable. The ordering is
    the positional contract with the deploy side, as in the stock ``_TorchCNNModel``.
    """
    return nn.ModuleList(
        nn.Sequential(copy.deepcopy(model.cnns[group]), copy.deepcopy(model.cnn_fcs[group]))
        for group in model.obs_groups_2d
    )


class _TorchCNNRNNModel(nn.Module):
    """Exportable CNN+LSTM student for JIT. Hidden state lives in buffers; call ``reset()`` on episode start."""

    def __init__(self, model: CNNRNNModel) -> None:
        """Create a TorchScript-friendly copy of a CNNRNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = _copy_encoders(model)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, proprio: torch.Tensor, commands: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run one inference step; ``obs_2d`` order must match ``obs_groups_2d``."""
        proprio = self.obs_normalizer(proprio)
        latent_list = []
        for i, encoder in enumerate(self.encoders):
            latent_list.append(encoder(obs_2d[i]))
        z_img = torch.cat(latent_list, dim=-1)
        x, (h, c) = self.rnn(
            torch.cat([proprio, z_img], dim=-1).unsqueeze(0), (self.hidden_state, self.cell_state)
        )
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        latent = torch.cat([x.squeeze(0), proprio, commands], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxCNNRNNModel(nn.Module):
    """Exportable CNN+LSTM student for ONNX. The caller owns the recurrent state and threads it through."""

    is_recurrent: bool = True

    def __init__(self, model: CNNRNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNRNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = _copy_encoders(model)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.proprio_dim = model.proprio_dim
        self.cmd_dim = model.cmd_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, proprio: torch.Tensor, commands: torch.Tensor, *rest: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deterministic inference for ONNX export; ``rest`` is ``(*depth_images, h_in, c_in)``."""
        obs_2d = rest[: len(self.encoders)]
        h_in, c_in = rest[len(self.encoders)], rest[len(self.encoders) + 1]

        proprio = self.obs_normalizer(proprio)
        z_img = torch.cat([encoder(obs_2d[i]) for i, encoder in enumerate(self.encoders)], dim=-1)
        x, (h, c) = self.rnn(torch.cat([proprio, z_img], dim=-1).unsqueeze(0), (h_in, c_in))
        latent = torch.cat([x.squeeze(0), proprio, commands], dim=-1)
        out = self.deterministic_output(self.mlp(latent))
        return out, h, c

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        proprio = torch.zeros(1, self.proprio_dim)
        commands = torch.zeros(1, self.cmd_dim)
        dummy_2d = [
            torch.zeros(1, self.obs_channels_2d[i], *self.obs_dims_2d[i]) for i in range(len(self.obs_groups_2d))
        ]
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        return (proprio, commands, *dummy_2d, h_in, c_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names; the image names identify the cameras."""
        return ["proprio", "commands", *self.obs_groups_2d, "h_in", "c_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions", "h_out", "c_out"]