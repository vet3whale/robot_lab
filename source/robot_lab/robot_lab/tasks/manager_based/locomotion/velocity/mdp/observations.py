# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    min_range: float = 0.15,
    max_range: float = 2.0,
) -> torch.Tensor:
    """Preprocess a ray-cast depth camera into the student's channel-first depth map.

    Follows Rudin et al., "Parkour in the Wild", Sec. 2.4: pixels closer than ``min_range`` are
    treated as empty and read as *far* (``max_range``), then depth is clipped at ``max_range`` and
    scaled to ``[0, 1]``. Invalid pixels always read far, never near. The sensor's
    ``depth_clipping_behavior="max"`` already maps misses and out-of-range hits to ``max_distance``,
    so no ``inf``/``nan`` reaches here.

    The ``RayCasterCamera`` writes ``(B, H, W, 1)`` (channels last); we permute to ``(B, 1, H, W)``
    so rsl_rl's ``CNNModel`` shape-sniff routes the group to a conv encoder.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    img = sensor.data.output["distance_to_image_plane"].clone()  # (B, H, W, 1) metres
    img[img < min_range] = max_range  # too-close pixels are empty -> far
    img = img.clamp(max=max_range) / max_range  # -> [0, 1]
    return img.permute(0, 3, 1, 2)  # (B, 1, H, W) channel-first for CNNModel
