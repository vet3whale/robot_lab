# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


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


# Structural constants for the depth domain randomization (Rudin et al. "Parkour in the Wild",
# Sec. 2.4). These shape the stateful buffers and are rarely tuned, so they stay module-level rather
# than per-term params; the per-call knobs (thresholds, probabilities, enables) are __call__ args.
_DR_PERLIN_CELLS_H = 6  # Perlin lattice cells along image height (hole patch scale ~ H/6)
_DR_PERLIN_CELLS_W = 8  # Perlin lattice cells along image width
_DR_PERLIN_ROT_STD = 0.05  # per-step random-walk std (rad) of the lattice gradient angles ("slowly evolving")
_DR_BLIND_MIN = 1  # leftmost blind columns, min (per-episode, per-camera)
_DR_BLIND_MAX = 5  # leftmost blind columns, max


def _perlin_fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin (2002) quintic fade 6t^5 - 15t^4 + 10t^3 (zero first/second derivative at 0 and 1)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class DepthImageDR(ManagerTermBase):
    """Preprocess a ray-cast depth camera into the student's channel-first depth map and apply
    the paper's sim-side depth domain randomization (Rudin et al., "Parkour in the Wild", Sec. 2.4).

    Base preprocessing: pixels closer than ``min_range`` read as far (``max_range``), depth is
    clipped at ``max_range`` and scaled to ``[0, 1]``, and the image is permuted from the sensor's
    ``(B, H, W, 1)`` to ``(B, 1, H, W)`` so rsl_rl's ``CNNModel`` routes it to a conv encoder.
    All corruptions run in this space, where empty == far == 1.0.

    Domain randomization, in the paper's order:

    * **Edge noise** (per step): pixels around depth-gradient edges (mask dilated one pixel) are
      emptied or replaced by a random neighbor.
    * **Holes** (temporally consistent): a per-env Perlin (2002) gradient-noise field whose lattice
      gradients random-walk slowly, thresholded into drifting patches of maximum depth.
    * **Blind spot** (per episode): the leftmost ``k`` columns are emptied.
    * **Gaussian blur** (per step): last, so it smooths the injected artifacts.

    Stateful class term: holes and the blind spot need memory across steps, so the obs manager
    calls :meth:`reset` for terminated envs. Disabling every ``enable_*`` flag leaves the clean
    clip+scale+permute baseline (the ablation switch).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._device = env.device
        self.blind_min = int(cfg.params.get("blind_min", _DR_BLIND_MIN))
        self.blind_max = int(cfg.params.get("blind_max", _DR_BLIND_MAX))
        # fixed Sobel kernels for the edge-gradient detector, shaped (out=1, in=1, 3, 3)
        self._sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=self._device
        ).view(1, 1, 3, 3)
        self._sobel_y = self._sobel_x.transpose(2, 3).contiguous()
        # per-env state, allocated lazily on the first call once the image shape is known
        self.angles: torch.Tensor | None = None  # Perlin lattice gradient angles, (B, GH+1, GW+1)
        self.blind_k: torch.Tensor | None = None

    def _lazy_init(self, batch: int, h: int, w: int) -> None:
        gh, gw = _DR_PERLIN_CELLS_H, _DR_PERLIN_CELLS_W
        self.angles = torch.rand(batch, gh + 1, gw + 1, device=self._device) * (2.0 * math.pi)
        self.blind_k = torch.randint(self.blind_min, self.blind_max + 1, (batch,), device=self._device)
        # fixed pixel -> lattice geometry: cell index, in-cell offset, and faded weights per pixel.
        # Pixel centres are mapped onto a gh x gw cell grid; corners index the (gh+1)*(gw+1) lattice.
        py = (torch.arange(h, device=self._device) + 0.5) * (gh / h)
        px = (torch.arange(w, device=self._device) + 0.5) * (gw / w)
        y0 = py.floor().long().clamp_(max=gh - 1)
        x0 = px.floor().long().clamp_(max=gw - 1)
        self._v = (py - y0).view(h, 1).expand(h, w).reshape(-1)  # in-cell offset, (H*W,)
        self._u = (px - x0).view(1, w).expand(h, w).reshape(-1)
        self._fv = _perlin_fade(self._v)
        self._fu = _perlin_fade(self._u)
        yy0 = y0.view(h, 1).expand(h, w).reshape(-1)
        xx0 = x0.view(1, w).expand(h, w).reshape(-1)
        stride = gw + 1  # lattice row stride for flat indexing
        self._idx00 = yy0 * stride + xx0
        self._idx10 = self._idx00 + 1
        self._idx01 = self._idx00 + stride
        self._idx11 = self._idx00 + stride + 1
        self._hw = (h, w)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # fired by the obs manager for envs that just terminated; re-roll per-episode state so a hole
        # or blind column never leaks across an episode boundary
        if self.angles is None:  # not built yet (manager calls reset() once at prepare time)
            return
        if env_ids is None:
            env_ids = torch.arange(self.angles.shape[0], device=self._device)
        n = len(env_ids)
        gh, gw = _DR_PERLIN_CELLS_H, _DR_PERLIN_CELLS_W
        self.angles[env_ids] = torch.rand(n, gh + 1, gw + 1, device=self._device) * (2.0 * math.pi)
        self.blind_k[env_ids] = torch.randint(self.blind_min, self.blind_max + 1, (n,), device=self._device)

    def _edge_noise(self, img: torch.Tensor, thresh: float, p_empty: float, p_shuffle: float) -> torch.Tensor:
        # replicate-pad before Sobel: zero-padding would fake a depth cliff along the image border
        padded = F.pad(img, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(padded, self._sobel_x)
        gy = F.conv2d(padded, self._sobel_y)
        edge = (gx * gx + gy * gy).sqrt() > thresh
        # "pixels around the edges": dilate the mask one pixel so the band straddles the discontinuity
        edge = F.max_pool2d(edge.float(), kernel_size=3, stride=1, padding=1) > 0.5
        u = torch.rand_like(img)
        empty = edge & (u < p_empty)
        shuffle = edge & (u >= p_empty) & (u < p_empty + p_shuffle)
        # a random 4-neighbor (up/down/left/right) per pixel, for the shuffle case
        stack = torch.stack(
            [
                torch.roll(img, 1, dims=2),
                torch.roll(img, -1, dims=2),
                torch.roll(img, 1, dims=3),
                torch.roll(img, -1, dims=3),
            ],
            dim=0,
        )  # (4, B, 1, H, W)
        pick = torch.randint(0, 4, img.shape, device=img.device).unsqueeze(0)  # (1, B, 1, H, W)
        neighbor = stack.gather(0, pick).squeeze(0)
        img = torch.where(empty, torch.ones_like(img), img)
        img = torch.where(shuffle, neighbor, img)
        return img

    def _holes(self, img: torch.Tensor, thresh: float) -> torch.Tensor:
        # slowly evolve the Perlin field: random-walk the lattice gradient angles. Rotating a unit
        # gradient keeps the noise amplitude stationary, so hole coverage stays constant over time.
        self.angles = self.angles + _DR_PERLIN_ROT_STD * torch.randn_like(self.angles)
        g = torch.stack((torch.cos(self.angles), torch.sin(self.angles)), dim=-1)
        g = g.view(g.shape[0], -1, 2)  # (B, (GH+1)*(GW+1), 2) unit gradients at lattice nodes
        # Perlin (2002): dot each corner gradient with the offset to the pixel, then interpolate
        # the four dots with the quintic-faded in-cell coordinates
        n00 = g[:, self._idx00, 0] * self._u + g[:, self._idx00, 1] * self._v
        n10 = g[:, self._idx10, 0] * (self._u - 1.0) + g[:, self._idx10, 1] * self._v
        n01 = g[:, self._idx01, 0] * self._u + g[:, self._idx01, 1] * (self._v - 1.0)
        n11 = g[:, self._idx11, 0] * (self._u - 1.0) + g[:, self._idx11, 1] * (self._v - 1.0)
        nx0 = n00 + self._fu * (n10 - n00)
        nx1 = n01 + self._fu * (n11 - n01)
        noise = (nx0 + self._fv * (nx1 - nx0)).view(img.shape[0], 1, *self._hw)
        return torch.where(noise > thresh, torch.ones_like(img), img)

    def _blind_spot(self, img: torch.Tensor) -> torch.Tensor:
        cols = torch.arange(img.shape[-1], device=img.device).view(1, 1, 1, -1)
        mask = cols < self.blind_k.view(-1, 1, 1, 1)  # (B, 1, 1, W) -> leftmost k columns
        return torch.where(mask.expand_as(img), torch.ones_like(img), img)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        min_range: float = 0.15,
        max_range: float = 2.0,
        enable_edge: bool = True,
        enable_holes: bool = True,
        enable_blind: bool = True,
        enable_blur: bool = True,
        edge_thresh: float = 0.1,
        edge_p_empty: float = 0.5,
        edge_p_shuffle: float = 0.5,
        hole_thresh: float = 0.25,
        blur_kernel: int = 3,
        blur_sigma: float = 0.8,
        blind_min: int = _DR_BLIND_MIN,
        blind_max: int = _DR_BLIND_MAX,
    ) -> torch.Tensor:
        img = env.scene.sensors[sensor_cfg.name].data.output["distance_to_image_plane"].clone()
        img[img < min_range] = max_range  # too-close pixels are empty -> far
        img = (img.clamp(max=max_range) / max_range).permute(0, 3, 1, 2)  # (B, 1, H, W) in [0, 1]
        # (re)allocate per-env state if this is the first call or the batch size changed
        if self.angles is None or self.angles.shape[0] != img.shape[0]:
            self._lazy_init(img.shape[0], img.shape[2], img.shape[3])
        if enable_edge:
            img = self._edge_noise(img, edge_thresh, edge_p_empty, edge_p_shuffle)
        if enable_holes:
            img = self._holes(img, hole_thresh)
        if enable_blind:
            img = self._blind_spot(img)
        if enable_blur:
            img = gaussian_blur(img, kernel_size=[blur_kernel, blur_kernel], sigma=[blur_sigma, blur_sigma])
        return img.clamp(0.0, 1.0)
