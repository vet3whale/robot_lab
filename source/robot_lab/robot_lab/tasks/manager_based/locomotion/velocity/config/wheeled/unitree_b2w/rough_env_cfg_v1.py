# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# v1 terrain: ROUGH STABILITY specialist (replaces the old flat expert in the distillation mix).
# Slope-up and staircase-up experts already cover discrete climbing skills, so this expert
# focuses purely on staying upright and steady over CONTINUOUS uneven ground - the terrains
# where a wheeled robot keeps rolling contact instead of doing precise foot placement.
# Deliberately drops v0's gap + stepping_stones (precise-support terrains that destabilise
# wheels) in favour of bumps, rolling waves, raised tiles and scattered low obstacles.
ROUGH_TERRAINS_V1_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # canonical rough: random bump field. Centrepiece of the stability expert.
        # row 0: 2 cm bumps; row 9: 10 cm bumps.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        # smooth sinusoidal rolling hills - wheel-friendly, tests pitch/roll balance with
        # continuous contact. row 0: 5 cm crests; row 9: 20 cm crests.
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.25, amplitude_range=(0.05, 0.20), num_waves=4, border_width=0.25
        ),
        # grid of randomly raised/lowered tiles; small heights keep the wheels riding over.
        # row 0: 2 cm tiles; row 9: 10 cm tiles.
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=2.0
        ),
        # scattered low boxes; trains balance recovery when a single wheel strikes a bump.
        # row 0: 5 cm obstacles; row 9: 15 cm obstacles.
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.15,
            obstacle_height_range=(0.05, 0.15),
            obstacle_width_range=(0.4, 1.0),
            num_obstacles=8,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeB2WRoughEnvCfgV1(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # random_grid + discrete obstacles spawn many collision contacts;
        # raise the GPU collision stack from the default 2**26 (64 MB) to 2**27 (128 MB)
        self.sim.physx.gpu_collision_stack_size = 2**27

        # swap in the v1 stability terrain generator; v0's shared ROUGH_TERRAINS_CFG is untouched
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_V1_CFG

        # parent applies curriculum=True to the generator it set; re-apply for ours
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # ------------------------------Rewards (stability tuning)------------------------------
        # This expert is the opposite of the slope/staircase climbers: those LOOSEN penalties to
        # encourage climbing; here we TIGHTEN the stability terms that fight "wonky" wobble, while
        # keeping tracking at the base 3.0/1.5 (steadiness is valued over speed).

        # keep the base level over the bumps. On the old flat terrain this was trivially satisfied
        # (weight 0); on uneven ground it has to be an explicit signal. Tunable: lower it if it
        # suppresses traversal of the taller waves/tiles.
        self.rewards.flat_orientation_l2.weight = -1.0  # was 0 (off)

        # damp roll/pitch oscillation - this is the actual "wonky" the student inherited.
        self.rewards.ang_vel_xy_l2.weight = -0.1  # was -0.05

        # smoother, less jerky actuation on uneven ground.
        self.rewards.action_rate_l2.weight = -0.02  # was -0.01

        # NOTE: base_height_l2 left off on purpose - its fixed world target (0.60) is noisy over
        # bumps; flat_orientation_l2 + lin_vel_z_l2 (inherited -2.0) carry posture/bounce instead.

        # parent's disable_zero_weight_rewards() is gated on the exact base class name;
        # re-run it for this class name (after the weight edits above so newly-enabled terms survive)
        if self.__class__.__name__ == "UnitreeB2WRoughEnvCfgV1":
            self.disable_zero_weight_rewards()
