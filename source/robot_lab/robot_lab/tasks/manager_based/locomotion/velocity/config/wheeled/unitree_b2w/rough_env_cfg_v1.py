# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# v1 terrain: mixed climbing/stability terrain.
# Identical to the base rough config in every respect (rewards, observations, actions, etc.);
# the ONLY difference is the terrain generator: 60% stairs, 20% slopes, 20% randomized obstacles.
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
        # 60% stairs - split 50/50 between solid-mesh (sharp edges) and heightfield (softer)
        "mesh_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.05),
            step_width=0.275,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.05),
            step_width=0.275,
            platform_width=2.0,
            border_width=0.25,
        ),
        # 20% slopes - inverted pyramid slope, robot climbs outward up a continuous incline
        "hf_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.176, 0.176),
            platform_width=2.0,
            border_width=0.25,
        ),
        # 20% randomized obstacles - scattered low boxes of random height/width
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
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

        # stair meshes + scattered obstacles spawn many collision contacts;
        # raise the GPU collision stack from the default 2**26 (64 MB) to 2**27 (128 MB)
        self.sim.physx.gpu_collision_stack_size = 2**27

        # swap in the v1 terrain generator; the base ROUGH_TERRAINS_CFG is untouched
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_V1_CFG

        # parent applies curriculum=True to the generator it set; re-apply for ours
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # parent's disable_zero_weight_rewards() is gated on the exact base class name;
        # re-run it for this subclass so zero-weight terms (e.g. feet_air_time_variance,
        # whose body_names is empty) are stripped before the reward manager resolves them
        if self.__class__.__name__ == "UnitreeB2WRoughEnvCfgV1":
            self.disable_zero_weight_rewards()
