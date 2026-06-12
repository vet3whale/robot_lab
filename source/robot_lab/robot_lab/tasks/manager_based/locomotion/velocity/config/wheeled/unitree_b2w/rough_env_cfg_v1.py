# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# v1 terrain: swaps boxes + slopes for stepping stones + gaps to train gap-crossing
# and precise foot placement, keeping stairs and random rough from v0.
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
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # row 0: stones nearly touching (2 cm gap); row 9: 35 cm gap between stones
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.25,
            stone_height_max=0.1,
            stone_width_range=(0.3, 0.8),
            stone_distance_range=(0.02, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        # row 0: 5 cm gap; row 9: 50 cm gap around the platform
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.05, 0.5),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class UnitreeB2WRoughEnvCfgV1(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # stepping stones + gap produce far more collision contacts than v0's smooth surfaces;
        # raise the GPU collision stack from the default 2**26 (64 MB) to 2**27 (128 MB)
        self.sim.physx.gpu_collision_stack_size = 2**27

        # swap in the v1 terrain generator; v0's shared ROUGH_TERRAINS_CFG is untouched
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_V1_CFG

        # parent applies curriculum=True to the generator it set; re-apply for ours
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # parent's disable_zero_weight_rewards() is gated on the exact base class name;
        # re-run it for this class name
        if self.__class__.__name__ == "UnitreeB2WRoughEnvCfgV1":
            self.disable_zero_weight_rewards()
