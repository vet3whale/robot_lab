# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# slope specialist - ascending only.
# 100% inverted pyramid slope (pit at centre, smooth incline climbs outward) so the
# robot is always forced to climb UP a continuous slope.  Only height-field slope
# variants exist in isaaclab (there is no mesh slope), so this is a single sub-terrain.
# Difficulty (slope_range) is widened beyond the rough generalist's 0.0-0.4 rad up to
# 0.60 rad (~34.4 deg), making row 9 a steeper climb than anything in the base curriculum.
SLOPEUP_TEACHER_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # inverted pyramid slope: platform low at centre, slopes rise outward -> robot climbs up
        "hf_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.0, 0.50),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeB2WSlopeUpTeacherEnvCfg(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # swap in the slopeup terrain generator
        self.scene.terrain.terrain_generator = SLOPEUP_TEACHER_CFG

        # train with curriculum
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # uprightness reward was so high the robot just stood still; cut it so tracking drives climbing
        self.rewards.upward.weight = 0.5  # was 3.0

        # start everyone on the easiest row so it learns to walk before the curriculum ramps up
        self.scene.terrain.max_init_terrain_level = 0

        # pay more for actually moving, so it beats standing still
        self.rewards.track_lin_vel_xy_exp.weight = 5.0  # was 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.5  # was 1.5

        # loosen the penalties that fight climbing: quick leg motions, big leg bends, rising up the slope
        self.rewards.action_rate_l2.weight = -0.0025  # was -0.01
        self.rewards.joint_pos_penalty.weight = -0.25  # was -1.0
        self.rewards.lin_vel_z_l2.weight = -0.5  # was -2.0; climbing means the body must go up

        # lift the feet higher; steep slopes can still need the legs to clear the ground
        self.rewards.feet_height_body.weight = -2.0  # was 0 (off)
        self.rewards.feet_height_body.params["target_height"] = -0.30  # was -0.4

        # parent's disable_zero_weight_rewards() is gated on the base class name;
        # re-run it for this subclass
        if self.__class__.__name__ == "UnitreeB2WSlopeUpTeacherEnvCfg":
            self.disable_zero_weight_rewards()
