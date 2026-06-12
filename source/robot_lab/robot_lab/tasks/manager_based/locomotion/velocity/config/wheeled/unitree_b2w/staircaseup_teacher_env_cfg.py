# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# v2 terrain: staircase specialist - ascending only.
# 100% inverted pyramid stairs (pit at centre, stairs climb outward) so the robot is
# always forced to climb UP out of a stepped pit.  Mesh and heightfield variants are
# split 50/50 so the robot sees both solid-edge steps (mesh) and the smoother
# continuous surface produced by the heightfield sampler.
# Difficulty range is wider than v0 (4 cm - 30 cm vs 5 cm - 23 cm) so row 9 is
# significantly harder than anything in the original rough curriculum.
STAIRCASEUP_TEACHER_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # solid-mesh inverted pyramid: sharp-edged steps, robot climbs out
        "mesh_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.04, 0.30),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # heightfield inverted pyramid: continuous sampled surface, same climbing task
        "hf_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.04, 0.30),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeB2WStaircaseUpTeacherEnvCfg(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # stair meshes generate many collision contacts, increase GPU tolerance
        self.sim.physx.gpu_collision_stack_size = 2**27

        # swap in the staircaseup terrain generator
        self.scene.terrain.terrain_generator = STAIRCASEUP_TEACHER_CFG

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

        # loosen the penalties that fight climbing: quick leg motions, big leg bends, rising onto a step
        self.rewards.action_rate_l2.weight = -0.0025  # was -0.01
        self.rewards.joint_pos_penalty.weight = -0.25  # was -1.0
        self.rewards.lin_vel_z_l2.weight = -0.5  # was -2.0; a step means the body must go up

        # lift the feet higher to clear taller steps
        self.rewards.feet_height_body.weight = -2.0  # was 0 (off)
        self.rewards.feet_height_body.params["target_height"] = -0.30  # was -0.4

        # parent's disable_zero_weight_rewards() is gated on the base class name;
        # re-run it for this subclass
        if self.__class__.__name__ == "UnitreeB2WStaircaseUpTeacherEnvCfg":
            self.disable_zero_weight_rewards()
