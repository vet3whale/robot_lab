import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .multiexpert_teacher_env_cfg import add_depth_perception
from .rough_env_cfg import UnitreeB2WRoughEnvCfg

ROUGH_TERRAINS_FT_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "mesh_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.10),
            step_width=0.275,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.10),
            step_width=0.275,
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.176, 0.29),
            platform_width=2.0,
            border_width=0.25,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_range=(0.05, 0.20),
            obstacle_width_range=(0.4, 1.0),
            num_obstacles=8,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeB2WStudentFinetuneEnvCfg(UnitreeB2WRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_FT_CFG
        # Eval the student on the stock Rough-v0 terrain instead of the FT obstacle terrain

        # import copy
        # from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
        # self.scene.terrain.terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        self.sim.physx.gpu_collision_stack_size = 2**27
        add_depth_perception(self)
        if self.__class__.__name__ == "UnitreeB2WStudentFinetuneEnvCfg":
            self.disable_zero_weight_rewards()
