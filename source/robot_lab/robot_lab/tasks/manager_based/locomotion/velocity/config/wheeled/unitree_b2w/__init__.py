# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2WRoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_v1:UnitreeB2WRoughEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WRoughV1PPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2WRoughDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2WRoughDistillationRunnerRecurrentCfg"
        ),
    },
)


gym.register(
    id="RobotLab-Isaac-Velocity-MultiExpert-Teacher-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multiexpert_teacher_env_cfg:UnitreeB2WMultiExpertTeacherEnvCfg",
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_multiexpert_distillation_cfg:UnitreeB2WMultiExpertDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Student-Finetune-Unitree-B2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_finetune_env_cfg:UnitreeB2WStudentFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2WStudentFinetunePPORunnerCfg",
    },
)
