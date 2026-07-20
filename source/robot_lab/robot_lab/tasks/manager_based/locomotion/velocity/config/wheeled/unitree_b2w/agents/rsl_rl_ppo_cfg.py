# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import copy

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .rsl_rl_multiexpert_distillation_cfg import UnitreeB2WMultiExpertDistillationRunnerCfg


@configclass
class UnitreeB2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2w_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2WFlatPPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2w_flat"


@configclass
class UnitreeB2WRoughV1PPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_rough_v1"


@configclass
class UnitreeB2WStudentFinetunePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_b2w_student_finetune"

    obs_groups = {
        "actor": ["student", "student_commands", "depth_front", "depth_rear"],
        "critic": ["policy"],  # privileged: proprio + height scan, same as the expert critic
    }

    actor = copy.deepcopy(UnitreeB2WMultiExpertDistillationRunnerCfg().student)
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)

    algorithm = RslRlPpoAlgorithmCfg(
        # expert values, except where noted
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        entropy_coef=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2WStaircaseUpTeacherPPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_staircaseup_teacher"


@configclass
class UnitreeB2WSlopeUpTeacherPPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_slopeup_teacher"
