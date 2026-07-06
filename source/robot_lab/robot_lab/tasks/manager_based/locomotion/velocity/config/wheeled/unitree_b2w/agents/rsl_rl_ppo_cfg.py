# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
class UnitreeB2WRoughV2PPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    # PPO hyperparameters follow the ANYmal Parkour paper (Hoeller et al. 2024).
    # The paper does not publish a hyperparameter table; it uses the standard
    # rsl_rl / legged_gym on-policy PPO setup, which the base
    # UnitreeB2WRoughPPORunnerCfg already matches exactly:
    #   4096 parallel envs, 24 steps/env, 5 epochs, 4 minibatches,
    #   gamma=0.99, lambda=0.95, clip=0.2, entropy=0.01, adaptive lr (kl=0.01),
    #   actor/critic MLP [512, 256, 128] with elu.
    # So v2 inherits those verbatim and only renames the experiment.
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_rough_v2"


@configclass
class UnitreeB2WRoughV3PPORunnerCfg(UnitreeB2WRoughPPORunnerCfg):
    # Same PPO hyperparameters as v2 (the paper's standard rsl_rl setup); v3 only
    # differs in the reward env (exact custom S2 functions vs v2's approximations).
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_rough_v3"


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

