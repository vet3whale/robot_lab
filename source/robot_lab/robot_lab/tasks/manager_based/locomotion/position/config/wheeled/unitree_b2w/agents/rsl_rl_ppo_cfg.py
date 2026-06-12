# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeB2WPositionRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2w_position_rough"
    policy = RslRlPpoActorCriticCfg(
        # mirror the velocity task: full exploration. The earlier tighter std + low entropy were needed
        # because the tanh position reward saturated flat far from the goal, so the entropy bonus
        # inflated the unbounded std until the policy collapsed. The dense move_to_goal reward now keeps
        # std bounded, so we can afford velocity-level exploration to discover tall-step climbing.
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
        # mirror the velocity task (0.01). Safe now that the dense move_to_goal reward provides a
        # gradient everywhere to bound the action std (previously the saturating tanh position reward
        # let this entropy bonus inflate the std until the policy collapsed, forcing 0.002).
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
class UnitreeB2WPositionStaircaseUpPPORunnerCfg(UnitreeB2WPositionRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2w_position_staircaseup"