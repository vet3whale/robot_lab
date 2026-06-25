# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@configclass
class UnitreeB2WRoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "unitree_b2w_rough"

    # The Phase-1 PPO actor checkpoint was trained on policy observations, so the
    # distillation teacher must use the same observation group when loading it.
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    # teacher dims MUST match the PPO actor exactly: [512, 256, 128]
    # (this checkpoint gets loaded from Phase 1)
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )

    # student is smaller and receives the deployable policy observations
    student = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        loss_type="mse",
    )


@configclass
class UnitreeB2WRoughDistillationRunnerRecurrentCfg(UnitreeB2WRoughDistillationRunnerCfg):
    """Recurrent (LSTM) variant — student uses memory to compensate for missing height scan."""

    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    # LSTM-specific algorithm tuning: widen the TBPTT window to the full rollout so the
    # student can integrate a gait cycle of history (velocity inference), and enable
    # gradient clipping to guard the recurrent student against exploding BPTT gradients.
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )
