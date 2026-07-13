# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationAlgorithmCfg, RslRlMLPModelCfg, RslRlRNNModelCfg

from .rsl_rl_distillation_cfg import UnitreeB2WRoughDistillationRunnerRecurrentCfg

# dotted path to the multi-teacher distillation algorithm (resolved by rsl_rl's resolve_callable)
_ALG = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.distillation.multiteacher.MultiTeacherDistillation"
# dotted path to the depth-aware recurrent student model (CNN encoders + 2-layer LSTM, Fig. 3)
_STUDENT = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.distillation.cnn_rnn_model.CNNRNNModel"


@configclass
class RslRlCNNRNNModelCfg(RslRlRNNModelCfg):
    """Config for the depth-aware CNN + LSTM student.

    Extends the recurrent (RNN) model config with the per-camera conv encoder (``cnn_cfg``), the
    per-image FC stack (``cnn_fc_dims``, last value = latent size), and the 1D groups that bypass the
    LSTM and join only at the head (``command_obs_groups``).
    """

    class_name: str = _STUDENT

    cnn_cfg: RslRlCNNModelCfg.CNNCfg = MISSING
    """Per-camera conv encoder; the same config is used for every depth group."""

    cnn_fc_dims: list[int] = MISSING
    """Per-image FC stack after the conv encoder. The last value is the 64-dim latent (paper Fig. 3)."""

    command_obs_groups: list[str] = MISSING
    """1D observation groups that bypass the LSTM and re-enter only at the head (velocity commands)."""


@configclass
class UnitreeB2WMultiExpertDistillationRunnerCfg(UnitreeB2WRoughDistillationRunnerRecurrentCfg):
    """Depth-aware LSTM student distilled from N frozen experts at once (per-env terrain routing).

    The teacher stays privileged (proprio + height_scan via the ``policy`` group); the student is
    deployable: proprio + velocity command + 2 depth maps, encoded by per-camera CNNs into a 2-layer
    LSTM (Rudin et al. "Parkour in the Wild", Table 3 / Figure 3). Teacher checkpoints come from the
    env's teachers.txt, not from this cfg.
    """

    experiment_name = "unitree_b2w_multiexpert"

    # Student sees its deployable groups; teacher keeps reading `policy` so its frozen weights load.
    obs_groups = {
        "student": ["student", "student_commands", "depth_front", "depth_rear"],
        "teacher": ["policy"],
    }

    # CNN + 2-layer LSTM student. Per image: 3 conv + max-pool (cnn_cfg) then 2 FC to 64 (cnn_fc_dims);
    # [depth latents, proprio] -> LSTM; head reads [LSTM out, proprio, commands].
    student = RslRlCNNRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,  # paper Fig. 3
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=3,
            stride=1,
            padding="none",
            activation="elu",
            max_pool=True,  # 3 conv + max-pool per image
            flatten=True,
        ),
        cnn_fc_dims=[128, 64],  # 2 FC layers to a 64-dim latent per image
        command_obs_groups=["student_commands"],
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=_ALG,
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )