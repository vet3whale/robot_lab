# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg

from .rsl_rl_distillation_cfg import UnitreeB2WRoughDistillationRunnerRecurrentCfg

# dotted path to the multi-teacher distillation algorithm (resolved by rsl_rl's resolve_callable)
_ALG = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.distillation.multiteacher.MultiTeacherDistillation"


@configclass
class UnitreeB2WMultiExpertDistillationRunnerCfg(UnitreeB2WRoughDistillationRunnerRecurrentCfg):
    """LSTM student distilled from N frozen experts at once (per-env terrain routing).

    Reuses the recurrent single-teacher cfg wholesale (LSTM student, teacher MLP [512,256,128],
    obs_groups); only the algorithm class is swapped. Teacher checkpoints come from the env's
    teachers.txt, not from this cfg.
    """

    experiment_name = "unitree_b2w_multiexpert"

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=_ALG,
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )