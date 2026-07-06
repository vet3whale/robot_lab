# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""B2W rough-terrain config v2.

Approximate Table S2 variant: reproduces Table S2 ("Locomotion Rewards") of
Hoeller et al., "ANYmal Parkour: Learning Agile Navigation for Quadrupedal
Robots" (Science Robotics 2024, arXiv:2306.14874) using ONLY reward functions
that already exist in the repo. Where the existing function differs from the S2
equation it is accepted as an approximation (see the "Status" column below);
the exact, custom-function variant is v3.

Terrain distribution is inherited verbatim from v1 (60% stairs / 20% slopes /
20% discrete obstacles). Both the actor and the critic are given privileged
terrain access (elevation map) as in Table 3 of the paper.

    #   S2 term              weight   Repo reward term              Status
    ---------------------------------------------------------------------------
    1   Position tracking     10      track_lin_vel_xy_exp          Different (velocity)
    2   Heading tracking       5      track_ang_vel_z_exp           Different (ang-vel)
    3   Joint velocity        -0.001  joint_vel_l2 (+wheel)         Match
    4   Torque                -1e-5   joint_torques_l2 (+wheel)     Match
    5   Joint velocity limit  -1      joint_vel_limits (legs)       Appx (soft, clipped)
    6   Torque limit          -0.2    applied_torque_limits (legs)  Appx (saturation)
    7   Base acc.             -0.001  body_lin_acc_l2 (base)        Appx (un-squared, no ang.)
    8   Feet acc.             -0.002  body_lin_acc_l2 (feet)        Match
    9   Action rate           -0.01   action_rate_l2                Match
    10  Feet contact force    -1e-5   contact_forces (thr 700)      Appx (not squared)
    11  Don't wait            omit    -                             Omitted
    12  Move in direction     omit    -                             Omitted
    13  Stand at target       -0.5    stand_still (legs)            Appx (L1, +u gate)
    14  Collision             -1      undesired_contacts (non-foot) Appx (count)
    15  Stumble               -1      feet_stumble (feet)           Appx (4x, +u gate)
    16  Termination           -200    is_terminated                 Appx (generic flag)
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg_v1 import UnitreeB2WRoughEnvCfgV1


@configclass
class UnitreeB2WRoughEnvCfgV2(UnitreeB2WRoughEnvCfgV1):
    def __post_init__(self):
        # Runs v1 __post_init__: applies the v1 terrain generator and the base
        # B2W reward/observation setup (blind actor, height-scanning critic).
        super().__post_init__()

        # ------------------------------Observations------------------------------
        # Table 3: give the actor (Expert column) privileged terrain access - base
        # linear velocity and the elevation map. The critic already carries both.
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Rewards (Table S2, existing functions)------------------------------
        # Clean slate: zero every reward the parent enabled, then set only S2 terms.
        r = self.rewards
        for attr in dir(r):
            if attr.startswith("__"):
                continue
            term = getattr(r, attr)
            if term is not None and not callable(term) and hasattr(term, "weight"):
                term.weight = 0.0

        # 1-2 Position/Heading tracking (velocity-env task-role substitution)
        r.track_lin_vel_xy_exp.weight = 10.0
        r.track_ang_vel_z_exp.weight = 5.0

        # 3 Joint velocity: ||q_dot||^2 (Match)
        r.joint_vel_l2.weight = -0.001
        r.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        r.joint_vel_wheel_l2.weight = -0.001
        r.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        # 4 Torque: ||tau||^2 (Match)
        r.joint_torques_l2.weight = -1e-5
        r.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        r.joint_torques_wheel_l2.weight = -1e-5
        r.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        # 5 Joint velocity limit (Appx: soft limit, clipped at 1)
        r.joint_vel_limits.weight = -1.0
        r.joint_vel_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        r.joint_vel_limits.params["soft_ratio"] = 1.0

        # 6 Torque limit (Appx: actuator saturation |tau_applied - tau_computed|)
        r.applied_torque_limits.weight = -0.2
        r.applied_torque_limits.params["asset_cfg"].joint_names = self.leg_joint_names

        # 7 Base acc. (Appx: linear-accel norm only, un-squared, no angular term)
        r.body_lin_acc_l2.weight = -0.001
        r.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 8 Feet acc.: sum_f ||v_dot_f|| (Match, via body_lin_acc_l2 on the feet)
        r.feet_acc = RewTerm(
            func=mdp.body_lin_acc_l2,
            weight=-0.002,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name])},
        )

        # 9 Action rate (Match)
        r.action_rate_l2.weight = -0.01

        # 10 Feet contact force (Appx: un-squared hinge)
        r.contact_forces.weight = -1e-5
        r.contact_forces.params["threshold"] = 700.0
        r.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 13 Stand at target (Appx: L1 deviation + upright gate)
        r.stand_still.weight = -0.5
        r.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names

        # 14 Collision (Appx: count of non-foot bodies in contact)
        r.undesired_contacts.weight = -1.0
        r.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        # 15 Stumble (Appx: 4x ratio + upright gate)
        r.feet_stumble.weight = -1.0
        r.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 16 Termination (Appx: generic terminated flag)
        r.is_terminated.weight = -200.0

        # Drop the zero-weight (non-S2) terms.
        self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # S2 termination fires on base collision; the parent disables the
        # illegal_contact termination, so re-enable it on the base link.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
        )
