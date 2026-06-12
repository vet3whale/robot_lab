# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with 2D pose-goal (position) commands.

A B2W-only parallel of the ``velocity`` package: the robot is driven to a 2D pose goal (go *there*)
instead of tracking a velocity setpoint. The only semantic difference from ``velocity`` lives in
``position_env_cfg.py`` (command, observations, position-tracking rewards, failure terminations, and
a native position terrain curriculum); everything else mirrors ``velocity``.
"""