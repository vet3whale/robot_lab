# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Table-4 style success-rate evaluation for the B2W policies.

Every policy is scored the same way, in the same environment: spawn N robots on one terrain,
hold one fixed velocity command for one episode, and count how many reach the goal distance
without falling.

Definitions used here:

* **Environment** - always ``RobotLab-Isaac-Velocity-Student-Finetune-Unitree-B2W-v0``. That env
  publishes every observation group any of the three policies needs (``policy`` with the height
  scan for the blind-to-vision expert, plus ``student`` / ``student_commands`` / ``depth_front`` /
  ``depth_rear`` for the vision students), so all three run on identical physics and identical
  spawns. Comparing policies across different envs would compare envs. Only the terrain generator
  is swapped, by ``--terrain``: ``finetune`` (the fine-tune terrain, in distribution for the
  fine-tuned student), ``rough_v1`` (the expert's own training terrain), or ``rough_v0``.
* **Command** - one constant ``(v_x, v_y, w_z)`` in the robot's base frame, never resampled.
  The heading controller is off, so the robot is not steered back onto a heading; it must hold
  the command itself.
* **Goal** - the "checkpoint to reach": a point ``GOAL_FRACTION * terrain_patch_size`` metres from
  the spawn, along the commanded direction in the world frame. Success needs *displacement along
  that direction*, not distance travelled, so a robot that circles or slides sideways fails.
  ``GOAL_FRACTION = 0.5`` = half a patch, the same bar the training curriculum
  (``mdp.terrain_levels_vel``) uses to promote a robot to a harder terrain row.
* **Success** - reached the goal at some point in the episode AND never fell. Both conditions are
  needed: this env disables the ``illegal_contact`` termination, so a robot that flips does not
  terminate and would otherwise be scored on where its carcass slid to. Falling is measured from
  gravity in the base frame (``projected_gravity_b[:, 2] > -0.3``, roughly 70 deg off vertical -
  well past any legitimate stair-climbing pitch).

Difficulty is fixed, not swept. The paper reports "the average success rate computed by collecting
1000 roll-outs in simulation on randomized terrains at 90% of the maximum training difficulty", so
the generator runs in its random (non-curriculum) mode with ``difficulty_range`` collapsed to the
single point ``DIFFICULTY = 0.9``: every patch is an independently randomized shape, all at 90% of
the hardest terrain seen in training. The robots are spread evenly over the ``NUM_ROWS x NUM_COLS``
grid of those patches, so the result is one success rate over N randomized rollouts.

Usage::

    setup_isaaclab
    cd /workspace/near-locomotion-quadruped/robot_lab
    python scripts/evaluation/evaluate.py --policy student_finetune --terrain all --num_envs 1000 --headless

``--policy`` takes one of the names in ``POLICIES`` below, or a path to a ``model_*.pt`` / run dir.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------------------------
# Evaluation protocol. These are deliberately constants, not flags: a success rate is only
# comparable across policies if every policy was scored against the same protocol.
# ---------------------------------------------------------------------------------------------

TASK = "RobotLab-Isaac-Velocity-Student-Finetune-Unitree-B2W-v0"

# Constant base-frame velocity command held for the whole episode: (v_x [m/s], v_y [m/s], w_z [rad/s]).
COMMAND = (0.6, 0.0, 0.0)

# Episode length in seconds. At 0.6 m/s a robot covers 12 m, three times the 4 m goal, so the
# window is generous enough that failure means "could not", not "ran out of time".
EPISODE_S = 20.0

# Goal distance as a fraction of the terrain patch size (patches are 8 x 8 m -> goal = 4 m).
GOAL_FRACTION = 0.5

# Terrain difficulty, as a fraction of the maximum training difficulty. The paper evaluates at 90%.
# Setting the generator's difficulty_range to (0.9, 0.9) pins every patch to exactly this value.
DIFFICULTY = 0.9

# Grid of independently sampled terrain patches, all at DIFFICULTY. Rows no longer mean difficulty
# levels; rows x cols is just how many distinct randomized patches the robots are spread over.
NUM_ROWS = 10
NUM_COLS = 20

# projected_gravity_b[:, 2] is -1 upright, 0 on its side, +1 upside down.
FALLEN_GRAVITY_Z = -0.3

OUT_CSV = "evaluation_table.csv"

# Named policies. Each entry is (checkpoint, description). The checkpoints are pinned rather than
# resolved to "newest run" because the newest run in each experiment dir is usually a smoke test.
POLICIES = {
    "expert": (
        "logs/rsl_rl/unitree_b2w_rough_v1/2026-07-07_06-00-00_height_scan_enabled/model_6500.pt",
        "privileged PPO expert (proprio + height scan), the distillation teacher",
    ),
    "student": (
        "logs/rsl_rl/unitree_b2w_multiexpert/2026-07-15_14-19-16_walking_expert_distillation/model_25998.pt",
        "distilled depth student (CNN + LSTM), before RL fine-tuning",
    ),
    "student_finetune": (
        "logs/rsl_rl/unitree_b2w_student_finetune/2026-07-20_10-12-43_phaseB_10000iter/model_9999.pt",
        "distilled student after PPO fine-tuning",
    ),
}

# Agent config used to rebuild each architecture. The distilled student and the fine-tuned student
# are the same network (the fine-tune cfg's `actor` is a deepcopy of the distillation cfg's
# `student`), so one entry point rebuilds both; only the checkpoint key differs.
EXPERT_AGENT_TASK = "RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v1"

# Terrain sets, borrowed from the tasks that define them rather than redeclared here, so a change
# to a training terrain shows up in the evaluation instead of silently drifting from it. Only the
# terrain generator is taken; everything else still comes from the fine-tune env.
TERRAINS = {
    "rough_v0": "RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0",
    "rough_v1": "RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v1",
    "finetune": TASK,
}

parser = argparse.ArgumentParser(description="Table-4 style success-rate evaluation for B2W policies.")
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    help=f"One of {sorted(POLICIES)}, or a path to a model_*.pt / run directory.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default="finetune",
    choices=sorted(TERRAINS),
    help="Which task's terrain set to evaluate on.",
)
parser.add_argument("--num_envs", type=int, default=1000, help="Number of robots to evaluate.")
parser.add_argument("--gui", action="store_true", help="Watch the robots attempt the terrain in the Isaac Sim GUI.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Evaluation is a headless batch measurement unless --gui asks to watch it. Rendering 1000 robots
# is slow, so pair --gui with a small --num_envs when you actually want to see what is happening.
args_cli.headless = not args_cli.gui

# clear out sys.argv so downstream libraries do not re-parse our flags
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import csv
import glob
import importlib.metadata as metadata
import re

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply, yaw_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401  # registers the gym task ids used below  # isort: skip

installed_version = metadata.version("rsl-rl-lib")


def resolve_checkpoint(spec: str) -> str:
    """Turn a policy name, a run directory, or a .pt path into one checkpoint path."""
    path = POLICIES[spec][0] if spec in POLICIES else spec
    if os.path.isfile(path):
        return os.path.abspath(path)
    candidates = glob.glob(os.path.join(path, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"--policy '{spec}' is neither a name in {sorted(POLICIES)} nor a path holding a model_*.pt file."
        )
    # newest run directory first, then highest iteration inside it
    return os.path.abspath(
        max(candidates, key=lambda p: (os.path.basename(os.path.dirname(p)), int(re.search(r"(\d+)\.pt$", p).group(1))))
    )


def load_actor_weights(checkpoint: str) -> tuple[dict, bool]:
    """Read the actor weights out of a checkpoint and report whether it is the depth student.

    A distillation checkpoint stores the deployable network under ``student_state_dict``; a PPO
    checkpoint stores it under ``actor_state_dict``. The architecture is told apart by the weights
    themselves rather than by the file's location: only the CNN + LSTM student has ``cnns.*`` keys.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = saved.get("student_state_dict", saved.get("actor_state_dict"))
    if state_dict is None:
        raise KeyError(f"{checkpoint} contains neither 'student_state_dict' nor 'actor_state_dict'.")
    is_depth_student = any(key.startswith("cnns.") for key in state_dict)
    return state_dict, is_depth_student


def build_env_cfg(is_depth_student: bool):
    """Fine-tune env, rebuilt as a measurement rig: one terrain, one command, no randomization."""
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = EPISODE_S
    env_cfg.seed = 0

    # --- terrain: the selected task's terrain set, every patch at 90% of max training difficulty ---
    source_cfg = load_cfg_from_registry(TERRAINS[args_cli.terrain], "env_cfg_entry_point")
    gen = copy.deepcopy(source_cfg.scene.terrain.terrain_generator)
    if gen is None:
        raise ValueError(f"--terrain '{args_cli.terrain}' resolves to a task with no terrain generator (flat ground).")
    gen.num_rows = NUM_ROWS
    gen.num_cols = NUM_COLS
    # curriculum=False makes the generator sample each patch independently: a random sub-terrain
    # and a difficulty drawn from difficulty_range. The range is still required - it defaults to
    # (0.0, 1.0), which would spread patches over the whole difficulty span. Collapsing it to a
    # single point is what pins every patch to DIFFICULTY while leaving the shape randomized.
    gen.curriculum = False
    gen.difficulty_range = (DIFFICULTY, DIFFICULTY)
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.terrain.max_init_terrain_level = NUM_ROWS - 1

    # --- no curriculum: nothing may move a robot to a different patch mid-run ---
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    # --- one constant command, never resampled, no heading controller ---
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (COMMAND[0], COMMAND[0])
    cmd.ranges.lin_vel_y = (COMMAND[1], COMMAND[1])
    cmd.ranges.ang_vel_z = (COMMAND[2], COMMAND[2])
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (EPISODE_S, EPISODE_S)
    cmd.debug_vis = True

    # --- clean spawns: measure locomotion, not whether a robot was dropped in upside down ---
    zero = (0.0, 0.0)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {"x": zero, "y": zero, "z": zero, "roll": zero, "pitch": zero, "yaw": zero},
        "velocity_range": {"x": zero, "y": zero, "z": zero, "roll": zero, "pitch": zero, "yaw": zero},
    }
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None

    # --- no observation noise. The depth domain randomization stays on: it is the student's
    # sensor model, not noise, and removing it would score the student on images it never saw. ---
    for name in env_cfg.observations.__dict__:
        group = getattr(env_cfg.observations, name)
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False

    # --- sensor visualization: show only the sensor the running policy actually reads, so what is
    # on screen is what the policy sees. The env carries both sets regardless of who is driving:
    # the expert reads the height scan and is blind to the cameras, the students read the depth
    # cameras and never see the height scan. Off entirely when headless - the visualizers cost
    # draw calls and nothing renders them. ---
    height_scan_vis = args_cli.gui and not is_depth_student
    depth_cam_vis = args_cli.gui and is_depth_student
    for sensor_name in ("height_scanner", "height_scanner_base"):
        sensor = getattr(env_cfg.scene, sensor_name, None)
        if sensor is not None:
            sensor.debug_vis = height_scan_vis
    for sensor_name in ("depth_cam_front", "depth_cam_rear"):
        sensor = getattr(env_cfg.scene, sensor_name, None)
        if sensor is not None:
            sensor.debug_vis = depth_cam_vis

    return env_cfg


def place_envs_across_patches(env) -> None:
    """Spread the robots evenly over the grid of randomized patches before the reset.

    Every patch sits at the same difficulty, so this is purely about sampling the generator's
    randomization evenly instead of letting the importer drop robots wherever it likes.
    """
    terrain = env.unwrapped.scene.terrain
    if terrain.terrain_origins is None:
        raise RuntimeError(f"Task {TASK} exposes no generated terrain origins to place robots on.")

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    index = torch.arange(num_envs, device=device)
    rows = (index // num_cols) % num_rows
    cols = index % num_cols

    terrain.terrain_levels = rows.clone()
    terrain.terrain_types = cols.clone()
    terrain.env_origins[:] = terrain.terrain_origins[rows, cols]


def build_agent_cfg(is_depth_student: bool):
    """Pick the agent config that rebuilds the architecture found in the checkpoint."""
    agent_task = TASK if is_depth_student else EXPERT_AGENT_TASK
    arch = "CNN + LSTM depth student (depth cameras)" if is_depth_student else "MLP expert (height scan)"
    print(f"[eval] architecture: {arch} (agent cfg from {agent_task})")
    agent_cfg = load_cfg_from_registry(agent_task, "rsl_rl_cfg_entry_point")
    return handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)


def build_policy(env, agent_cfg, state_dict: dict):
    """Build the network from ``agent_cfg``, load the checkpoint's actor weights, return it."""
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Actor only, loaded by hand rather than via runner.load(). The three checkpoints do not share
    # a schema: a distillation checkpoint has student_state_dict / teacher_state_dict and no
    # actor_state_dict at all, so PPO.load's default path raises KeyError on it. The critic is
    # never used for inference either way.
    runner.alg._raw_actor.load_state_dict(state_dict, strict=True)
    return runner.get_inference_policy(device=env.unwrapped.device)


def goal_direction(env) -> torch.Tensor:
    """World-frame unit vector the robots must make progress along, from their spawn heading."""
    robot = env.unwrapped.scene["robot"]
    linear = torch.tensor([COMMAND[0], COMMAND[1], 0.0], device=env.unwrapped.device)
    if torch.norm(linear) < 1e-6:
        return None  # pure yaw command: there is no direction to make progress along
    linear = linear / torch.norm(linear)
    heading = yaw_quat(robot.data.root_quat_w)  # spawn yaw, roll/pitch stripped
    return quat_apply(heading, linear.expand(env.unwrapped.num_envs, 3))[:, :2]


def rollout(env, policy) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run one episode per robot and return (reached_goal, fell, goal_distance)."""
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    horizon = int(env.unwrapped.max_episode_length)

    goal_distance = env.unwrapped.scene.terrain.cfg.terrain_generator.size[0] * GOAL_FRACTION
    direction = goal_direction(env)
    start_xy = robot.data.root_pos_w[:, :2].clone()

    reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Each robot is scored on its first episode only. Anything after a reset is a different trial
    # from a different spawn, so it must not leak into this one's score.
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def sample():
        live = ~finished
        displacement = robot.data.root_pos_w[:, :2] - start_xy
        progress = torch.norm(displacement, dim=1) if direction is None else (displacement * direction).sum(dim=1)
        reached.logical_or_(live & (progress >= goal_distance))
        fell.logical_or_(live & (robot.data.projected_gravity_b[:, 2] > FALLEN_GRAVITY_Z))

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(horizon):
            sample()
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            finished.logical_or_(dones.bool())
        sample()  # final state of the robots that never terminated

    return reached, fell, goal_distance


def report(policy_label: str, success: torch.Tensor) -> None:
    """Update this run's cell of the evaluation matrix and print the whole matrix.

    The CSV is a matrix, not a log: one row per terrain, one ``<policy>_success`` column per
    policy, so the file reads as the comparison itself. Each run overwrites its own cell and
    leaves every other cell alone, which means the three policies can be run in any order, one
    at a time, and the table fills in.
    """
    cleared = int(success.sum())
    rollouts = success.numel()
    rate = success.float().mean().item()

    print(f"\n[eval] policy={policy_label} terrain={args_cli.terrain} difficulty={DIFFICULTY:g}")
    print(f"[eval] SUCCESS {cleared}/{rollouts} = {rate * 100:.1f}%")

    out_csv = os.path.abspath(OUT_CSV)
    rows: dict[str, dict[str, str]] = {}
    if os.path.isfile(out_csv):
        with open(out_csv, newline="") as handle:
            for existing in csv.DictReader(handle):
                rows[existing["terrain"]] = existing

    row = rows.setdefault(args_cli.terrain, {"terrain": args_cli.terrain})
    row[f"{policy_label}_success"] = f"{rate * 100:.1f}"
    row[f"{policy_label}_rollouts"] = str(rollouts)

    # union of every column seen so far, terrain first, so a new policy just widens the table
    columns = ["terrain"]
    for existing in rows.values():
        columns.extend(key for key in existing if key not in columns)

    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows.values())

    print(f"[eval] {out_csv}\n")
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for existing in rows.values():
        print("| " + " | ".join(str(existing.get(column, "")) for column in columns) + " |")


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be positive.")

    checkpoint = resolve_checkpoint(args_cli.policy)
    policy_label = args_cli.policy if args_cli.policy in POLICIES else os.path.basename(os.path.dirname(checkpoint))
    description = POLICIES[args_cli.policy][1] if args_cli.policy in POLICIES else "checkpoint given by path"
    print(f"[eval] policy '{policy_label}' ({description})\n[eval] checkpoint: {checkpoint}")

    state_dict, is_depth_student = load_actor_weights(checkpoint)
    agent_cfg = build_agent_cfg(is_depth_student)

    env = gym.make(TASK, cfg=build_env_cfg(is_depth_student), render_mode=None)
    place_envs_across_patches(env)
    env.reset()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = build_policy(env, agent_cfg, state_dict)

    reached, fell, goal_distance = rollout(env, policy)
    success = reached & ~fell
    print(
        f"[eval] goal={goal_distance:.2f} m along the command direction | "
        f"reached={int(reached.sum())}/{reached.numel()} fell={int(fell.sum())}/{fell.numel()}"
    )
    report(policy_label, success)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
