# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-task evaluation worker for the B2W cross-evaluation matrix.

Launches Isaac Sim, loads one policy (actor-only) onto one task's terrain, rolls out a fixed
window, scores success (distance-traverse), and appends one cell / per-level breakdown to the
eval CSV(s). One task per process - hydra binds
a single task per process - so the cross-terrain sweep is driven by the orchestrator
``scripts/evaluation/evaluation.py``, which launches this worker once per (policy, terrain) pair.

This holds the evaluation logic that previously lived inline in
``scripts/reinforcement_learning/rsl_rl/play_cs.py`` so that play_cs.py stays a pure play script.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# reuse the rsl_rl scripts' cli_args helper (lives next to play_cs.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy on one task's terrain.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# --- cross-evaluation matrix mode ---
# This worker always evaluates one policy on one task terrain, writes per-level CSV rows,
# and updates the sibling summary CSV. The policy is always loaded actor-only (see runner.load below).
# -------------------------------------
parser.add_argument("--out_csv", type=str, default="eval_matrix.csv", help="CSV the eval cell is appended to.")
parser.add_argument("--eval_levels", type=int, default=9, help="Number of terrain difficulty levels to evaluate.")
parser.add_argument("--eval_duration_s", type=float, default=None, help="Seconds to run velocity-task evaluation rollouts.")
parser.add_argument(
    "--eval_robots_per_level", type=int, default=8, help="Robots to spawn on each terrain difficulty level."
)
parser.add_argument("--eval_policy_label", type=str, default=None, help="Policy label written to per-level eval CSV.")
parser.add_argument("--eval_terrain_label", type=str, default=None, help="Terrain label written to per-level eval CSV.")
parser.add_argument("--eval_command_x", type=float, default=0.6, help="Forward velocity command used by evaluation.")
parser.add_argument("--eval_command_y", type=float, default=0.0, help="Lateral velocity command used by evaluation.")
parser.add_argument("--eval_command_yaw", type=float, default=0.0, help="Yaw velocity command used by evaluation.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip


# --- difficulty range ---
# Evaluation sweeps terrain difficulty across `--eval_levels` rows. The difficulty fraction is
# relative to the policy's *trained* max difficulty: 1.0 == the hardest terrain seen in training,
# >1.0 extrapolates the terrain parameters past training (taller steps, steeper slopes, wider
# gaps - linear with no clamp, see isaaclab mesh_terrains). Rows interpolate linearly from the low
# to the high of this range. This is the only knob for the difficulty sweep and intentionally not
# a CLI flag.
EVAL_DIFFICULTY_RANGE = (0.9, 0.9)  # (low, high) difficulty fraction across the level rows


# --- distance-traverse success threshold (velocity tasks, which have no pose goal) ---
# Velocity policies can't use the goal-reached metric, so "success" must mean they actually crossed
# the terrain - NOT merely that they stayed upright (a robot stalled in a stair pit never falls and
# would trivially "succeed", masking any difficulty increase). A robot succeeds if it walked at
# least this fraction of the terrain size from its spawn without falling. This mirrors the training
# curriculum's move_up criterion (isaaclab terrain_levels_vel: distance > terrain size / 2).
EVAL_TRAVERSE_FRACTION = 0.5


def eval_level_difficulty(level: int, num_levels: int) -> float:
    """Nominal difficulty fraction of an eval level row (matches the curriculum row interpolation)."""
    low, high = EVAL_DIFFICULTY_RANGE
    return low + (high - low) * (level + 0.5) / num_levels


def log_eval_matrix(out_csv, terrain, label, success_rate, mean_level):
    """Record one (terrain, policy) cell of the cross-evaluation matrix.

    Updates the cell `(row=terrain, col=label)` in `out_csv` with success % and mean terrain
    level, rewrites the CSV, and prints the full matrix as markdown. Safe to call once per
    (policy, terrain) run; later calls accumulate into the same CSV.
    """
    import csv

    out_csv = os.path.abspath(out_csv)
    rows = {}
    if os.path.isfile(out_csv):
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["terrain"]] = r

    row = rows.get(terrain, {"terrain": terrain})
    row[f"{label}_success"] = f"{success_rate * 100:.1f}"
    row[f"{label}_level"] = f"{mean_level:.2f}"
    rows[terrain] = row

    # union of all columns seen so far, terrain first
    cols = ["terrain"]
    for r in rows.values():
        for k in r:
            if k not in cols:
                cols.append(k)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)

    # markdown to stdout (paste into docs/README.md)
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows.values():
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


def log_eval_levels(out_csv, terrain, label, checkpoint, task, levels, success, duration_s=None):
    """Record per-terrain-level cleared counts for one (policy, terrain) evaluation run."""
    import csv

    out_csv = os.path.abspath(out_csv)
    key_fields = ("policy", "terrain", "terrain_level")
    columns = ["policy", "terrain", "terrain_level", "difficulty", "cleared", "total", "success_rate", "duration_s", "checkpoint", "task"]
    rows = {}
    if os.path.isfile(out_csv):
        with open(out_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = tuple(row[k] for k in key_fields)
                rows[key] = {col: row.get(col, "") for col in columns}

    for level in range(args_cli.eval_levels):
        mask = levels == level
        total = int(mask.sum().item())
        cleared = int(success[mask].sum().item()) if total else 0
        rate = 0.0 if total == 0 else cleared / total
        difficulty = eval_level_difficulty(level, args_cli.eval_levels)
        row = {
            "policy": label,
            "terrain": terrain,
            "terrain_level": str(level + 1),
            "difficulty": f"{difficulty:.3f}",
            "cleared": str(cleared),
            "total": str(total),
            "success_rate": f"{rate:.4f}",
            "duration_s": "" if duration_s is None else f"{duration_s:.1f}",
            "checkpoint": os.path.abspath(checkpoint),
            "task": task,
        }
        rows[(label, terrain, str(level + 1))] = row
        print(f"[eval] terrain={terrain} policy={label} level={level + 1} "
              f"diff={difficulty:.2f} cleared={cleared}/{total}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], k[1], int(k[2]))):
            writer.writerow(rows[key])


def place_eval_envs_by_level(env):
    """Place eval envs evenly across terrain difficulty rows before reset."""
    terrain = env.unwrapped.scene.terrain
    if terrain.terrain_origins is None:
        raise RuntimeError(f"Task {args_cli.task} does not expose generated terrain origins for level evaluation.")

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    expected = args_cli.eval_levels * args_cli.eval_robots_per_level
    if num_envs != expected:
        raise RuntimeError(f"Expected {expected} eval envs, got {num_envs}.")

    num_cols = terrain.terrain_origins.shape[1]
    levels = torch.arange(args_cli.eval_levels, device=device, dtype=torch.long).repeat_interleave(
        args_cli.eval_robots_per_level
    )
    types = torch.arange(num_envs, device=device, dtype=torch.long) % num_cols
    terrain.terrain_levels = levels.clone()
    terrain.terrain_types = types.clone()
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
    return levels.clone()


def run_eval(env, policy, resume_path, policy_nn=None, eval_levels=None):
    """Roll out one episode-length window, count falls, and log one matrix cell.

    Counts a robot as a failure if it falls at any step of the window - via the `illegal_contact`
    termination where the env defines it, otherwise via a base-orientation test (envs like B2W
    disable that termination). Reports success rate and the mean terrain difficulty of the survivors,
    then hands them to log_eval_matrix(). The row (terrain) is derived from --task and the
    column (policy) from the checkpoint's experiment-dir name. `policy_nn` is only needed on
    rsl-rl < 4.0.0 for the recurrent-state reset.
    """
    obs = env.get_observations()
    num_envs = env.unwrapped.num_envs
    fell = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
    horizon = int(env.unwrapped.max_episode_length)
    # Fall detection. The orientation test is always on: projected_gravity_b[:, 2] is ~-1 upright,
    # ~0 on the side, >0 upside down; past ~70 deg from vertical counts as fallen (well clear of
    # normal stair-climbing pitch). We keep it always-on because the `illegal_contact` termination is
    # not a reliable fall signal everywhere - B2W velocity disables it. Where illegal_contact IS
    # real, we OR it in.
    term_mgr = env.unwrapped.termination_manager
    use_contact_term = "illegal_contact" in term_mgr.active_terms
    robot = env.unwrapped.scene["robot"]
    grav_z_fallen = -0.3

    # Velocity tasks score on distance traversed instead of survival, so a robot that stalls in a
    # stair pit fails rather than trivially "surviving". Track planar distance from the spawn; sample
    # before stepping so the latched value is the terminal pre-reset state. terrain size / 2 matches
    # the training curriculum's "cleared the terrain" threshold.
    start_xy = robot.data.root_pos_w[:, :2].clone()
    traveled = torch.zeros(num_envs, device=env.unwrapped.device)

    with torch.inference_mode():
        for _ in range(horizon):
            traveled = torch.norm(robot.data.root_pos_w[:, :2] - start_xy, dim=1)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # mark any robot that has fallen this step (orientation always; illegal_contact if real)
            fell |= robot.data.projected_gravity_b[:, 2] > grav_z_fallen
            if use_contact_term:
                fell |= term_mgr.get_term("illegal_contact")
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

    # crossed the terrain (distance from spawn) AND never fell
    gen = getattr(env.unwrapped.scene.terrain.cfg, "terrain_generator", None)
    if gen is not None:
        traverse_dist = gen.size[0] * EVAL_TRAVERSE_FRACTION
    else:  # flat / no generator: fall back to the commanded forward distance
        traverse_dist = args_cli.eval_command_x * env.unwrapped.max_episode_length_s * EVAL_TRAVERSE_FRACTION
    success = (traveled >= traverse_dist) & ~fell
    metric = f"distance-traverse(>={traverse_dist:.2f}m)"
    print(f"[eval] metric={metric} fallen={int(fell.sum())}/{num_envs}")

    terrain = args_cli.eval_terrain_label or args_cli.task.replace("RobotLab-Isaac-Velocity-", "").split("-Unitree")[0]
    label = args_cli.eval_policy_label or os.path.basename(os.path.dirname(os.path.dirname(resume_path))).replace(
        "unitree_b2w_", ""
    )
    if eval_levels is None:
        eval_levels = env.unwrapped.scene.terrain.terrain_levels

    # Log the duration actually run.
    duration_s = float(env.unwrapped.max_episode_length_s)
    log_eval_levels(args_cli.out_csv, terrain, label, resume_path, args_cli.task, eval_levels, success, duration_s)

    # Collapse all difficulty levels into one summative success rate for this (terrain, policy):
    # sum(cleared over all levels) / sum(total over all levels).
    summary_csv = os.path.splitext(args_cli.out_csv)[0] + "_summary.csv"
    success_rate = success.float().mean().item()
    mean_level = 0.0 if success.sum() == 0 else eval_levels[success].float().mean().item()
    print(
        f"[eval] SUMMARY terrain={terrain} policy={label} "
        f"success={success_rate * 100:.1f}% (collapsed over {args_cli.eval_levels} levels) "
        f"mean_level={mean_level:.2f}"
    )
    log_eval_matrix(summary_csv, terrain, label, success_rate, mean_level)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent on one task's own terrain."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    # Eval mode: clean, repeatable measurement on the task's own terrain. Robots are placed
    # evenly across terrain difficulty rows so the CSV can report cleared counts per row.
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.scene.num_envs = args_cli.eval_levels * args_cli.eval_robots_per_level
    # Velocity tasks walk a fixed command for a fixed window, so eval_duration_s sets the rollout
    # length.
    if args_cli.eval_duration_s is not None:
        env_cfg.episode_length_s = args_cli.eval_duration_s
    # Eval should measure locomotion, not whether a random reset spawned the robot upside down.
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.eval_command_x, args_cli.eval_command_x)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.eval_command_y, args_cli.eval_command_y)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.eval_command_yaw, args_cli.eval_command_yaw)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)
        env_cfg.commands.base_velocity.debug_vis = False
    if env_cfg.scene.terrain.terrain_generator is not None:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.num_rows = args_cli.eval_levels
        # curriculum=True makes per-row difficulty monotonic, so the per-level CSV rows are
        # actually ordered by difficulty (place_eval_envs_by_level assumes this). The range
        # sweeps from the low to the high difficulty fraction (>1.0 extrapolates past training).
        gen.curriculum = True
        gen.difficulty_range = EVAL_DIFFICULTY_RANGE
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.eval_levels - 1

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    eval_levels = place_eval_envs_by_level(env)
    env.reset()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # eval loads policies into tasks they may not have trained on, so the saved critic (whose
    # privileged-obs size can differ per task, e.g. 247 on rough vs 60 on flat) would fail to
    # load - skip it; eval only runs the deployable network forward.
    #
    # The load_cfg keys are runner-specific: PPO checkpoints store actor/critic, while distillation
    # checkpoints store student/teacher (no actor_state_dict). get_inference_policy returns the
    # student, so for distillation we load only the student and skip the task-specific teacher -
    # passing the PPO {"actor", "critic"} keys to Distillation.load is a silent no-op that would
    # leave the student randomly initialized.
    if agent_cfg.class_name == "DistillationRunner":
        load_cfg = {"student": True, "teacher": False, "optimizer": False, "iteration": False}
    else:
        load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": False}
    runner.load(resume_path, load_cfg=load_cfg)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # rsl-rl < 4.0.0 resets recurrent state on the raw network rather than the inference policy
    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    # measure success + log one matrix cell, then exit
    run_eval(env, policy, resume_path, policy_nn, eval_levels)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()