# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator for B2W terrain-level evaluation.

This script resolves policy paths, infers which terrains should be evaluated from those policy
experiment names, and launches the eval_worker.py worker once per (policy, terrain) pair. Isaac
Sim is only started by the worker, never by this orchestrator.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_LAB_ROOT = REPO_ROOT / "robot_lab"
LOGS_ROOT = ROBOT_LAB_ROOT / "logs" / "rsl_rl"
EVAL_WORKER = Path(__file__).resolve().parent / "eval_worker.py"

KNOWN_TERRAINS = {
    "flat": "Flat",
    "rough": "Rough",
    "rough_v1": "RoughV1",
    "slopeup": "SlopeUp",
    "slopeup_teacher": "SlopeUp",
    "staircaseup": "StaircaseUp",
    "staircaseup_teacher": "StaircaseUp",
    "multiexpert": "MultiExpert",
}

DEFAULT_AGENT = "rsl_rl_cfg_entry_point"

# Policies whose actor architecture is not the task's default PPO MLP must be loaded with a
# matching agent entry point, otherwise the actor-only checkpoint load fails (e.g. an LSTM
# distillation student cannot load into a PPO MLP actor). The target task must also register the
# chosen entry point; the B2W teacher tasks (rough/slopeup/staircaseup) all register the
# distillation-recurrent one.
KNOWN_EXPERIMENT_AGENTS = {
    "unitree_b2w_multiexpert": "rsl_rl_distillation_recurrent_cfg_entry_point",
}

KNOWN_EXPERIMENT_TASKS = {
    "unitree_b2w_flat": ("Flat", "velocity", "RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0"),
    "unitree_b2w_rough": ("Rough", "velocity", "RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0"),
    "unitree_b2w_rough_v1": ("RoughV1", "velocity", "RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v1"),
    "unitree_b2w_slopeup_teacher": (
        "SlopeUp",
        "velocity",
        "RobotLab-Isaac-Velocity-SlopeUp-Teacher-Unitree-B2W-v0",
    ),
    "unitree_b2w_staircaseup_teacher": (
        "StaircaseUp",
        "velocity",
        "RobotLab-Isaac-Velocity-StaircaseUp-Teacher-Unitree-B2W-v0",
    ),
    "unitree_b2w_multiexpert": (
        "MultiExpert",
        "velocity",
        "RobotLab-Isaac-Velocity-MultiExpert-Teacher-Unitree-B2W-v0",
    ),
}

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
MODEL_RE = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class TerrainSpec:
    name: str
    family: str
    task: str


@dataclass(frozen=True)
class PolicySpec:
    input_path: str
    experiment: str
    label: str
    checkpoint: str
    terrain: TerrainSpec
    agent: str


def model_index(path: str) -> int:
    match = MODEL_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else -1


def newest_checkpoint_in_run(run_dir: str) -> str | None:
    models = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(models, key=model_index) if models else None


def newest_checkpoint(experiment_dir: str) -> str | None:
    direct = newest_checkpoint_in_run(experiment_dir)
    if direct is not None:
        return direct

    runs = sorted(d for d in glob.glob(os.path.join(experiment_dir, "*")) if os.path.isdir(d))
    for run in reversed(runs):
        ckpt = newest_checkpoint_in_run(run)
        if ckpt is not None:
            return ckpt
    return None


def experiment_from_path(path: str) -> str:
    path = os.path.abspath(path)
    if path.endswith(".pt"):
        return os.path.basename(os.path.dirname(os.path.dirname(path)))

    base = os.path.basename(path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    if TIMESTAMP_RE.match(base) and parent:
        return parent
    return base


def available_experiments() -> list[str]:
    if not LOGS_ROOT.is_dir():
        return []
    return sorted(path.name for path in LOGS_ROOT.iterdir() if path.is_dir())


def infer_terrain_from_experiment(experiment: str) -> TerrainSpec:
    known = KNOWN_EXPERIMENT_TASKS.get(experiment)
    if known is not None:
        terrain, family, task = known
        return TerrainSpec(terrain, family, task)

    if experiment.startswith("unitree_b2w_"):
        terrain_key = experiment.removeprefix("unitree_b2w_")
        terrain = KNOWN_TERRAINS.get(terrain_key)
        if terrain is not None:
            suffix = "-Teacher" if terrain_key.endswith("_teacher") else ""
            version = "v1" if terrain_key == "rough_v1" else "v0"
            return TerrainSpec(terrain, "velocity", f"RobotLab-Isaac-Velocity-{terrain}{suffix}-Unitree-B2W-{version}")

    available = ", ".join(available_experiments()) or "<none found>"
    raise ValueError(
        f"Could not infer terrain/task for experiment '{experiment}'. "
        f"Available experiments under {LOGS_ROOT}: {available}"
    )


def terrain_spec_from_name(name: str) -> TerrainSpec:
    """Resolve a user-supplied --terrain value (a KNOWN_TERRAINS key) into a TerrainSpec."""
    key = name.strip().lower()
    experiment = key if key.startswith("unitree_b2w_") else f"unitree_b2w_{key}"
    try:
        return infer_terrain_from_experiment(experiment)
    except ValueError:
        valid = ", ".join(sorted(KNOWN_TERRAINS))
        raise ValueError(f"Unknown --terrain '{name}'. Valid terrains: {valid}")


def unique_terrains_for_family(policies: list[PolicySpec], family: str) -> list[TerrainSpec]:
    terrains: list[TerrainSpec] = []
    seen = set()
    for policy in policies:
        terrain = policy.terrain
        key = (terrain.family, terrain.name, terrain.task)
        if terrain.family == family and key not in seen:
            terrains.append(terrain)
            seen.add(key)
    return terrains


def build_eval_runs(
    policies: list[PolicySpec], terrains: list[TerrainSpec] | None = None
) -> list[tuple[PolicySpec, TerrainSpec]]:
    runs: list[tuple[PolicySpec, TerrainSpec]] = []
    # Explicit --terrain overrides inference: every policy is evaluated on every requested terrain.
    if terrains:
        for policy in policies:
            for terrain in terrains:
                runs.append((policy, terrain))
        return runs
    terrains_by_family = {
        family: unique_terrains_for_family(policies, family)
        for family in sorted({policy.terrain.family for policy in policies})
    }
    for policy in policies:
        for terrain in terrains_by_family[policy.terrain.family]:
            runs.append((policy, terrain))
    return runs


def resolve_policy(spec: str) -> PolicySpec:
    candidate = spec
    if not os.path.exists(candidate):
        candidate = str(LOGS_ROOT / spec)

    if candidate.endswith(".pt"):
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Policy checkpoint does not exist: {spec}")
        checkpoint = os.path.abspath(candidate)
    else:
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Policy path does not exist: {spec}")
        checkpoint = newest_checkpoint(candidate)
        if checkpoint is None:
            raise FileNotFoundError(f"No model_*.pt checkpoint found under: {candidate}")
        checkpoint = os.path.abspath(checkpoint)

    experiment = experiment_from_path(candidate)
    label = experiment.removeprefix("unitree_b2w_")
    terrain = infer_terrain_from_experiment(experiment)
    agent = KNOWN_EXPERIMENT_AGENTS.get(experiment, DEFAULT_AGENT)
    return PolicySpec(
        input_path=spec, experiment=experiment, label=label, checkpoint=checkpoint, terrain=terrain, agent=agent
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policies on the terrains they were trained on.")
    parser.add_argument("--policies", nargs="+", required=True, help="Policy run dirs, experiment dirs, names, or .pt files.")
    parser.add_argument(
        "--terrain",
        nargs="+",
        default=None,
        metavar="TERRAIN",
        help=(
            "Explicit terrain(s) to evaluate every policy on, e.g. --terrain flat rough staircaseup_teacher. "
            f"Choices: {', '.join(sorted(KNOWN_TERRAINS))}. "
            "If omitted, each policy is tested on the terrains inferred from its experiment name."
        ),
    )
    parser.add_argument("--out_csv", default="evaluation.csv", help="CSV output path.")
    parser.add_argument("--levels", type=int, default=9, help="Number of difficulty levels to test per terrain.")
    parser.add_argument("--robots_per_level", type=int, default=512, help="Robots spawned on each terrain level.")
    parser.add_argument(
        "--duration_s",
        type=float,
        default=None,
        help="Seconds to run each evaluation rollout. Default is chosen from the terrain.",
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac workers headlessly.")
    parser.add_argument("extras", nargs="*", help=argparse.SUPPRESS)
    args, extras = parser.parse_known_args(argv)
    args.extras.extend(extras)
    # Be forgiving of the typo-like form: `-- headless`.
    if "headless" in args.extras:
        args.headless = True
        args.extras = [x for x in args.extras if x != "headless"]
    return args



def validate_robot_count(robots_per_level: int) -> None:
    if robots_per_level <= 0:
        raise ValueError("--robots_per_level must be positive")


def auto_duration_s(terrain_name: str, robots_per_level: int) -> float:
    """Choose rollout duration from both terrain type and sample count.

    Harder traversal terrains get more time when the sample count is small. As robots_per_level
    grows, the large parallel sample gives a stable estimate, so the rollout can be shorter.
    """
    validate_robot_count(robots_per_level)
    terrain_base = 60.0 if terrain_name in {"StaircaseUp", "SlopeUp"} else 20.0
    if robots_per_level >= 512:
        return 20.0
    if robots_per_level >= 32:
        return min(terrain_base, 20.0)
    if robots_per_level >= 16:
        return min(terrain_base, 30.0)
    return terrain_base


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    policies = [resolve_policy(p) for p in args.policies]
    requested_terrains: list[TerrainSpec] | None = None
    if args.terrain:
        requested_terrains = []
        seen_keys = set()
        for name in args.terrain:
            terrain = terrain_spec_from_name(name)
            key = (terrain.family, terrain.name, terrain.task)
            if key not in seen_keys:
                requested_terrains.append(terrain)
                seen_keys.add(key)
    eval_runs = build_eval_runs(policies, requested_terrains)
    validate_robot_count(args.robots_per_level)
    out_csv = os.path.abspath(args.out_csv)

    inferred_terrains = []
    seen = set()
    for _, terrain in eval_runs:
        key = (terrain.family, terrain.name, terrain.task)
        if key not in seen:
            inferred_terrains.append(terrain)
            seen.add(key)

    print(f"[evaluation] worker: {EVAL_WORKER}")
    print(f"[evaluation] policies: {len(policies)}")
    for policy in policies:
        agent_note = "" if policy.agent == DEFAULT_AGENT else f" (agent={policy.agent})"
        print(f"  - {policy.label} [{policy.terrain.family}/{policy.terrain.name}]{agent_note}: {policy.checkpoint}")
    source = "requested via --terrain" if requested_terrains else "inferred from policies"
    print(
        f"[evaluation] terrains {source}: "
        + ", ".join(f"{terrain.family}/{terrain.name}" for terrain in inferred_terrains)
    )
    if not requested_terrains and len({policy.terrain.family for policy in policies}) > 1:
        print("[evaluation] mixed policy families detected; cross-evaluating only within matching families")
    print(f"[evaluation] levels per terrain: {args.levels}; robots per level: {args.robots_per_level}")
    if args.duration_s is None:
        auto_summary = ", ".join(
            f"{terrain.family}/{terrain.name}={auto_duration_s(terrain.name, args.robots_per_level):.0f}s"
            for terrain in inferred_terrains
        )
        print(f"[evaluation] duration per terrain: {auto_summary} (auto)")
    else:
        print(f"[evaluation] duration per run: {args.duration_s:.1f}s")
    print(f"[evaluation] writing: {out_csv}\n")

    failures = []
    total = len(eval_runs)
    for run_idx, (policy, terrain) in enumerate(eval_runs, start=1):
        duration_s = args.duration_s if args.duration_s is not None else auto_duration_s(terrain.name, args.robots_per_level)
        cmd = [
            sys.executable,
            str(EVAL_WORKER),
            f"--task={terrain.task}",
            "--checkpoint",
            policy.checkpoint,
            "--eval_policy_label",
            policy.label,
            "--eval_terrain_label",
            terrain.name,
            "--out_csv",
            out_csv,
            "--eval_levels",
            str(args.levels),
            "--eval_robots_per_level",
            str(args.robots_per_level),
            "--eval_duration_s",
            str(duration_s),
            "--agent",
            policy.agent,
        ]
        if args.headless:
            cmd.append("--headless")
        cmd += args.extras

        print(f"[evaluation] ({run_idx}/{total}) policy={policy.label} terrain={terrain.family}/{terrain.name}")
        result = subprocess.run(cmd, cwd=str(ROBOT_LAB_ROOT))
        if result.returncode != 0:
            failures.append((policy.label, f"{terrain.family}/{terrain.name}", result.returncode))
            print(f"[evaluation]   FAILED with exit code {result.returncode}")

    if failures:
        print("\n[evaluation] failed runs:")
        for policy, terrain, code in failures:
            print(f"  - policy={policy} terrain={terrain} exit={code}")
        return 1

    print(f"\n[evaluation] done: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
