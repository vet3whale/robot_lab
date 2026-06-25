# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Dynamic combined-terrain env for simultaneous multi-expert distillation.

The terrain and the per-env expert routing are shaped from ``teachers.txt``: each listed teacher
task contributes its own ``sub_terrains`` into one merged curriculum terrain, and the column ->
expert map is precomputed from the generator's deterministic column assignment. Add an expert by
adding a line to ``teachers.txt`` and the terrain + routing reshape automatically.
"""

import copy
import glob
import os
import re

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# default location of the shared teachers.txt (override with env var TEACHERS_TXT)
TEACHERS_TXT = os.environ.get("TEACHERS_TXT", os.path.join(os.path.dirname(__file__), "teachers.txt"))


def resolve_checkpoint(path: str) -> str:
    """Accept a .pt file, a run dir, or an experiment dir; return the newest model_*.pt.

    For a directory, picks the highest-iteration ``model_<N>.pt`` in the most recent run
    (run folders sort chronologically by their timestamp name).
    """
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(path, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No 'model_*.pt' checkpoint found under: {path}")

    def _key(p: str) -> tuple[str, int]:
        run = os.path.basename(os.path.dirname(p))  # e.g. 2026-06-18_03-18-00
        idx = int(re.search(r"model_(\d+)\.pt", os.path.basename(p)).group(1))
        return (run, idx)

    return max(candidates, key=_key)


def parse_teachers_txt(path: str) -> list[tuple[str, str]]:
    """Each non-comment line: '<task_id> <expert_ckpt>'. Order = expert id."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            entries.append((parts[0], parts[1]))  # (task_id, ckpt)
    if not entries:
        raise ValueError(f"No teacher entries found in {path}")
    return entries


def build_multiexpert_terrain(
    entries: list[tuple[str, str]], num_rows: int = 10, num_cols: int = 20
) -> tuple[TerrainGeneratorCfg, list[int]]:
    """Merge each expert's sub_terrains into one curriculum terrain and map columns -> expert id."""
    merged: dict = {}
    subterrain_expert: list[int] = []  # expert id for each merged sub-terrain, in insertion order
    first_gen = None
    for expert_id, (task_id, _ckpt) in enumerate(entries):
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")  # instantiated teacher cfg
        gen = env_cfg.scene.terrain.terrain_generator
        # a flat teacher has no terrain generator; contribute a single flat sub-terrain so it
        # still gets its equal share of columns in the merged curriculum
        if gen is None:
            sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        else:
            first_gen = first_gen or gen
            sub_terrains = gen.sub_terrains
        # each teacher's sub-terrain proportions already sum to 1, giving every expert an equal
        # share of columns once the merged proportions are normalized below
        for name, sub_cfg in sub_terrains.items():
            sub_cfg = copy.deepcopy(sub_cfg)
            merged[f"e{expert_id}_{name}"] = sub_cfg  # prefix keeps keys unique across experts
            subterrain_expert.append(expert_id)

    combined = TerrainGeneratorCfg(
        size=first_gen.size,
        border_width=first_gen.border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=first_gen.horizontal_scale,
        vertical_scale=first_gen.vertical_scale,
        slope_threshold=first_gen.slope_threshold,
        use_cache=False,
        curriculum=True,
        sub_terrains=merged,
    )

    # deterministic column -> sub-terrain -> expert (matches _generate_curriculum_terrains)
    props = np.array([c.proportion for c in merged.values()], dtype=float)
    props /= props.sum()
    cum = np.cumsum(props)
    column_to_expert = [
        int(subterrain_expert[int(np.min(np.where(c / num_cols + 0.001 < cum)[0]))]) for c in range(num_cols)
    ]
    return combined, column_to_expert


@configclass
class UnitreeB2WMultiExpertTeacherEnvCfg(UnitreeB2WRoughEnvCfg):
    teachers_txt: str = TEACHERS_TXT
    teacher_checkpoints: list[str] = []  # filled in __post_init__, read by the algorithm
    column_to_expert: list[int] = []  # filled in __post_init__, read by the algorithm

    def __post_init__(self):
        super().__post_init__()

        entries = parse_teachers_txt(self.teachers_txt)
        self.teacher_checkpoints = [resolve_checkpoint(ckpt) for (_task, ckpt) in entries]

        gen, column_to_expert = build_multiexpert_terrain(entries)
        self.scene.terrain.terrain_generator = gen
        self.column_to_expert = column_to_expert

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0
        self.sim.physx.gpu_collision_stack_size = 2**27  # stair meshes -> many contacts

        if self.__class__.__name__ == "UnitreeB2WMultiExpertTeacherEnvCfg":
            self.disable_zero_weight_rewards()