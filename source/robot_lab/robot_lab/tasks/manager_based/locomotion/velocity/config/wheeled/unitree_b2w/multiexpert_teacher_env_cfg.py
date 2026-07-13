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
import math
import os
import re

import numpy as np
import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2WRoughEnvCfg

# default location of the shared teachers.txt (override with env var TEACHERS_TXT)
TEACHERS_TXT = os.environ.get("TEACHERS_TXT", os.path.join(os.path.dirname(__file__), "teachers.txt"))

# --- Depth-camera intrinsics (student perception, Rudin et al. "Parkour in the Wild", Sec. 2.4) ---
# The paper images are 48x32 px (W x H). Solve the pinhole focal length from the real D435i depth
# HFOV so sim matches hardware: focal_length = horizontal_aperture / (2 * tan(HFOV / 2)).
# Datasheet HFOV is 87 deg (+/- 3 deg unit spread, absorbed by the depth domain randomization);
# read the exact value off the robot (`rs-enumerate-devices -c`) before deployment.
DEPTH_WIDTH = 48
DEPTH_HEIGHT = 32
DEPTH_HFOV_DEG = 87.0
DEPTH_HORIZONTAL_APERTURE = 20.955  # PinholeCameraPatternCfg default (cm)
DEPTH_FOCAL_LENGTH = DEPTH_HORIZONTAL_APERTURE / (2 * math.tan(math.radians(DEPTH_HFOV_DEG) / 2))  # ~11.04 cm
DEPTH_MAX_RANGE = 2.0  # paper clips depth at 2 m; keep identical on the robot
DEPTH_MIN_RANGE = 0.15  # pixels closer than this are treated as empty (read far)

# Extrinsics from b2w_description.urdf (relative to base_link), ROS convention.
# Both cameras pitch 45 deg down (rpy roll = -2.3562 rad = -135 deg in the optical frame).
DEPTH_CAM_FRONT_POS = (0.42161, 0.025, 0.061851)
DEPTH_CAM_FRONT_RPY = (-2.3562, 0.0, -1.5708)
DEPTH_CAM_REAR_POS = (-0.42244, -0.025, 0.038026)
DEPTH_CAM_REAR_RPY = (-2.3562, 0.0, 1.5708)


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """URDF (roll, pitch, yaw) -> (w, x, y, z) quaternion for the sensor OffsetCfg."""
    q = quat_from_euler_xyz(torch.tensor(rpy[0]), torch.tensor(rpy[1]), torch.tensor(rpy[2]))
    return tuple(q.tolist())


def _depth_camera_cfg(base_link_path: str, pos, rpy) -> RayCasterCameraCfg:
    """A ray-cast depth camera matching the real D435i intrinsics/extrinsics.

    Ray-casts against ``/World/ground`` only (blind to the robot's own legs, like ``height_scanner``),
    so it is cheap for thousands of envs. Images update at 15 Hz while the policy runs at 50 Hz
    (paper Sec. 3.6); the sensor then serves the last-rendered image for ~3 policy steps, matching
    the staleness the real robot sees.
    """
    return RayCasterCameraCfg(
        prim_path=base_link_path,
        offset=RayCasterCameraCfg.OffsetCfg(pos=pos, rot=_quat_from_rpy(rpy), convention="ros"),
        pattern_cfg=PinholeCameraPatternCfg(
            width=DEPTH_WIDTH,
            height=DEPTH_HEIGHT,
            focal_length=DEPTH_FOCAL_LENGTH,
            horizontal_aperture=DEPTH_HORIZONTAL_APERTURE,
            # vertical_aperture=None -> square pixels, derived from the width/height aspect ratio
        ),
        data_types=["distance_to_image_plane"],  # Z-depth in metres (matches RealSense)
        depth_clipping_behavior="max",  # misses / out-of-range -> max_distance (far), paper convention
        max_distance=DEPTH_MAX_RANGE,
        mesh_prim_paths=["/World/ground"],
        update_period=1 / 15,  # paper Sec. 3.6: images at 15 Hz (do NOT copy the 50 Hz height_scanner)
        debug_vis=True,  # verify the frustum before scaling up
    )


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

        self._add_depth_perception()

        if self.__class__.__name__ == "UnitreeB2WMultiExpertTeacherEnvCfg":
            self.disable_zero_weight_rewards()

    def _add_depth_perception(self):
        """Add the 2 student depth cameras and the student's own (depth) observation groups.

        The teacher keeps reading the privileged ``policy`` group (proprio + height_scan); only the
        student gains a deployable view: proprio (no height scan) + 2 depth maps. Each camera becomes
        its own 4D image group so rsl_rl's ``CNNModel`` routes it to a conv encoder, and the velocity
        commands are split into their own 1D group so they can bypass the LSTM and re-enter at the
        head (Rudin et al. Fig. 3).
        """
        base_link_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # 1) two ray-cast depth cameras at the URDF mounts
        self.scene.depth_cam_front = _depth_camera_cfg(base_link_path, DEPTH_CAM_FRONT_POS, DEPTH_CAM_FRONT_RPY)
        self.scene.depth_cam_rear = _depth_camera_cfg(base_link_path, DEPTH_CAM_REAR_POS, DEPTH_CAM_REAR_RPY)

        # 2) proprio `student` group: deep-copy of `policy` minus the privileged height scan and the
        #    commands (commands move to their own group so they can bypass the LSTM)
        student = copy.deepcopy(self.observations.policy)
        student.height_scan = None
        student.velocity_commands = None
        self.observations.student = student

        # 3) `student_commands` group: the velocity command split out on its own (1D)
        student_commands = copy.deepcopy(self.observations.policy)
        for name in list(student_commands.__dict__.keys()):
            term = getattr(student_commands, name)
            if isinstance(term, ObsTerm) and name != "velocity_commands":
                setattr(student_commands, name, None)
        self.observations.student_commands = student_commands

        # 4) one image group per camera; a single term keeps the default concatenate_terms=True an
        #    identity so the group stays 4D (B, 1, H, W) for the CNN
        self.observations.depth_front = self._make_depth_group("depth_cam_front")
        self.observations.depth_rear = self._make_depth_group("depth_cam_rear")

    def _make_depth_group(self, sensor_name: str):
        """One image observation group holding a single channel-first depth term for ``sensor_name``."""
        group = copy.deepcopy(self.observations.policy)
        for name in list(group.__dict__.keys()):
            if isinstance(getattr(group, name), ObsTerm):
                setattr(group, name, None)
        group.enable_corruption = False
        group.depth = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg(sensor_name),
                "min_range": DEPTH_MIN_RANGE,
                "max_range": DEPTH_MAX_RANGE,
            },
        )
        return group