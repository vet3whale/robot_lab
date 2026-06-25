# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Multi-expert distillation: N frozen teachers, each env supervised by the expert matching its terrain.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class MultiTeacherDistillation(Distillation):
    """Distillation with N frozen teachers; each env is supervised by the expert matching its terrain."""

    def __init__(self, student, teachers, storage, env, column_to_expert, **kwargs):
        # base sets self.student, self.teacher (= teachers[0]), optimizer (student only), storage, loss_fn
        super().__init__(student, teachers[0], storage, **kwargs)
        self.teachers = nn.ModuleList(teachers).to(self.device)

        # per-env expert id from the static terrain column assignment
        terrain = env.unwrapped.scene.terrain
        col2exp = torch.as_tensor(column_to_expert, dtype=torch.long, device=self.device)
        self.expert_ids = col2exp[terrain.terrain_types.to(self.device)]  # [num_envs]
        self.teacher_loaded = True

        counts = torch.bincount(self.expert_ids, minlength=len(self.teachers))
        print(f"[MultiTeacherDistillation] {len(self.teachers)} experts, envs per expert: {counts.tolist()}")

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        targets = torch.stack([t(obs) for t in self.teachers], dim=0)  # [E, N, A]
        idx = self.expert_ids.view(1, -1, 1).expand(1, -1, targets.shape[-1])  # [1, N, A]
        self.transition.privileged_actions = targets.gather(0, idx).squeeze(0).detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.student.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        for t in self.teachers:
            t.reset(dones)

    def train_mode(self) -> None:
        self.student.train()
        for t in self.teachers:
            t.eval()

    def eval_mode(self) -> None:
        self.student.eval()
        for t in self.teachers:
            t.eval()

    def save(self) -> dict:
        saved = {
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # keep a 'teacher_state_dict' so the stock single-teacher loader / play.py stays happy
            "teacher_state_dict": self.teachers[0].state_dict(),
        }
        for i, t in enumerate(self.teachers):
            saved[f"teacher_{i}_state_dict"] = t.state_dict()
        return saved

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "MultiTeacherDistillation":
        student_class = resolve_callable(cfg["student"].pop("class_name"))
        teacher_class = resolve_callable(cfg["teacher"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["student", "teacher"])
        cfg["algorithm"]["rnd_cfg"] = None
        cfg["algorithm"]["symmetry_cfg"] = None
        cfg["algorithm"].pop("class_name", None)

        env_cfg = env.unwrapped.cfg
        ckpts: list[str] = env_cfg.teacher_checkpoints
        column_to_expert: list[int] = env_cfg.column_to_expert
        if not ckpts:
            raise ValueError("No teacher checkpoints found on the env cfg; check teachers.txt.")

        student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(device)
        teachers = []
        for path in ckpts:
            # deep-copy: the model ctor pops nested keys (e.g. distribution_cfg.class_name),
            # which would mutate the shared cfg["teacher"] and break the next teacher build
            teacher_kwargs = copy.deepcopy(cfg["teacher"])
            t = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **teacher_kwargs).to(device)
            sd = torch.load(path, map_location=device, weights_only=False)
            t.load_state_dict(sd.get("teacher_state_dict") or sd["actor_state_dict"], strict=True)
            t.eval()
            teachers.append(t)
        print(f"Student: {student}\nLoaded {len(teachers)} teachers from {ckpts}")

        storage = RolloutStorage(
            "distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return MultiTeacherDistillation(
            student,
            teachers,
            storage,
            env,
            column_to_expert,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )