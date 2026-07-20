"""OmniSafe PPOLag/CPO references for the v2 Safety-Gymnasium domains."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import perf_counter

from experiments.frontier_v2_baseline_audit import (
    audit_baseline_manifest,
    file_sha256,
    select_checkpoint,
)
from experiments.frontier_v2_baseline_design import (
    BASELINE_SOURCE_LOCKS,
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_baseline_runner_hash import (
    runner_implementation_files,
    runner_implementation_sha256,
)
from experiments.frontier_v2_baseline_source_audit import (
    BASELINE_SOURCE_DIRECTORIES,
    audit_baseline_source,
)
from experiments.frontier_v2_external_adapters import (
    _activate_verified_source,
    bounded_score,
)
from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    all_tasks,
    canonical_episode_seed_base,
    domain_tasks,
    expected_episode_seeds,
    task_manifest_sha256,
)
from experiments.frontier_v2_source_audit import (
    SOURCE_DIRECTORIES,
    audit_codebase_source,
)


SUPPORTED_BASELINES = {
    baseline.identifier: baseline
    for baseline in COMPETITIVE_BASELINES
    if baseline.algorithm in {"PPO-Lagrangian", "constrained policy optimization"}
}
OMNISAFE_ALGORITHM = {
    "PPO-Lagrangian": "PPOLag",
    "constrained policy optimization": "CPO",
}
TRAINING_ENVIRONMENTS = {
    "safety_gymnasium_point_goal": "RiskShiftPointGoalDevelopment-v0",
    "safety_gymnasium_point_button": "RiskShiftPointButtonDevelopment-v0",
}
PADDED_OBSERVATION_SIZE = {
    "safety_gymnasium_point_goal": 60,
    "safety_gymnasium_point_button": 76,
}
STEPS_PER_EPOCH = 10_000
CHECKPOINT_EPOCH_INTERVAL = 5
_POOL_REGISTERED = False


def pooled_safety_task_index(training_seed: int, episode_index: int) -> int:
    if episode_index < 0:
        raise ValueError("episode index must be nonnegative")
    return (training_seed + episode_index) % 4


def expected_omnisafe_checkpoints(total_steps: int) -> tuple[tuple[int, int], ...]:
    if total_steps <= 0 or total_steps % (STEPS_PER_EPOCH * CHECKPOINT_EPOCH_INTERVAL):
        raise ValueError("OmniSafe budget must be a positive multiple of 50,000")
    return tuple(
        (step // STEPS_PER_EPOCH, step)
        for step in range(
            STEPS_PER_EPOCH * CHECKPOINT_EPOCH_INTERVAL,
            total_steps + 1,
            STEPS_PER_EPOCH * CHECKPOINT_EPOCH_INTERVAL,
        )
    )


def omnisafe_custom_config(
    algorithm: str,
    *,
    training_seed: int,
    total_steps: int,
    log_directory: Path,
    device: str,
) -> dict:
    if algorithm not in OMNISAFE_ALGORITHM:
        raise KeyError(algorithm)
    if total_steps % STEPS_PER_EPOCH:
        raise ValueError("OmniSafe total steps must be divisible by steps per epoch")
    config = {
        "seed": training_seed,
        "train_cfgs": {
            "device": device,
            "torch_threads": 4,
            "vector_env_nums": 1,
            "parallel": 1,
            "total_steps": total_steps,
        },
        "algo_cfgs": {
            "steps_per_epoch": STEPS_PER_EPOCH,
            "cost_limit": 25.0,
        },
        "logger_cfgs": {
            "use_wandb": False,
            "use_tensorboard": False,
            "save_model_freq": CHECKPOINT_EPOCH_INTERVAL,
            "log_dir": str(log_directory.resolve()),
        },
        "env_cfgs": {"training_seed": training_seed},
    }
    if algorithm == "PPO-Lagrangian":
        config["algo_cfgs"].pop("cost_limit")
        config["lagrange_cfgs"] = {"cost_limit": 25.0}
    return config


def installed_distributions() -> dict[str, str]:
    return dict(
        sorted(
            (
                str(distribution.metadata.get("Name", "unknown")).lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
        )
    )


def register_pooled_safety_environment() -> None:
    global _POOL_REGISTERED  # noqa: PLW0603
    if _POOL_REGISTERED:
        return
    import numpy as np
    import safety_gymnasium
    import torch
    from gymnasium import spaces
    from omnisafe.envs.core import CMDP, env_register

    @env_register
    class RiskShiftSafetyDevelopmentPool(CMDP):
        _support_envs = list(TRAINING_ENVIRONMENTS.values())
        need_auto_reset_wrapper = True
        need_time_limit_wrapper = False

        def __init__(self, env_id: str, **kwargs) -> None:
            inverse = {value: key for key, value in TRAINING_ENVIRONMENTS.items()}
            self._domain = inverse[env_id]
            self._tasks = domain_tasks(self._domain, "development")
            self._training_seed = int(kwargs["training_seed"])
            self._episode_index = -1
            self._steps = 0
            self._num_envs = 1
            self._environment_index = 0
            self._environment = safety_gymnasium.make(
                self._tasks[self._environment_index].environment_id
            )
            self._action_space = self._environment.action_space
            self._observation_size = PADDED_OBSERVATION_SIZE[self._domain]
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._observation_size,),
                dtype=np.float32,
            )
            for task in self._tasks[1:]:
                probe = safety_gymnasium.make(task.environment_id)
                try:
                    if (
                        probe.observation_space.shape[0] > self._observation_size
                        or probe.action_space.shape != self._action_space.shape
                    ):
                        raise RuntimeError("pooled Safety-Gymnasium spaces changed")
                finally:
                    probe.close()
                    del probe
            gc.collect()

        def _encode(self, observation):
            encoded = np.zeros(self._observation_size, dtype=np.float32)
            encoded[: len(observation)] = observation
            return torch.as_tensor(encoded, dtype=torch.float32)

        @property
        def max_episode_steps(self) -> int:
            return 500

        def reset(self, seed=None, options=None):
            del options
            if seed is not None and self._episode_index < 0:
                self._training_seed = int(seed)
            self._episode_index += 1
            index = pooled_safety_task_index(
                self._training_seed, self._episode_index
            )
            if index != self._environment_index:
                self._environment.close()
                del self._environment
                gc.collect()
                self._environment = safety_gymnasium.make(
                    self._tasks[index].environment_id
                )
                self._environment_index = index
            observation, info = self._environment.reset(
                seed=self._training_seed + self._episode_index
            )
            self._steps = 0
            return self._encode(observation), info

        def step(self, action):
            numeric_action = action.detach().cpu().numpy().reshape(-1)
            observation, reward, cost, terminated, truncated, info = (
                self._environment.step(numeric_action)
            )
            self._steps += 1
            truncated = bool(truncated or self._steps >= self.max_episode_steps)
            observation_tensor = self._encode(observation)
            info["final_observation"] = observation_tensor
            return (
                observation_tensor,
                torch.as_tensor(float(reward), dtype=torch.float32),
                torch.as_tensor(float(cost), dtype=torch.float32),
                torch.as_tensor(bool(terminated)),
                torch.as_tensor(truncated),
                info,
            )

        def set_seed(self, seed: int) -> None:
            self._training_seed = int(seed)

        def close(self) -> None:
            self._environment.close()

        def render(self):
            return self._environment.render()

    _POOL_REGISTERED = True


def _activate_sources(source_root: Path, baseline_source_root: Path):
    environment_source = source_root / SOURCE_DIRECTORIES["safety_gymnasium"]
    environment_audit = audit_codebase_source(
        environment_source, "safety_gymnasium"
    )
    _activate_verified_source(environment_source, "safety_gymnasium")
    baseline_source = baseline_source_root / BASELINE_SOURCE_DIRECTORIES["omnisafe"]
    baseline_audit = audit_baseline_source(baseline_source, "omnisafe")
    resolved = str(baseline_source.resolve())
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)
    register_pooled_safety_environment()
    return environment_audit, baseline_audit


def _pad_safety_observation(domain: str, observation):
    import numpy as np

    size = PADDED_OBSERVATION_SIZE[domain]
    values = np.asarray(observation, dtype=np.float32)
    if values.ndim != 1 or len(values) > size:
        raise RuntimeError("Safety-Gymnasium observation exceeds frozen padding")
    encoded = np.zeros(size, dtype=np.float32)
    encoded[: len(values)] = values
    return encoded


def _load_policy(checkpoint_path: Path, domain: str):
    import numpy as np
    import safety_gymnasium
    import torch
    from gymnasium import spaces
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor.actor_builder import ActorBuilder

    environment = safety_gymnasium.make(domain_tasks(domain, "calibration")[0].environment_id)
    try:
        action_space = environment.action_space
    finally:
        environment.close()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(PADDED_OBSERVATION_SIZE[domain],),
        dtype=np.float32,
    )
    actor = ActorBuilder(
        obs_space=observation_space,
        act_space=action_space,
        hidden_sizes=[64, 64],
        activation="tanh",
        weight_initialization_mode="kaiming_uniform",
    ).build_actor("gaussian_learning")
    actor.load_state_dict(checkpoint["pi"], strict=True)
    actor.eval()
    normalizer = Normalizer(shape=observation_space.shape, clip=5)
    normalizer.load_state_dict(checkpoint["obs_normalizer"], strict=True)
    normalizer.eval()
    return actor, normalizer


def _frozen_normalize(torch, observation, normalizer):
    tensor = torch.as_tensor(observation, dtype=torch.float32)
    if int(normalizer.state_dict()["_count"].item()) <= 1:
        return tensor
    normalized = (tensor - normalizer.mean) / normalizer.std
    return torch.clamp(normalized, -5.0, 5.0)


def evaluate_omnisafe_checkpoint(
    checkpoint_path: Path,
    domain: str,
    *,
    episodes_per_task: int,
) -> dict:
    import safety_gymnasium
    import torch

    actor, normalizer = _load_policy(checkpoint_path, domain)
    task_scores = []
    task_costs = []
    for task in domain_tasks(domain, "calibration"):
        parameters = task.parameter_dict()
        cost_weight = float(parameters["cost_weight"])
        max_steps = int(parameters["max_steps"])
        scores = []
        costs = []
        for seed in expected_episode_seeds(
            task,
            episodes=episodes_per_task,
            seed_base=canonical_episode_seed_base(task),
        ):
            environment = safety_gymnasium.make(task.environment_id)
            try:
                observation, _info = environment.reset(seed=seed)
                raw_return = 0.0
                total_cost = 0.0
                successes = 0
                steps = 0
                terminated = truncated = False
                while not (terminated or truncated) and steps < max_steps:
                    normalized = _frozen_normalize(
                        torch,
                        _pad_safety_observation(domain, observation),
                        normalizer,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action = (
                            actor.predict(normalized, deterministic=True)
                            .reshape(-1)
                            .cpu()
                            .numpy()
                        )
                    observation, reward, cost, terminated, truncated, info = (
                        environment.step(action)
                    )
                    raw_return += float(reward)
                    total_cost += float(cost)
                    successes += int(bool(info.get("goal_met", False)))
                    steps += 1
                failure = successes == 0
                raw_utility = (
                    raw_return
                    + 5.0 * successes
                    - cost_weight * total_cost
                    - 2.0 * failure
                )
                scores.append(
                    bounded_score(0.5 + 0.5 * math.tanh(raw_utility / 25.0))
                )
                costs.append(total_cost)
            finally:
                environment.close()
                del environment
                gc.collect()
        task_scores.append(fmean(scores))
        task_costs.append(fmean(costs))
    return {
        "calibration_equal_task_mean_score": fmean(task_scores),
        "calibration_equal_task_mean_cost": fmean(task_costs),
        "calibration_task_mean_scores": task_scores,
        "calibration_task_mean_costs": task_costs,
        "calibration_episodes_per_task": episodes_per_task,
    }


def train_omnisafe_seed(
    baseline,
    *,
    training_seed: int,
    output_root: Path,
    total_steps: int,
    evaluation_episodes_per_task: int,
    device: str,
) -> dict:
    import omnisafe

    algorithm = OMNISAFE_ALGORITHM[baseline.algorithm]
    raw_log_root = (
        output_root
        / "_omnisafe_runs"
        / baseline.domain
        / baseline.name
        / f"seed-{training_seed}"
    )
    config = omnisafe_custom_config(
        baseline.algorithm,
        training_seed=training_seed,
        total_steps=total_steps,
        log_directory=raw_log_root,
        device=device,
    )
    started = perf_counter()
    agent = omnisafe.Agent(
        algorithm,
        TRAINING_ENVIRONMENTS[baseline.domain],
        custom_cfgs=config,
    )
    try:
        agent.learn()
    finally:
        agent.agent._env.close()
        gc.collect()
    log_directory = Path(agent.agent.logger.log_dir)
    checkpoints = []
    for epoch, step in expected_omnisafe_checkpoints(total_steps):
        source_path = log_directory / "torch_save" / f"epoch-{epoch}.pt"
        if not source_path.is_file():
            raise RuntimeError(f"expected OmniSafe checkpoint is missing: {source_path}")
        relative = (
            Path(baseline.domain)
            / baseline.name
            / f"seed-{training_seed}"
            / f"step-{step}.pt"
        )
        checkpoint_path = output_root / relative
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, checkpoint_path)
        checkpoints.append(
            {
                "step": step,
                "checkpoint_path": relative.as_posix(),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                **evaluate_omnisafe_checkpoint(
                    checkpoint_path,
                    baseline.domain,
                    episodes_per_task=evaluation_episodes_per_task,
                ),
            }
        )
    selected = select_checkpoint(
        checkpoints, cost_limit=baseline.safety_cost_limit
    )
    return {
        "training_seed": training_seed,
        "training_steps": total_steps,
        "training_episodes": None,
        "runtime_seconds": perf_counter() - started,
        "omnisafe_log_directory": log_directory.resolve()
        .relative_to(output_root.resolve())
        .as_posix(),
        "checkpoints": checkpoints,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }


def audit_selected_checkpoint_replay(payload: dict, *, checkpoint_root: Path) -> dict:
    domain = str(payload["baseline_identifier"]).split(":", 1)[0]
    records = []
    for run in payload["runs"]:
        selected_sha256 = run["selected_checkpoint_sha256"]
        matches = [
            checkpoint
            for checkpoint in run["checkpoints"]
            if checkpoint["checkpoint_sha256"] == selected_sha256
        ]
        if len(matches) != 1:
            raise RuntimeError("selected OmniSafe checkpoint is not unique")
        expected = matches[0]
        replay = evaluate_omnisafe_checkpoint(
            checkpoint_root / expected["checkpoint_path"],
            domain,
            episodes_per_task=int(expected["calibration_episodes_per_task"]),
        )
        for field in (
            "calibration_equal_task_mean_score",
            "calibration_equal_task_mean_cost",
        ):
            if not math.isclose(
                float(replay[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"selected OmniSafe replay changed {field}")
        records.append(
            {
                "training_seed": int(run["training_seed"]),
                "selected_step": int(expected["step"]),
                "selected_checkpoint_sha256": selected_sha256,
                "calibration_equal_task_mean_score": replay[
                    "calibration_equal_task_mean_score"
                ],
                "calibration_equal_task_mean_cost": replay[
                    "calibration_equal_task_mean_cost"
                ],
            }
        )
    return {
        "design": "riskshiftbench-frontier-v2-omnisafe-selected-replay-audit-v1",
        "checkpoint_count": len(records),
        "calibration_replay_exact": True,
        "records": records,
    }


def train_omnisafe_baseline(
    baseline_identifier: str,
    *,
    output_root: Path,
    source_root: Path,
    baseline_source_root: Path,
    device: str,
) -> dict:
    try:
        baseline = SUPPORTED_BASELINES[baseline_identifier]
    except KeyError as error:
        raise KeyError(f"unknown OmniSafe baseline: {baseline_identifier}") from error
    environment_audit, implementation_audit = _activate_sources(
        source_root, baseline_source_root
    )
    environment_audit_payload = asdict(environment_audit)
    environment_audit_payload["source"] = SOURCE_DIRECTORIES["safety_gymnasium"]
    implementation_audit_payload = asdict(implementation_audit)
    implementation_audit_payload["source"] = BASELINE_SOURCE_DIRECTORIES["omnisafe"]
    design = baseline_design_summary()
    runs = [
        train_omnisafe_seed(
            baseline,
            training_seed=seed,
            output_root=output_root,
            total_steps=baseline.training_steps_per_seed,
            evaluation_episodes_per_task=100,
            device=device,
        )
        for seed in baseline.training_seeds
    ]
    import omnisafe
    import torch

    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-baseline-checkpoints-v1",
        "baseline_design_sha256": design["design_sha256"],
        "baseline_implementation_sha256": design["internal_implementation_sha256"],
        "runner_implementation_files": list(
            runner_implementation_files(baseline.algorithm)
        ),
        "runner_implementation_sha256": runner_implementation_sha256(
            baseline.algorithm
        ),
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "development_manifest_sha256": task_manifest_sha256(
            all_tasks("development")
        ),
        "calibration_manifest_sha256": task_manifest_sha256(
            all_tasks("calibration")
        ),
        "source_lock": asdict(BASELINE_SOURCE_LOCKS["omnisafe"]),
        "baseline_source_audit": implementation_audit_payload,
        "environment_codebase_lock": asdict(CODEBASE_LOCKS["safety_gymnasium"]),
        "environment_source_audit": environment_audit_payload,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "omnisafe": omnisafe.__version__,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "installed_distributions": installed_distributions(),
        },
        "algorithm_hyperparameters": {
            "upstream_algorithm": OMNISAFE_ALGORITHM[baseline.algorithm],
            "pooled_environment": TRAINING_ENVIRONMENTS[baseline.domain],
            "steps_per_epoch": STEPS_PER_EPOCH,
            "checkpoint_epoch_interval": CHECKPOINT_EPOCH_INTERVAL,
            "cost_limit": 25.0,
            "actor_hidden_sizes": [64, 64],
            "actor_activation": "tanh",
            "observation_normalization": "training running moments; frozen at evaluation",
            "vector_env_nums": 1,
            "task_sampling": "episode-stratified pooled development round robin",
            "maximum_episode_steps": 500,
        },
        "runs": runs,
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root
    )
    manifest_path = output_root / baseline.domain / baseline.name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit_baseline_manifest(payload, checkpoint_root=output_root)
    return payload


def smoke_omnisafe(
    baseline_identifier: str,
    *,
    output_root: Path,
    source_root: Path,
    baseline_source_root: Path,
    device: str,
) -> dict:
    try:
        baseline = SUPPORTED_BASELINES[baseline_identifier]
    except KeyError as error:
        raise KeyError(baseline_identifier) from error
    _activate_sources(source_root, baseline_source_root)
    seed = baseline.training_seeds[0]
    run = train_omnisafe_seed(
        baseline,
        training_seed=seed,
        output_root=output_root,
        total_steps=50_000,
        evaluation_episodes_per_task=2,
        device=device,
    )
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-omnisafe-smoke-v1",
        "nonconfirmatory": True,
        "baseline_identifier": baseline.identifier,
        "steps": 50_000,
        "run": run,
        "runs": [run],
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", choices=tuple(SUPPORTED_BASELINES), required=True
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baselines"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--baseline-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baseline_sources"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        payload = smoke_omnisafe(
            args.baseline,
            output_root=args.output_root,
            source_root=args.source_root,
            baseline_source_root=args.baseline_source_root,
            device=args.device,
        )
    else:
        payload = train_omnisafe_baseline(
            args.baseline,
            output_root=args.output_root,
            source_root=args.source_root,
            baseline_source_root=args.baseline_source_root,
            device=args.device,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
