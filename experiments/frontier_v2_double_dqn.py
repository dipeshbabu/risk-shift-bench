"""Pooled-development Double-DQN references for v2 discrete-action domains.

The network and replay-loop structure follows the pinned CleanRL DQN reference.
The Double-DQN action-selection target, pooled task sampler, observation adapters,
checkpoint schedule, and calibration evaluator are local and hash-bound here.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict, dataclass
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
    DOMAIN_SPECS,
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
    baseline.domain: baseline
    for baseline in COMPETITIVE_BASELINES
    if baseline.algorithm == "double DQN"
}
MAX_IMAGE_SIZE = {
    "minigrid_dynamic_obstacles": 16,
    "minigrid_lava_crossing": 11,
}
ACTION_COUNT = {
    "minigrid_dynamic_obstacles": 3,
    "minigrid_lava_crossing": 7,
    "or_gym_online_knapsack": 2,
}


@dataclass
class EpisodeContext:
    domain: str
    task: object
    environment: object
    steps: int = 0
    raw_return: float = 0.0
    fractional_upper_bound: float | None = None


def epsilon_at_step(step: int, total_steps: int) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("step and total_steps must define a nonnegative schedule")
    progress = min(1.0, step / max(1.0, 0.5 * total_steps))
    return 1.0 + progress * (0.05 - 1.0)


def pooled_task_index(training_seed: int, episode_index: int, task_count: int) -> int:
    if episode_index < 0 or task_count <= 0:
        raise ValueError("episode index and task count are invalid")
    return (training_seed + episode_index) % task_count


def _torch_modules():
    import numpy as np
    import torch

    return np, torch


def _build_q_network(torch, input_size: int, action_count: int):
    class QNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_count),
            )

        def forward(self, observation):
            return self.network(observation)

    return QNetwork()


def _minigrid_kwargs(task) -> dict[str, object]:
    parameters = task.parameter_dict()
    if task.domain == "minigrid_dynamic_obstacles":
        return {
            "size": int(parameters["size"]),
            "n_obstacles": int(parameters["n_obstacles"]),
            "agent_start_pos": None
            if bool(parameters["agent_start_random"])
            else (1, 1),
            "max_steps": int(parameters["max_steps"]),
        }
    if task.domain == "minigrid_lava_crossing":
        return {
            "size": int(parameters["size"]),
            "num_crossings": int(parameters["num_crossings"]),
            "max_steps": int(parameters["max_steps"]),
        }
    raise KeyError(task.domain)


def _start_episode(domain: str, task, seed: int):
    np, _torch = _torch_modules()
    if domain.startswith("minigrid_"):
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import FullyObsWrapper

        environment = FullyObsWrapper(
            gym.make(task.environment_id, **_minigrid_kwargs(task))
        )
        observation, _info = environment.reset(seed=seed)
        context = EpisodeContext(domain=domain, task=task, environment=environment)
        return context, observation

    if domain == "or_gym_online_knapsack":
        from experiments.external_domain_adapters import _knapsack_catalog
        from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv

        parameters = task.parameter_dict()
        weights, values = _knapsack_catalog(task)
        np.random.seed(seed % (2**32))
        environment = OnlineKnapsackEnv()
        environment.max_weight = int(parameters["capacity"])
        environment.step_limit = int(parameters["horizon"])
        environment.item_weights = np.asarray(weights, dtype=np.int32)
        environment.item_values = np.asarray(values, dtype=np.int32)
        environment.item_limits_init = np.ones(len(weights), dtype=np.int32)
        environment.item_probs = (
            environment.item_limits_init / environment.item_limits_init.sum()
        )
        np.random.seed(seed % (2**32))
        observation = environment.reset()
        upper = int(parameters["capacity"]) * max(
            value / weight for weight, value in zip(weights, values, strict=True)
        )
        context = EpisodeContext(
            domain=domain,
            task=task,
            environment=environment,
            fractional_upper_bound=upper,
        )
        return context, observation
    raise KeyError(domain)


def _encoded_observation(context: EpisodeContext, observation):
    np, _torch = _torch_modules()
    if context.domain.startswith("minigrid_"):
        image = np.asarray(observation["image"], dtype=np.float32)
        size = MAX_IMAGE_SIZE[context.domain]
        if image.ndim != 3 or image.shape[2] != 3 or max(image.shape[:2]) > size:
            raise RuntimeError("MiniGrid image is incompatible with the frozen encoder")
        canvas = np.zeros((size, size, 3), dtype=np.float32)
        canvas[: image.shape[0], : image.shape[1], :] = image / 10.0
        return canvas.reshape(-1)

    state = observation["state"]
    parameters = context.task.parameter_dict()
    capacity = float(parameters["capacity"])
    horizon = float(parameters["horizon"])
    catalog_size = float(parameters["catalog_size"])
    maximum_value = float(context.environment.item_values.max())
    return np.asarray(
        (
            float(state[0]) / capacity,
            float(state[1]) / max(1.0, catalog_size - 1.0),
            float(state[2]) / capacity,
            float(state[3]) / max(1.0, maximum_value),
            context.steps / horizon,
            capacity / 400.0,
        ),
        dtype=np.float32,
    )


def _action_mask(context: EpisodeContext, observation):
    np, _torch = _torch_modules()
    if context.domain.startswith("minigrid_"):
        return np.ones(int(context.environment.action_space.n), dtype=np.bool_)
    return np.asarray(observation["action_mask"], dtype=np.bool_)


def _step_episode(context: EpisodeContext, action: int):
    if context.domain.startswith("minigrid_"):
        observation, reward, terminated, truncated, _info = context.environment.step(
            action
        )
        done = bool(terminated or truncated)
    else:
        observation, reward, done, _info = context.environment.step(action)
        done = bool(done)
    context.steps += 1
    context.raw_return += float(reward)
    return observation, float(reward), done


def _episode_metrics(context: EpisodeContext) -> tuple[float, float]:
    if context.domain.startswith("minigrid_"):
        return bounded_score(context.raw_return), float(context.raw_return < 0.0)
    assert context.fractional_upper_bound is not None
    horizon = int(context.task.parameter_dict()["horizon"])
    early_exhaustion = context.steps < horizon
    return (
        bounded_score(context.raw_return / context.fractional_upper_bound),
        float(early_exhaustion),
    )


def _training_reward(context: EpisodeContext, reward: float) -> float:
    if context.domain.startswith("minigrid_"):
        return reward
    assert context.fractional_upper_bound is not None
    return reward / context.fractional_upper_bound


def _greedy_action(torch, network, observation, mask, device) -> int:
    with torch.no_grad():
        values = network(
            torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        )[0]
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
        values = values.masked_fill(~valid, -torch.inf)
        return int(values.argmax().item())


def _evaluate_network(
    domain: str,
    network,
    *,
    episodes_per_task: int,
    device,
) -> dict:
    _np, torch = _torch_modules()
    task_scores = []
    task_costs = []
    network.eval()
    for task in domain_tasks(domain, "calibration"):
        scores = []
        costs = []
        for seed in expected_episode_seeds(
            task,
            episodes=episodes_per_task,
            seed_base=canonical_episode_seed_base(task),
        ):
            context, raw_observation = _start_episode(domain, task, seed)
            try:
                done = False
                while not done:
                    observation = _encoded_observation(context, raw_observation)
                    mask = _action_mask(context, raw_observation)
                    action = _greedy_action(
                        torch, network, observation, mask, device
                    )
                    raw_observation, _reward, done = _step_episode(context, action)
                score, cost = _episode_metrics(context)
                scores.append(score)
                costs.append(cost)
            finally:
                context.environment.close()
        task_scores.append(fmean(scores))
        task_costs.append(fmean(costs))
    network.train()
    return {
        "calibration_equal_task_mean_score": fmean(task_scores),
        "calibration_equal_task_mean_cost": fmean(task_costs),
        "calibration_task_mean_scores": task_scores,
        "calibration_task_mean_costs": task_costs,
        "calibration_episodes_per_task": episodes_per_task,
    }


def _input_and_action_sizes(domain: str) -> tuple[int, int]:
    if domain.startswith("minigrid_"):
        return MAX_IMAGE_SIZE[domain] ** 2 * 3, ACTION_COUNT[domain]
    if domain == "or_gym_online_knapsack":
        return 6, ACTION_COUNT[domain]
    raise KeyError(domain)


def _portable_state_dict(network) -> dict:
    return {
        name: value.detach().cpu() for name, value in network.state_dict().items()
    }


def audit_selected_checkpoint_replay(payload: dict, *, checkpoint_root: Path) -> dict:
    """Reload and deterministically reevaluate every selected seed checkpoint."""

    _np, torch = _torch_modules()
    domain = str(payload["baseline_identifier"]).split(":", 1)[0]
    input_size, action_count = _input_and_action_sizes(domain)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for run in payload["runs"]:
        selected_sha256 = run["selected_checkpoint_sha256"]
        matches = [
            checkpoint
            for checkpoint in run["checkpoints"]
            if checkpoint["checkpoint_sha256"] == selected_sha256
        ]
        if len(matches) != 1:
            raise RuntimeError("selected Double-DQN checkpoint is not unique")
        expected = matches[0]
        path = checkpoint_root / expected["checkpoint_path"]
        checkpoint_payload = torch.load(path, map_location=device, weights_only=True)
        required = {
            "protocol_id": "riskshiftbench-frontier-v2-double-dqn-checkpoint-v1",
            "domain": domain,
            "training_seed": int(run["training_seed"]),
            "step": int(expected["step"]),
            "input_size": input_size,
            "action_count": action_count,
        }
        if any(checkpoint_payload.get(key) != value for key, value in required.items()):
            raise RuntimeError("selected Double-DQN checkpoint metadata changed")
        network = _build_q_network(torch, input_size, action_count).to(device)
        network.load_state_dict(checkpoint_payload["online_state_dict"], strict=True)
        replay = _evaluate_network(
            domain,
            network,
            episodes_per_task=int(expected["calibration_episodes_per_task"]),
            device=device,
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
                raise RuntimeError(f"selected checkpoint replay changed {field}")
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
        "design": "riskshiftbench-frontier-v2-double-dqn-selected-replay-audit-v1",
        "checkpoint_count": len(records),
        "calibration_replay_exact": True,
        "records": records,
    }


def train_double_dqn_seed(
    domain: str,
    *,
    training_seed: int,
    output_root: Path,
    total_steps: int,
    checkpoint_interval_steps: int,
    evaluation_episodes_per_task: int,
) -> dict:
    np, torch = _torch_modules()
    if total_steps <= 0 or checkpoint_interval_steps <= 0:
        raise ValueError("training and checkpoint steps must be positive")
    if total_steps % checkpoint_interval_steps:
        raise ValueError("training steps must be divisible by the checkpoint interval")

    torch.manual_seed(training_seed)
    np.random.seed(training_seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size, action_count = _input_and_action_sizes(domain)
    online = _build_q_network(torch, input_size, action_count).to(device)
    target = _build_q_network(torch, input_size, action_count).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=2.5e-4)

    buffer_size = 50_000
    observations = np.zeros((buffer_size, input_size), dtype=np.float32)
    next_observations = np.zeros((buffer_size, input_size), dtype=np.float32)
    actions = np.zeros(buffer_size, dtype=np.int64)
    rewards = np.zeros(buffer_size, dtype=np.float32)
    dones = np.zeros(buffer_size, dtype=np.float32)
    next_masks = np.zeros((buffer_size, action_count), dtype=np.bool_)
    rng = np.random.default_rng(training_seed)
    tasks = domain_tasks(domain, "development")
    episode_index = 0
    context, raw_observation = _start_episode(
        domain,
        tasks[pooled_task_index(training_seed, episode_index, len(tasks))],
        training_seed + episode_index,
    )
    checkpoints = []
    started = perf_counter()
    try:
        for step in range(1, total_steps + 1):
            observation = _encoded_observation(context, raw_observation)
            mask = _action_mask(context, raw_observation)
            valid_actions = np.flatnonzero(mask)
            if rng.random() < epsilon_at_step(step - 1, total_steps):
                action = int(valid_actions[int(rng.integers(len(valid_actions)))])
            else:
                action = _greedy_action(torch, online, observation, mask, device)
            next_raw_observation, raw_reward, done = _step_episode(context, action)
            next_observation = _encoded_observation(context, next_raw_observation)
            next_mask = _action_mask(context, next_raw_observation)

            slot = (step - 1) % buffer_size
            observations[slot] = observation
            next_observations[slot] = next_observation
            actions[slot] = action
            rewards[slot] = _training_reward(context, raw_reward)
            dones[slot] = float(done)
            next_masks[slot] = next_mask

            if step >= 10_000 and step % 4 == 0:
                population = min(step, buffer_size)
                indices = rng.integers(population, size=128)
                obs_batch = torch.as_tensor(
                    observations[indices], dtype=torch.float32, device=device
                )
                next_obs_batch = torch.as_tensor(
                    next_observations[indices], dtype=torch.float32, device=device
                )
                action_batch = torch.as_tensor(
                    actions[indices], dtype=torch.int64, device=device
                ).unsqueeze(1)
                reward_batch = torch.as_tensor(
                    rewards[indices], dtype=torch.float32, device=device
                )
                done_batch = torch.as_tensor(
                    dones[indices], dtype=torch.float32, device=device
                )
                mask_batch = torch.as_tensor(
                    next_masks[indices], dtype=torch.bool, device=device
                )
                with torch.no_grad():
                    online_next = online(next_obs_batch).masked_fill(
                        ~mask_batch, -torch.inf
                    )
                    next_actions = online_next.argmax(dim=1, keepdim=True)
                    next_values = target(next_obs_batch).gather(1, next_actions).squeeze(1)
                    targets = reward_batch + 0.99 * (1.0 - done_batch) * next_values
                predictions = online(obs_batch).gather(1, action_batch).squeeze(1)
                loss = torch.nn.functional.mse_loss(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if step % 1_000 == 0:
                target.load_state_dict(online.state_dict())

            if done:
                context.environment.close()
                episode_index += 1
                task = tasks[
                    pooled_task_index(training_seed, episode_index, len(tasks))
                ]
                context, raw_observation = _start_episode(
                    domain, task, training_seed + episode_index
                )
            else:
                raw_observation = next_raw_observation

            if step % checkpoint_interval_steps == 0:
                evaluation = _evaluate_network(
                    domain,
                    online,
                    episodes_per_task=evaluation_episodes_per_task,
                    device=device,
                )
                relative = Path(domain) / SUPPORTED_BASELINES[domain].name / (
                    f"seed-{training_seed}"
                ) / f"step-{step}.pt"
                checkpoint_path = output_root / relative
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "protocol_id": "riskshiftbench-frontier-v2-double-dqn-checkpoint-v1",
                        "domain": domain,
                        "training_seed": training_seed,
                        "step": step,
                        "input_size": input_size,
                        "action_count": action_count,
                        "online_state_dict": _portable_state_dict(online),
                    },
                    checkpoint_path,
                )
                checkpoints.append(
                    {
                        "step": step,
                        "checkpoint_path": relative.as_posix(),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                        **evaluation,
                    }
                )
    finally:
        context.environment.close()

    selected = select_checkpoint(checkpoints, cost_limit=None)
    return {
        "training_seed": training_seed,
        "training_steps": total_steps,
        "training_episodes": episode_index,
        "runtime_seconds": perf_counter() - started,
        "checkpoints": checkpoints,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }


def _activate_domain_source(domain: str, source_root: Path):
    codebase = DOMAIN_SPECS[domain].codebase
    source = source_root / SOURCE_DIRECTORIES[codebase]
    audit = audit_codebase_source(source, codebase)
    _activate_verified_source(source, codebase)
    return codebase, audit


def train_double_dqn_baseline(
    domain: str,
    *,
    output_root: Path,
    source_root: Path,
    baseline_source_root: Path,
) -> dict:
    try:
        baseline = SUPPORTED_BASELINES[domain]
    except KeyError as error:
        raise KeyError(f"no Double-DQN baseline for {domain}") from error
    codebase, environment_audit = _activate_domain_source(domain, source_root)
    baseline_source = (
        baseline_source_root / BASELINE_SOURCE_DIRECTORIES[baseline.implementation_source]
    )
    implementation_audit = audit_baseline_source(
        baseline_source, baseline.implementation_source
    )
    implementation_audit_payload = asdict(implementation_audit)
    implementation_audit_payload["source"] = BASELINE_SOURCE_DIRECTORIES[
        baseline.implementation_source
    ]
    environment_audit_payload = asdict(environment_audit)
    environment_audit_payload["source"] = SOURCE_DIRECTORIES[codebase]
    design = baseline_design_summary()
    runs = [
        train_double_dqn_seed(
            domain,
            training_seed=seed,
            output_root=output_root,
            total_steps=baseline.training_steps_per_seed,
            checkpoint_interval_steps=baseline.checkpoint_interval_steps,
            evaluation_episodes_per_task=100,
        )
        for seed in baseline.training_seeds
    ]
    _np, torch = _torch_modules()
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
        "source_lock": asdict(BASELINE_SOURCE_LOCKS[baseline.implementation_source]),
        "baseline_source_audit": implementation_audit_payload,
        "environment_codebase_lock": asdict(CODEBASE_LOCKS[codebase]),
        "environment_source_audit": environment_audit_payload,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        },
        "algorithm_hyperparameters": {
            "architecture": "MLP(256,256)",
            "optimizer": "Adam",
            "learning_rate": 2.5e-4,
            "discount": 0.99,
            "replay_buffer_size": 50_000,
            "batch_size": 128,
            "learning_starts": 10_000,
            "train_frequency": 4,
            "target_network_frequency": 1_000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_fraction": 0.5,
            "task_sampling": "episode-stratified pooled development round robin",
            "target_rule": "online argmax with target-network evaluation (Double DQN)",
        },
        "runs": runs,
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root
    )
    manifest_path = output_root / domain / baseline.name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit_baseline_manifest(payload, checkpoint_root=output_root)
    return payload


def smoke_double_dqn(
    domain: str,
    *,
    steps: int,
    output_root: Path,
    source_root: Path,
) -> dict:
    if domain not in SUPPORTED_BASELINES:
        raise KeyError(domain)
    _activate_domain_source(domain, source_root)
    seed = SUPPORTED_BASELINES[domain].training_seeds[0]
    run = train_double_dqn_seed(
        domain,
        training_seed=seed,
        output_root=output_root,
        total_steps=steps,
        checkpoint_interval_steps=steps,
        evaluation_episodes_per_task=2,
    )
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-double-dqn-smoke-v1",
        "nonconfirmatory": True,
        "domain": domain,
        "baseline_identifier": SUPPORTED_BASELINES[domain].identifier,
        "steps": steps,
        "run": run,
        "runs": [run],
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(SUPPORTED_BASELINES), required=True)
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
    parser.add_argument("--smoke-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_steps is not None:
        payload = smoke_double_dqn(
            args.domain,
            steps=args.smoke_steps,
            output_root=args.output_root,
            source_root=args.source_root,
        )
    else:
        payload = train_double_dqn_baseline(
            args.domain,
            output_root=args.output_root,
            source_root=args.source_root,
            baseline_source_root=args.baseline_source_root,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
