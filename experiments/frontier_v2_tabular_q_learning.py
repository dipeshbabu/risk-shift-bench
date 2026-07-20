"""Train the frozen tabular Q-learning competitive references for v2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import perf_counter

from experiments.frontier_v2_baseline_audit import file_sha256, select_checkpoint
from experiments.frontier_v2_baseline_design import (
    COMPETITIVE_BASELINES,
    baseline_design_summary,
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
    baseline.domain: baseline
    for baseline in COMPETITIVE_BASELINES
    if baseline.algorithm == "tabular Q-learning"
}


def _make_environment(task):
    import gymnasium as gym

    parameters = task.parameter_dict()
    if task.domain == "gymnasium_cliffwalking":
        kwargs = {"is_slippery": bool(parameters["is_slippery"])}
    elif task.domain == "gymnasium_taxi":
        kwargs = {
            "is_rainy": bool(parameters["is_rainy"]),
            "rainy_probability": float(parameters["rainy_probability"]),
            "fickle_passenger": bool(parameters["fickle_passenger"]),
            "fickle_probability": float(parameters["fickle_probability"]),
        }
    else:
        raise KeyError(task.domain)
    return gym.make(
        task.environment_id,
        **kwargs,
        max_episode_steps=int(parameters["max_steps"]),
    )


def _greedy_action(q_values, state: int) -> int:
    row = q_values[state]
    maximum = float(row.max())
    return min(index for index, value in enumerate(row) if float(value) == maximum)


def _training_action(q_values, state: int, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[1]))
    row = q_values[state]
    candidates = [
        index for index, value in enumerate(row) if float(value) == float(row.max())
    ]
    return int(candidates[int(rng.integers(len(candidates)))])


def _episode_score(task, *, success: bool, steps: int, cost: float) -> float:
    max_steps = int(task.parameter_dict()["max_steps"])
    return bounded_score(
        float(success) * (1.0 - 0.5 * steps / max_steps) - 0.1 * cost
    )


def evaluate_q_table(
    domain: str,
    q_values,
    *,
    episodes_per_task: int,
) -> dict:
    tasks = domain_tasks(domain, "calibration")
    task_scores = []
    task_costs = []
    for task in tasks:
        environment = _make_environment(task)
        scores = []
        costs = []
        try:
            seed_base = canonical_episode_seed_base(task)
            seeds = expected_episode_seeds(
                task,
                episodes=episodes_per_task,
                seed_base=seed_base,
            )
            for seed in seeds:
                state, _info = environment.reset(seed=seed)
                success = False
                cost = 0.0
                steps = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = _greedy_action(q_values, int(state))
                    state, reward, terminated, truncated, _info = environment.step(
                        action
                    )
                    numeric_reward = float(reward)
                    cost += float(
                        (domain == "gymnasium_cliffwalking" and numeric_reward <= -100.0)
                        or (domain == "gymnasium_taxi" and numeric_reward <= -10.0)
                    )
                    success = success or (
                        (domain == "gymnasium_cliffwalking" and bool(terminated))
                        or (domain == "gymnasium_taxi" and numeric_reward >= 20.0)
                    )
                    steps += 1
                scores.append(
                    _episode_score(task, success=success, steps=steps, cost=cost)
                )
                costs.append(cost)
        finally:
            environment.close()
        task_scores.append(fmean(scores))
        task_costs.append(fmean(costs))
    return {
        "calibration_equal_task_mean_score": fmean(task_scores),
        "calibration_equal_task_mean_cost": fmean(task_costs),
        "calibration_task_mean_scores": task_scores,
        "calibration_task_mean_costs": task_costs,
        "calibration_episodes_per_task": episodes_per_task,
    }


def train_q_learning_seed(
    domain: str,
    *,
    training_seed: int,
    output_root: Path,
    source_root: Path,
    evaluation_episodes_per_task: int = 100,
) -> dict:
    import numpy as np

    baseline = SUPPORTED_BASELINES[domain]
    source = source_root / SOURCE_DIRECTORIES["gymnasium"]
    _activate_verified_source(source, "gymnasium")
    tasks = domain_tasks(domain, "development")
    environments = [_make_environment(task) for task in tasks]
    state_count = int(environments[0].observation_space.n)
    action_count = int(environments[0].action_space.n)
    if any(
        int(environment.observation_space.n) != state_count
        or int(environment.action_space.n) != action_count
        for environment in environments
    ):
        raise RuntimeError("tabular development tasks have incompatible spaces")

    q_values = np.zeros((state_count, action_count), dtype=np.float64)
    rng = np.random.default_rng(training_seed)
    total_steps = 0
    episode_index = 0
    checkpoints = []
    next_checkpoint = baseline.checkpoint_interval_steps
    started = perf_counter()
    try:
        while total_steps < baseline.training_steps_per_seed:
            task_index = (episode_index + training_seed) % len(tasks)
            environment = environments[task_index]
            state, _info = environment.reset(seed=training_seed + episode_index)
            terminated = truncated = False
            while not (terminated or truncated):
                progress = total_steps / baseline.training_steps_per_seed
                epsilon = max(0.05, 1.0 - progress / 0.8 * 0.95)
                action = _training_action(q_values, int(state), epsilon, rng)
                next_state, reward, terminated, truncated, _info = environment.step(
                    action
                )
                continuation = 0.0 if (terminated or truncated) else 0.99 * float(
                    q_values[int(next_state)].max()
                )
                target = float(reward) + continuation
                q_values[int(state), action] += 0.15 * (
                    target - q_values[int(state), action]
                )
                state = next_state
                total_steps += 1

                if total_steps == next_checkpoint:
                    evaluation = evaluate_q_table(
                        domain,
                        q_values,
                        episodes_per_task=evaluation_episodes_per_task,
                    )
                    relative = Path(domain) / baseline.name / f"seed-{training_seed}" / (
                        f"step-{total_steps}.json"
                    )
                    checkpoint_path = output_root / relative
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_payload = {
                        "protocol_id": "riskshiftbench-frontier-v2-tabular-q-checkpoint-v1",
                        "baseline_identifier": baseline.identifier,
                        "training_seed": training_seed,
                        "step": total_steps,
                        "state_count": state_count,
                        "action_count": action_count,
                        "q_values": q_values.tolist(),
                    }
                    checkpoint_path.write_text(
                        json.dumps(checkpoint_payload, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    checkpoints.append(
                        {
                            "step": total_steps,
                            "checkpoint_path": relative.as_posix(),
                            "checkpoint_sha256": file_sha256(checkpoint_path),
                            **evaluation,
                        }
                    )
                    next_checkpoint += baseline.checkpoint_interval_steps
                if total_steps >= baseline.training_steps_per_seed:
                    break
            episode_index += 1
    finally:
        for environment in environments:
            environment.close()
    selected = select_checkpoint(checkpoints, cost_limit=None)
    return {
        "training_seed": training_seed,
        "training_steps": total_steps,
        "training_episodes": episode_index,
        "runtime_seconds": perf_counter() - started,
        "checkpoints": checkpoints,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }


def train_tabular_q_baseline(
    domain: str,
    *,
    output_root: Path,
    source_root: Path,
) -> dict:
    try:
        baseline = SUPPORTED_BASELINES[domain]
    except KeyError as error:
        raise KeyError(f"no tabular Q-learning baseline for {domain}") from error
    source = source_root / SOURCE_DIRECTORIES["gymnasium"]
    source_audit = audit_codebase_source(source, "gymnasium")
    source_payload = asdict(source_audit)
    source_payload["source"] = SOURCE_DIRECTORIES["gymnasium"]
    design = baseline_design_summary()
    runs = [
        train_q_learning_seed(
            domain,
            training_seed=seed,
            output_root=output_root,
            source_root=source_root,
        )
        for seed in baseline.training_seeds
    ]
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-baseline-checkpoints-v1",
        "baseline_design_sha256": design["design_sha256"],
        "baseline_implementation_sha256": design[
            "internal_implementation_sha256"
        ],
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "development_manifest_sha256": task_manifest_sha256(
            all_tasks("development")
        ),
        "calibration_manifest_sha256": task_manifest_sha256(
            all_tasks("calibration")
        ),
        "source_lock": {"name": "riskshiftbench_internal"},
        "environment_codebase_lock": asdict(CODEBASE_LOCKS["gymnasium"]),
        "environment_source_audit": source_payload,
        "algorithm_hyperparameters": {
            "learning_rate": 0.15,
            "discount": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_fraction": 0.8,
            "task_sampling": "episode-stratified round robin",
        },
        "runs": runs,
    }
    manifest_path = output_root / domain / baseline.name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_tabular_q_baseline(
        args.domain,
        output_root=args.output_root,
        source_root=args.source_root,
    )
    selected = [
        {
            "training_seed": run["training_seed"],
            "selected_checkpoint_sha256": run["selected_checkpoint_sha256"],
            "runtime_seconds": run["runtime_seconds"],
        }
        for run in payload["runs"]
    ]
    print(json.dumps({"baseline": payload["baseline_identifier"], "runs": selected}, indent=2))


if __name__ == "__main__":
    main()
