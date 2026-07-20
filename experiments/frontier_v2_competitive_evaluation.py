"""Evaluate frozen competitive references after the registered primary final stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from experiments.frontier_v2_baseline_audit import audit_baseline_manifest
from experiments.frontier_v2_baseline_design import COMPETITIVE_BASELINES
from experiments.frontier_v2_confirmation_runtime import (
    run_registered_final_manifest_only,
    write_json_once,
)
from experiments.frontier_v2_external_adapters import bounded_score, outcome_rows
from experiments.frontier_v2_external_design import (
    all_tasks,
    canonical_episode_seed_base,
    domain_tasks,
    expected_episode_seeds,
    task_manifest_sha256,
)
from experiments.frontier_v2_protocol_lock import validate_protocol


EPISODES_PER_TASK_PER_CHECKPOINT = 100


def _baseline(identifier: str):
    try:
        return next(
            baseline
            for baseline in COMPETITIVE_BASELINES
            if baseline.identifier == identifier
        )
    except StopIteration as error:
        raise KeyError(identifier) from error


def byte_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_checkpoint_records(manifest: dict, baseline_root: Path) -> list[dict]:
    records = []
    for run in manifest["runs"]:
        matches = [
            checkpoint
            for checkpoint in run["checkpoints"]
            if checkpoint["checkpoint_sha256"] == run["selected_checkpoint_sha256"]
        ]
        if len(matches) != 1:
            raise RuntimeError("selected competitive checkpoint is not unique")
        selected = matches[0]
        path = baseline_root / selected["checkpoint_path"]
        if not path.is_file() or byte_sha256(path) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"selected competitive checkpoint changed: {path}")
        records.append(
            {
                "training_seed": int(run["training_seed"]),
                "step": int(selected["step"]),
                "checkpoint_path": path,
                "checkpoint_sha256": selected["checkpoint_sha256"],
            }
        )
    return sorted(records, key=lambda record: record["training_seed"])


def _score_records_summary(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("competitive evaluation emitted no episode records")
    task_names = sorted({record["task"] for record in records})
    task_scores = {
        task: fmean(
            float(record["score"]) for record in records if record["task"] == task
        )
        for task in task_names
    }
    task_costs = {
        task: fmean(
            float(record["cost"]) for record in records if record["task"] == task
        )
        for task in task_names
    }
    return {
        "episode_record_count": len(records),
        "task_count": len(task_names),
        "equal_task_mean_score": fmean(task_scores.values()),
        "equal_task_mean_cost": fmean(task_costs.values()),
        "task_mean_scores": task_scores,
        "task_mean_costs": task_costs,
    }


def _evaluate_nonlearned(baseline, design: dict) -> tuple[list[dict], list[dict]]:
    from experiments.frontier_v2_nonlearned_baselines import (
        _build_v1_frozenlake_router,
        _run_reference,
    )

    roots = design["artifact_roots"]
    source_root = Path(roots["environment_source_root"])
    v1_router = None
    if baseline.name == "v1_fixed_router":
        v1_router, _inputs = _build_v1_frozenlake_router(
            v1_development_root=Path(roots["v1_development_root"]),
            v1_router_root=Path(roots["v1_router_root"]),
        )
    records = []
    metadata = []
    for task in domain_tasks(baseline.domain, "confirmation"):
        rows, task_metadata = _run_reference(
            baseline,
            task,
            episodes=EPISODES_PER_TASK_PER_CHECKPOINT,
            seed_base=canonical_episode_seed_base(task, stream="final"),
            source_root=source_root,
            v1_router=v1_router,
        )
        records.extend(
            {
                "baseline_identifier": baseline.identifier,
                "training_seed": None,
                "checkpoint_sha256": None,
                **row,
            }
            for row in outcome_rows(rows)
        )
        metadata.append({"task": task.name, **task_metadata})
    return records, metadata


def _evaluate_tabular_q(
    baseline,
    design: dict,
    selected: list[dict],
) -> list[dict]:
    import numpy as np

    from experiments.frontier_v2_external_adapters import _activate_verified_source
    from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES
    from experiments.frontier_v2_tabular_q_learning import (
        _episode_score,
        _greedy_action,
        _make_environment,
    )

    source_root = Path(design["artifact_roots"]["environment_source_root"])
    _activate_verified_source(source_root / SOURCE_DIRECTORIES["gymnasium"], "gymnasium")
    records = []
    for checkpoint in selected:
        payload = json.loads(
            checkpoint["checkpoint_path"].read_text(encoding="utf-8")
        )
        q_values = np.asarray(payload["q_values"], dtype=np.float64)
        for task in domain_tasks(baseline.domain, "confirmation"):
            environment = _make_environment(task)
            try:
                seeds = expected_episode_seeds(
                    task,
                    episodes=EPISODES_PER_TASK_PER_CHECKPOINT,
                    seed_base=canonical_episode_seed_base(task, stream="final"),
                )
                for episode, seed in enumerate(seeds):
                    state, _info = environment.reset(seed=seed)
                    raw_return = 0.0
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
                        raw_return += numeric_reward
                        cost += float(
                            (
                                baseline.domain == "gymnasium_cliffwalking"
                                and numeric_reward <= -100.0
                            )
                            or (
                                baseline.domain == "gymnasium_taxi"
                                and numeric_reward <= -10.0
                            )
                        )
                        success = success or (
                            (
                                baseline.domain == "gymnasium_cliffwalking"
                                and bool(terminated)
                            )
                            or (
                                baseline.domain == "gymnasium_taxi"
                                and numeric_reward >= 20.0
                            )
                        )
                        steps += 1
                    records.append(
                        {
                            "baseline_identifier": baseline.identifier,
                            "training_seed": checkpoint["training_seed"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "task": task.name,
                            "domain": task.domain,
                            "episode": episode,
                            "seed": seed,
                            "score": _episode_score(
                                task, success=success, steps=steps, cost=cost
                            ),
                            "cost": cost,
                            "raw_return": raw_return,
                            "steps": steps,
                            "successes": int(success),
                            "failure": not success,
                        }
                    )
            finally:
                environment.close()
    return records


def _evaluate_double_dqn(
    baseline,
    design: dict,
    selected: list[dict],
) -> list[dict]:
    from experiments.frontier_v2_double_dqn import (
        _action_mask,
        _activate_domain_source,
        _build_q_network,
        _encoded_observation,
        _episode_metrics,
        _greedy_action,
        _input_and_action_sizes,
        _start_episode,
        _step_episode,
        _torch_modules,
    )

    source_root = Path(design["artifact_roots"]["environment_source_root"])
    _activate_domain_source(baseline.domain, source_root)
    _np, torch = _torch_modules()
    device = torch.device("cpu")
    input_size, action_count = _input_and_action_sizes(baseline.domain)
    records = []
    for checkpoint in selected:
        payload = torch.load(
            checkpoint["checkpoint_path"], map_location=device, weights_only=True
        )
        network = _build_q_network(torch, input_size, action_count).to(device)
        network.load_state_dict(payload["online_state_dict"], strict=True)
        network.eval()
        for task in domain_tasks(baseline.domain, "confirmation"):
            seeds = expected_episode_seeds(
                task,
                episodes=EPISODES_PER_TASK_PER_CHECKPOINT,
                seed_base=canonical_episode_seed_base(task, stream="final"),
            )
            for episode, seed in enumerate(seeds):
                context, raw_observation = _start_episode(
                    baseline.domain, task, seed
                )
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
                    records.append(
                        {
                            "baseline_identifier": baseline.identifier,
                            "training_seed": checkpoint["training_seed"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "task": task.name,
                            "domain": task.domain,
                            "episode": episode,
                            "seed": seed,
                            "score": score,
                            "cost": cost,
                            "raw_return": context.raw_return,
                            "steps": context.steps,
                        }
                    )
                finally:
                    context.environment.close()
    return records


def _evaluate_ppo(
    baseline,
    design: dict,
    selected: list[dict],
) -> list[dict]:
    from experiments.frontier_v2_double_dqn import _activate_domain_source
    from experiments.frontier_v2_ppo import (
        _build_agent,
        _metrics,
        _start,
        _step,
        _torch_modules,
        _zero_memory,
    )

    source_root = Path(design["artifact_roots"]["environment_source_root"])
    _activate_domain_source(baseline.domain, source_root)
    _np, torch = _torch_modules()
    device = torch.device("cpu")
    records = []
    for checkpoint in selected:
        payload = torch.load(
            checkpoint["checkpoint_path"], map_location=device, weights_only=True
        )
        agent, _input_size = _build_agent(baseline.domain, torch)
        agent.load_state_dict(payload["agent_state_dict"], strict=True)
        agent.to(device)
        agent.eval()
        for task in domain_tasks(baseline.domain, "confirmation"):
            seeds = expected_episode_seeds(
                task,
                episodes=EPISODES_PER_TASK_PER_CHECKPOINT,
                seed_base=canonical_episode_seed_base(task, stream="final"),
            )
            for episode, seed in enumerate(seeds):
                context, observation = _start(baseline.domain, task, seed)
                hidden, cell = _zero_memory(agent, torch, device)
                episode_start = torch.ones(1, device=device)
                try:
                    done = False
                    while not done:
                        observation_tensor = torch.as_tensor(
                            observation, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            if agent.recurrent:
                                distribution, _value, hidden, cell = agent.step(
                                    observation_tensor,
                                    hidden,
                                    cell,
                                    episode_start,
                                )
                                action = int(distribution.probs.argmax(dim=1).item())
                            else:
                                distributions, _value = agent.distribution_and_value(
                                    observation_tensor
                                )
                                action = [
                                    int(distribution.probs.argmax(dim=1).item())
                                    for distribution in distributions
                                ]
                        observation, _reward, done = _step(
                            baseline.domain, context, action
                        )
                        episode_start.zero_()
                    score, cost = _metrics(baseline.domain, context)
                    records.append(
                        {
                            "baseline_identifier": baseline.identifier,
                            "training_seed": checkpoint["training_seed"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "task": task.name,
                            "domain": task.domain,
                            "episode": episode,
                            "seed": seed,
                            "score": score,
                            "cost": cost,
                            "raw_return": context.raw_return,
                            "steps": context.steps,
                        }
                    )
                finally:
                    context.environment.close()
    return records


def _evaluate_safe_rl(
    baseline,
    design: dict,
    selected: list[dict],
) -> list[dict]:
    import safety_gymnasium
    import torch

    from experiments.frontier_v2_omnisafe import (
        _activate_sources,
        _frozen_normalize,
        _load_policy,
        _pad_safety_observation,
    )

    roots = design["artifact_roots"]
    _activate_sources(
        Path(roots["environment_source_root"]),
        Path(roots["baseline_source_root"]),
    )
    records = []
    for checkpoint in selected:
        actor, normalizer = _load_policy(
            checkpoint["checkpoint_path"], baseline.domain
        )
        for task in domain_tasks(baseline.domain, "confirmation"):
            parameters = task.parameter_dict()
            cost_weight = float(parameters["cost_weight"])
            max_steps = int(parameters["max_steps"])
            seeds = expected_episode_seeds(
                task,
                episodes=EPISODES_PER_TASK_PER_CHECKPOINT,
                seed_base=canonical_episode_seed_base(task, stream="final"),
            )
            for episode, seed in enumerate(seeds):
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
                            _pad_safety_observation(baseline.domain, observation),
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
                    records.append(
                        {
                            "baseline_identifier": baseline.identifier,
                            "training_seed": checkpoint["training_seed"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "task": task.name,
                            "domain": task.domain,
                            "episode": episode,
                            "seed": seed,
                            "score": bounded_score(
                                0.5 + 0.5 * math.tanh(raw_utility / 25.0)
                            ),
                            "cost": total_cost,
                            "raw_return": raw_return,
                            "raw_utility": raw_utility,
                            "steps": steps,
                            "successes": successes,
                            "failure": failure,
                        }
                    )
                finally:
                    environment.close()
                    del environment
                    gc.collect()
    return records


def evaluate_registered_baseline(protocol_path: Path, identifier: str) -> dict:
    """Evaluate one frozen reference only after the primary final manifest exists."""

    baseline = _baseline(identifier)
    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    primary_final = run_registered_final_manifest_only(design)
    baseline_root = Path(design["artifact_roots"]["baseline_root"])
    manifest_path = baseline_root / baseline.domain / baseline.name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = (
        Path(design["confirmation"]["output_root"])
        / "competitive"
        / baseline.domain
        / f"{baseline.name}.json"
    )
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("baseline_manifest_byte_sha256") != byte_sha256(manifest_path):
            raise RuntimeError("competitive result baseline manifest changed")
        return payload

    selected = []
    metadata = []
    if baseline.kind == "learned_policy":
        audit_baseline_manifest(manifest, checkpoint_root=baseline_root)
        selected = selected_checkpoint_records(manifest, baseline_root)
        if baseline.algorithm == "tabular Q-learning":
            records = _evaluate_tabular_q(baseline, design, selected)
        elif baseline.algorithm == "double DQN":
            records = _evaluate_double_dqn(baseline, design, selected)
        elif baseline.algorithm in {"clipped PPO", "recurrent PPO"}:
            records = _evaluate_ppo(baseline, design, selected)
        elif baseline.algorithm in {
            "PPO-Lagrangian",
            "constrained policy optimization",
        }:
            records = _evaluate_safe_rl(baseline, design, selected)
        else:
            raise KeyError(baseline.algorithm)
    else:
        records, metadata = _evaluate_nonlearned(baseline, design)

    expected_rows = (
        EPISODES_PER_TASK_PER_CHECKPOINT
        * len(domain_tasks(baseline.domain, "confirmation"))
        * (len(selected) if selected else 1)
    )
    if len(records) != expected_rows:
        raise RuntimeError("competitive reference episode coverage is incomplete")
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-competitive-final-v1",
        "scope": (
            "Registered confirmation reference evaluated only after the primary "
            "pilot decisions and primary final manifest were frozen."
        ),
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "baseline_manifest_path": manifest_path.as_posix(),
        "baseline_manifest_byte_sha256": byte_sha256(manifest_path),
        "primary_final_task_count": int(primary_final["task_count"]),
        "confirmation_task_manifest_sha256": task_manifest_sha256(
            all_tasks("confirmation")
        ),
        "domain_confirmation_manifest_sha256": task_manifest_sha256(
            domain_tasks(baseline.domain, "confirmation")
        ),
        "episodes_per_task_per_selected_checkpoint": (
            EPISODES_PER_TASK_PER_CHECKPOINT
        ),
        "selected_checkpoints": [
            {
                **record,
                "checkpoint_path": record["checkpoint_path"].as_posix(),
            }
            for record in selected
        ],
        "reference_metadata": metadata,
        "summary": _score_records_summary(records),
        "records": records,
    }
    write_json_once(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=tuple(baseline.identifier for baseline in COMPETITIVE_BASELINES),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_registered_baseline(args.protocol, args.baseline)
    print(
        json.dumps(
            {
                "baseline_identifier": payload["baseline_identifier"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
