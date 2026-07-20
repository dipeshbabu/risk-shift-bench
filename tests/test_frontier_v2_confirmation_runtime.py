from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.frontier_v2_confirmation_runtime import (
    build_pilot_record,
    next_pilot_request,
    replay_pilot_records,
)
from experiments.frontier_v2_external_design import (
    all_tasks,
    canonical_sha256,
    outcome_implementation_sha256,
    task_manifest_sha256,
    task_sha256,
)


def _router_lock(*, budget: int = 4) -> dict:
    tasks = all_tasks("confirmation")
    proposals = [
        {
            "task": task.name,
            "domain": task.domain,
            "task_sha256": task_sha256(task),
            "candidate_policy": "candidate",
            "fallback_policy": "fallback",
            "planning_effect_gap": 0.2,
            "task_weight": 1.0 / len(tasks),
        }
        for task in tasks
    ]
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-router-lock-v1",
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "confirmation_task_manifest_sha256": task_manifest_sha256(tasks),
        "proposal_family_size": len(tasks),
        "anytime_plan": {
            "familywise_alpha": 0.05,
            "futility_familywise_alpha": 0.05,
            "effect_margin": 0.0,
            "observation_bounds": [-1.0, 1.0],
            "minimum_observations": 1,
            "maximum_observations_per_task": 3,
            "global_observation_budget": budget,
            "forced_initial_observations": 1,
            "e_process_method": "betting_mixture",
            "betting_fraction_grid": [0.2, 0.4],
            "allocation": "certified",
            "resolution_familywise_beta": 0.05,
        },
        "proposals": proposals,
    }
    payload["router_lock_canonical_sha256"] = canonical_sha256(payload)
    return payload


def _outcome(request: dict, policy: str, score: float) -> dict:
    return {
        "task": request["task"],
        "domain": request["domain"],
        "policy": policy,
        "seed": request["expected_seed"],
        "score": score,
    }


def _append(lock: dict, records: list[dict], difference: float = 0.1) -> None:
    request = next_pilot_request(lock, records)
    assert request is not None
    fallback_score = 0.4
    records.append(
        build_pilot_record(
            lock,
            sequence_index=request["sequence_index"],
            within_task_index=request["within_task_index"],
            task_name=request["task"],
            candidate_outcome=_outcome(
                request, request["candidate_policy"], fallback_score + difference
            ),
            fallback_outcome=_outcome(
                request, request["fallback_policy"], fallback_score
            ),
            previous_record_sha256=request["previous_record_sha256"],
        )
    )


def test_pilot_state_machine_replays_registered_order_and_budget() -> None:
    lock = _router_lock(budget=3)
    records: list[dict] = []
    for _ in range(3):
        _append(lock, records)
    router = replay_pilot_records(lock, records)
    assert router.total_observations() == 3
    assert next_pilot_request(lock, records) is None


def test_pilot_hash_chain_rejects_edited_outcome() -> None:
    lock = _router_lock()
    records: list[dict] = []
    _append(lock, records)
    edited = deepcopy(records)
    edited[0]["candidate_outcome"]["score"] = 0.9
    with pytest.raises(RuntimeError, match="content hash"):
        replay_pilot_records(lock, edited)


def test_pilot_replay_rejects_nonregistered_adaptive_order() -> None:
    lock = _router_lock()
    records: list[dict] = []
    _append(lock, records)
    edited = deepcopy(records)
    edited[0]["task"] = all_tasks("confirmation")[-1].name
    edited[0]["record_sha256"] = canonical_sha256(
        {key: value for key, value in edited[0].items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="allocation order"):
        replay_pilot_records(lock, edited)


def test_router_lock_hash_is_required_before_pilot_replay() -> None:
    lock = _router_lock()
    lock["anytime_plan"]["global_observation_budget"] += 1
    with pytest.raises(RuntimeError, match="canonical hash"):
        replay_pilot_records(lock, [])
