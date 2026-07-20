"""Guarded, resumable confirmation state for the registered v2 protocol.

This module does not make confirmation tasks executable by itself. It defines
the deterministic pilot state machine that a separately registered wrapper
must validate before collecting or finalizing any confirmation outcome.
"""

from __future__ import annotations

from dataclasses import asdict

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
)
from experiments.frontier_v2_external_design import (
    all_tasks,
    canonical_episode_seed_base,
    canonical_sha256,
    expected_episode_seeds,
    outcome_implementation_sha256,
    task_manifest_sha256,
)


PILOT_RECORD_PROTOCOL = "riskshiftbench-frontier-v2-pilot-record-v1"


def _proposal_index(router_lock: dict) -> dict[str, dict]:
    proposals = router_lock.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise RuntimeError("router lock has no proposal family")
    index = {proposal["task"]: proposal for proposal in proposals}
    if len(index) != len(proposals):
        raise RuntimeError("router lock contains duplicate proposal tasks")
    expected_tasks = {task.name for task in all_tasks("confirmation")}
    if set(index) != expected_tasks:
        raise RuntimeError("router lock no longer covers the confirmation suite")
    return index


def validate_router_lock_for_confirmation(router_lock: dict) -> None:
    if router_lock.get("protocol_id") != "riskshiftbench-frontier-v2-router-lock-v1":
        raise RuntimeError("unexpected v2 router-lock protocol")
    expected_hash = router_lock.get("router_lock_canonical_sha256")
    unsigned = dict(router_lock)
    unsigned.pop("router_lock_canonical_sha256", None)
    if expected_hash != canonical_sha256(unsigned):
        raise RuntimeError("router-lock canonical hash changed")
    if router_lock.get("outcome_implementation_sha256") != outcome_implementation_sha256():
        raise RuntimeError("outcome implementation changed after the router lock")
    if router_lock.get("confirmation_task_manifest_sha256") != task_manifest_sha256(
        all_tasks("confirmation")
    ):
        raise RuntimeError("confirmation task manifest changed after the router lock")
    proposals = _proposal_index(router_lock)
    if int(router_lock.get("proposal_family_size", -1)) != len(proposals):
        raise RuntimeError("proposal-family size changed")


def plan_from_router_lock(router_lock: dict) -> AnytimeFamilywisePlan:
    validate_router_lock_for_confirmation(router_lock)
    proposals = sorted(router_lock["proposals"], key=lambda item: item["task"])
    settings = router_lock["anytime_plan"]
    if settings.get("e_process_method") != "betting_mixture":
        raise RuntimeError("registered primary e-process is not the betting mixture")
    if settings.get("allocation") != "certified":
        raise RuntimeError("registered primary allocation is not certified")
    bounds = settings["observation_bounds"]
    return AnytimeFamilywisePlan(
        task_names=tuple(proposal["task"] for proposal in proposals),
        familywise_alpha=float(settings["familywise_alpha"]),
        futility_familywise_alpha=float(settings["futility_familywise_alpha"]),
        effect_margin=float(settings["effect_margin"]),
        observation_lower=float(bounds[0]),
        observation_upper=float(bounds[1]),
        minimum_observations=int(settings["minimum_observations"]),
        maximum_observations_per_task=int(
            settings["maximum_observations_per_task"]
        ),
        task_weights=tuple(
            (proposal["task"], float(proposal["task_weight"]))
            for proposal in proposals
        ),
        e_process_method=settings["e_process_method"],
        betting_fraction_grid=tuple(
            float(value) for value in settings["betting_fraction_grid"]
        ),
        planning_effect_gaps=tuple(
            (proposal["task"], float(proposal["planning_effect_gap"]))
            for proposal in proposals
        ),
        resolution_familywise_beta=float(
            settings["resolution_familywise_beta"]
        ),
    )


def pilot_record_sha256(record: dict) -> str:
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    return canonical_sha256(unsigned)


def _confirmation_task_index() -> dict[str, object]:
    return {task.name: task for task in all_tasks("confirmation")}


def _validate_outcome_identity(
    outcome: dict,
    *,
    task: str,
    domain: str,
    policy: str,
    seed: int,
) -> None:
    if outcome.get("task") != task or outcome.get("domain") != domain:
        raise RuntimeError("pilot outcome task identity changed")
    if outcome.get("policy") != policy:
        raise RuntimeError("pilot outcome policy changed")
    if int(outcome.get("seed", -1)) != seed:
        raise RuntimeError("pilot outcome seed changed")
    score = float(outcome["score"])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("pilot outcome score lies outside [0, 1]")


def build_pilot_record(
    router_lock: dict,
    *,
    sequence_index: int,
    within_task_index: int,
    task_name: str,
    candidate_outcome: dict,
    fallback_outcome: dict,
    previous_record_sha256: str | None,
) -> dict:
    proposals = _proposal_index(router_lock)
    try:
        proposal = proposals[task_name]
        task = _confirmation_task_index()[task_name]
    except KeyError as error:
        raise KeyError(f"unknown confirmation proposal: {task_name}") from error
    seed_base = canonical_episode_seed_base(task, stream="pilot") + within_task_index
    expected_seed = expected_episode_seeds(
        task,
        episodes=1,
        seed_base=seed_base,
    )[0]
    _validate_outcome_identity(
        candidate_outcome,
        task=task_name,
        domain=proposal["domain"],
        policy=proposal["candidate_policy"],
        seed=expected_seed,
    )
    _validate_outcome_identity(
        fallback_outcome,
        task=task_name,
        domain=proposal["domain"],
        policy=proposal["fallback_policy"],
        seed=expected_seed,
    )
    difference = float(candidate_outcome["score"]) - float(
        fallback_outcome["score"]
    )
    record = {
        "protocol_id": PILOT_RECORD_PROTOCOL,
        "sequence_index": int(sequence_index),
        "within_task_index": int(within_task_index),
        "task": task_name,
        "domain": proposal["domain"],
        "candidate_policy": proposal["candidate_policy"],
        "fallback_policy": proposal["fallback_policy"],
        "seed": expected_seed,
        "candidate_outcome": candidate_outcome,
        "fallback_outcome": fallback_outcome,
        "paired_score_difference": difference,
        "previous_record_sha256": previous_record_sha256,
    }
    record["record_sha256"] = pilot_record_sha256(record)
    return record


def replay_pilot_records(
    router_lock: dict,
    records: list[dict],
) -> AnytimeFamilywiseRouter:
    """Replay and authenticate a possibly partial primary pilot log."""

    plan = plan_from_router_lock(router_lock)
    router = AnytimeFamilywiseRouter(plan)
    settings = router_lock["anytime_plan"]
    budget = int(settings["global_observation_budget"])
    forced = int(settings["forced_initial_observations"])
    if len(records) > budget:
        raise RuntimeError("pilot log exceeds the registered global budget")
    previous_hash = None
    task_index = _confirmation_task_index()
    proposals = _proposal_index(router_lock)
    for sequence_index, record in enumerate(records):
        if record.get("protocol_id") != PILOT_RECORD_PROTOCOL:
            raise RuntimeError("unexpected pilot-record protocol")
        if int(record.get("sequence_index", -1)) != sequence_index:
            raise RuntimeError("pilot sequence is not contiguous")
        if record.get("previous_record_sha256") != previous_hash:
            raise RuntimeError("pilot record hash chain is broken")
        if record.get("record_sha256") != pilot_record_sha256(record):
            raise RuntimeError("pilot record content hash changed")
        expected_task = router.next_task(
            "certified", forced_initial_observations=forced
        )
        if expected_task is None or record.get("task") != expected_task:
            raise RuntimeError("pilot allocation order differs from the registered rule")
        evidence = router.evidence(expected_task)
        if int(record.get("within_task_index", -1)) != evidence.observations:
            raise RuntimeError("within-task pilot index changed")
        proposal = proposals[expected_task]
        task = task_index[expected_task]
        expected_seed = expected_episode_seeds(
            task,
            episodes=1,
            seed_base=(
                canonical_episode_seed_base(task, stream="pilot")
                + evidence.observations
            ),
        )[0]
        if int(record.get("seed", -1)) != expected_seed:
            raise RuntimeError("pilot seed schedule changed")
        _validate_outcome_identity(
            record["candidate_outcome"],
            task=expected_task,
            domain=proposal["domain"],
            policy=proposal["candidate_policy"],
            seed=expected_seed,
        )
        _validate_outcome_identity(
            record["fallback_outcome"],
            task=expected_task,
            domain=proposal["domain"],
            policy=proposal["fallback_policy"],
            seed=expected_seed,
        )
        observed_difference = float(record["candidate_outcome"]["score"]) - float(
            record["fallback_outcome"]["score"]
        )
        if float(record.get("paired_score_difference")) != observed_difference:
            raise RuntimeError("paired pilot difference changed")
        router.update(expected_task, observed_difference)
        previous_hash = record["record_sha256"]
    return router


def next_pilot_request(router_lock: dict, records: list[dict]) -> dict | None:
    """Return the unique next registered request, or ``None`` at termination."""

    router = replay_pilot_records(router_lock, records)
    settings = router_lock["anytime_plan"]
    if router.total_observations() >= int(settings["global_observation_budget"]):
        return None
    task_name = router.next_task(
        "certified",
        forced_initial_observations=int(settings["forced_initial_observations"]),
    )
    if task_name is None:
        return None
    task = _confirmation_task_index()[task_name]
    proposal = _proposal_index(router_lock)[task_name]
    within_task_index = router.evidence(task_name).observations
    seed_base = canonical_episode_seed_base(task, stream="pilot") + within_task_index
    return {
        "sequence_index": len(records),
        "within_task_index": within_task_index,
        "task": task_name,
        "domain": proposal["domain"],
        "candidate_policy": proposal["candidate_policy"],
        "fallback_policy": proposal["fallback_policy"],
        "seed_base": seed_base,
        "expected_seed": expected_episode_seeds(
            task,
            episodes=1,
            seed_base=seed_base,
        )[0],
        "previous_record_sha256": (
            records[-1]["record_sha256"] if records else None
        ),
    }


def pilot_decision_summary(router_lock: dict, records: list[dict]) -> dict:
    router = replay_pilot_records(router_lock, records)
    settings = router_lock["anytime_plan"]
    next_request = next_pilot_request(router_lock, records)
    if next_request is not None:
        raise RuntimeError("pilot decisions cannot be frozen before termination")
    decisions = {
        task: asdict(evidence) for task, evidence in router.decisions().items()
    }
    return {
        "protocol_id": "riskshiftbench-frontier-v2-pilot-decisions-v1",
        "router_lock_canonical_sha256": router_lock[
            "router_lock_canonical_sha256"
        ],
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "confirmation_task_manifest_sha256": task_manifest_sha256(
            all_tasks("confirmation")
        ),
        "global_observation_budget": int(settings["global_observation_budget"]),
        "total_observations": router.total_observations(),
        "pilot_terminated_by": (
            "all_tasks_resolved"
            if router.next_task(
                "certified",
                forced_initial_observations=int(
                    settings["forced_initial_observations"]
                ),
            )
            is None
            else "global_budget"
        ),
        "accepted_tasks": list(router.accepted_tasks()),
        "decisions": decisions,
        "pilot_record_count": len(records),
        "pilot_chain_tip_sha256": (
            records[-1]["record_sha256"] if records else None
        ),
    }
