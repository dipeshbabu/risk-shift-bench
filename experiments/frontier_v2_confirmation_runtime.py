"""Guarded, resumable confirmation state for the registered v2 protocol.

This module does not make confirmation tasks executable by itself. It defines
the deterministic pilot state machine that a separately registered wrapper
must validate before collecting or finalizing any confirmation outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
)
from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    V2ExternalTask,
    all_tasks,
    canonical_episode_seed_base,
    canonical_sha256,
    expected_episode_seeds,
    outcome_implementation_sha256,
    task_manifest_sha256,
)
from experiments.frontier_v2_external_adapters import (
    outcome_rows,
    run_gymnasium_task,
    run_minigrid_task,
    run_or_gym_task,
    run_safety_gymnasium_task,
)
from experiments.frontier_v2_protocol_lock import validate_protocol
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


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


def load_pilot_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank line in pilot log at {line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid pilot JSON record at line {line_number}"
                ) from error
    return records


def append_pilot_record(path: Path, record: dict) -> None:
    """Durably append one authenticated record without rewriting prior outcomes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite confirmation result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _registered_adapter_task(task: V2ExternalTask) -> V2ExternalTask:
    """Create the capability-bearing view consumed by development adapters.

    The environment identity, parameters, features, and task name are unchanged.
    Only the execution label is changed after the registered wrapper has passed
    every protocol check. This is the separately registered wrapper anticipated
    by the development adapters' confirmation guard.
    """

    if task.split != "confirmation":
        raise RuntimeError("registered adapter capability requires a confirmation task")
    return replace(task, split="calibration")


def _run_registered_policy(
    task: V2ExternalTask,
    policy: str,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> list[dict]:
    executable = _registered_adapter_task(task)
    source = source_root / SOURCE_DIRECTORIES[DOMAIN_SPECS[task.domain].codebase]
    codebase = DOMAIN_SPECS[task.domain].codebase
    if codebase == "gymnasium":
        outcomes = run_gymnasium_task(executable, policy, episodes, seed_base, source)
    elif codebase == "minigrid":
        outcomes = run_minigrid_task(executable, policy, episodes, seed_base, source)
    elif codebase == "or_gym":
        outcomes = run_or_gym_task(executable, policy, episodes, seed_base, source)
    elif codebase == "safety_gymnasium":
        outcomes = run_safety_gymnasium_task(
            executable, policy, episodes, seed_base, source
        )
    else:
        raise KeyError(codebase)
    return outcome_rows(outcomes)


def collect_registered_pair(
    design: dict,
    request: dict,
    *,
    codebase: str,
) -> dict:
    """Collect exactly one registered pilot pair inside its pinned environment."""

    router_lock = design["router_lock"]["content"]
    proposals = _proposal_index(router_lock)
    tasks = _confirmation_task_index()
    task_name = request["task"]
    try:
        task = tasks[task_name]
        proposal = proposals[task_name]
    except KeyError as error:
        raise RuntimeError("worker received an unknown confirmation task") from error
    if DOMAIN_SPECS[task.domain].codebase != codebase:
        raise RuntimeError("worker received a task from another codebase")
    within_task_index = int(request["within_task_index"])
    expected_seed_base = (
        canonical_episode_seed_base(task, stream="pilot") + within_task_index
    )
    if int(request["seed_base"]) != expected_seed_base:
        raise RuntimeError("worker pilot seed base differs from the registered schedule")
    if request["candidate_policy"] != proposal["candidate_policy"] or request[
        "fallback_policy"
    ] != proposal["fallback_policy"]:
        raise RuntimeError("worker pilot policy pair differs from the router lock")
    source_root = Path(design["artifact_roots"]["environment_source_root"])
    candidate = _run_registered_policy(
        task,
        proposal["candidate_policy"],
        episodes=1,
        seed_base=expected_seed_base,
        source_root=source_root,
    )[0]
    fallback = _run_registered_policy(
        task,
        proposal["fallback_policy"],
        episodes=1,
        seed_base=expected_seed_base,
        source_root=source_root,
    )[0]
    return {"candidate_outcome": candidate, "fallback_outcome": fallback}


def authenticate_worker_request(
    router_lock: dict,
    pilot_log_path: Path,
    request: dict,
) -> None:
    """Require the worker request to equal the unique next hash-chained step."""

    records = load_pilot_records(pilot_log_path)
    expected_request = next_pilot_request(router_lock, records)
    observed_request = {key: request.get(key) for key in expected_request or {}}
    if expected_request is None or observed_request != expected_request:
        raise RuntimeError(
            "worker request is not the unique next authenticated pilot step"
        )


def run_worker(protocol_path: Path, codebase: str, pilot_log_path: Path) -> None:
    """Serve registered episode requests over newline-delimited JSON."""

    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    if codebase not in set(SOURCE_DIRECTORIES):
        raise KeyError(codebase)
    print(json.dumps({"status": "ready", "codebase": codebase}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                print(json.dumps({"status": "closed", "codebase": codebase}), flush=True)
                return
            if request.get("command") != "pilot_pair":
                raise RuntimeError("unknown registered worker command")
            authenticate_worker_request(
                design["router_lock"]["content"], pilot_log_path, request
            )
            result = collect_registered_pair(design, request, codebase=codebase)
            print(json.dumps({"status": "ok", **result}, sort_keys=True), flush=True)
        except Exception as error:  # worker boundary must return an actionable failure
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return


class RegisteredWorkerPool:
    def __init__(
        self,
        protocol_path: Path,
        interpreters: dict[str, Path],
        pilot_log_path: Path,
    ):
        self.protocol_path = protocol_path
        self.interpreters = interpreters
        self.pilot_log_path = pilot_log_path
        self.processes: dict[str, subprocess.Popen[str]] = {}

    def _start(self, codebase: str) -> subprocess.Popen[str]:
        try:
            interpreter = self.interpreters[codebase]
        except KeyError as error:
            raise RuntimeError(f"no registered worker interpreter for {codebase}") from error
        if not interpreter.is_file():
            raise RuntimeError(f"registered worker interpreter is missing: {interpreter}")
        process = subprocess.Popen(
            [
                str(interpreter),
                "-m",
                "experiments.frontier_v2_confirmation_runtime",
                "worker",
                "--protocol",
                str(self.protocol_path),
                "--codebase",
                codebase,
                "--pilot-log",
                str(self.pilot_log_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("registered worker stdout was not created")
        ready_line = process.stdout.readline()
        if not ready_line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"registered worker failed to start: {stderr}")
        ready = json.loads(ready_line)
        if ready != {"status": "ready", "codebase": codebase}:
            raise RuntimeError(f"registered worker returned invalid readiness: {ready}")
        self.processes[codebase] = process
        return process

    def request(self, codebase: str, request: dict) -> dict:
        process = self.processes.get(codebase) or self._start(codebase)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("registered worker pipes are unavailable")
        process.stdin.write(json.dumps({"command": "pilot_pair", **request}) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"registered worker exited without a response: {stderr}")
        response = json.loads(response_line)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"registered worker failed: {response.get('error_type')}: "
                f"{response.get('error')}"
            )
        return response

    def close(self) -> None:
        for process in self.processes.values():
            if process.poll() is not None:
                continue
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=30)

    def __enter__(self) -> RegisteredWorkerPool:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def run_registered_pilot(
    protocol_path: Path,
    *,
    interpreters: dict[str, Path],
) -> dict:
    """Resume or complete the registered primary adaptive pilot."""

    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    router_lock = design["router_lock"]["content"]
    output_root = Path(design["confirmation"]["output_root"])
    log_path = output_root / "pilot" / "primary_records.jsonl"
    decisions_path = output_root / "pilot" / "primary_decisions.json"
    records = load_pilot_records(log_path)
    replay_pilot_records(router_lock, records)
    if decisions_path.exists():
        observed = json.loads(decisions_path.read_text(encoding="utf-8"))
        if observed != pilot_decision_summary(router_lock, records):
            raise RuntimeError("frozen pilot decisions changed")
        return observed

    with RegisteredWorkerPool(protocol_path, interpreters, log_path) as workers:
        while True:
            request = next_pilot_request(router_lock, records)
            if request is None:
                break
            task = _confirmation_task_index()[request["task"]]
            codebase = DOMAIN_SPECS[task.domain].codebase
            outcome = workers.request(codebase, request)
            record = build_pilot_record(
                router_lock,
                sequence_index=request["sequence_index"],
                within_task_index=request["within_task_index"],
                task_name=request["task"],
                candidate_outcome=outcome["candidate_outcome"],
                fallback_outcome=outcome["fallback_outcome"],
                previous_record_sha256=request["previous_record_sha256"],
            )
            append_pilot_record(log_path, record)
            records.append(record)
    summary = pilot_decision_summary(router_lock, records)
    write_json_once(decisions_path, summary)
    return summary


def _default_interpreters() -> dict[str, Path]:
    return {
        "gymnasium": Path("artifacts/frontier_v2_envs/gymnasium/Scripts/python.exe"),
        "or_gym": Path("artifacts/frontier_v2_envs/or-gym/Scripts/python.exe"),
        "safety_gymnasium": Path(
            "artifacts/frontier_v2_envs/safety-gymnasium/Scripts/python.exe"
        ),
        "minigrid": Path("artifacts/frontier_v2_envs/minigrid/Scripts/python.exe"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--protocol", type=Path, required=True)
    worker.add_argument("--codebase", choices=tuple(SOURCE_DIRECTORIES), required=True)
    worker.add_argument("--pilot-log", type=Path, required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--protocol", type=Path, required=True)
    for codebase, path in _default_interpreters().items():
        pilot.add_argument(
            f"--{codebase.replace('_', '-')}-python",
            type=Path,
            default=path,
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        run_worker(args.protocol, args.codebase, args.pilot_log)
        return
    interpreters = {
        codebase: getattr(args, f"{codebase}_python")
        for codebase in SOURCE_DIRECTORIES
    }
    summary = run_registered_pilot(args.protocol, interpreters=interpreters)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
