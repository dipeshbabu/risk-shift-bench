"""Exact pilot-budget accounting for the external confirmation study."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from experiments.external_familywise_verifier import FamilywisePilotPlan


@dataclass(frozen=True)
class TaskPilotAllocation:
    task: str
    batches: int
    episodes_per_batch: int

    @property
    def paired_policy_episodes(self) -> int:
        return 2 * self.batches * self.episodes_per_batch


def proposal_focused_allocation(
    proposal_tasks: list[str],
    plan: FamilywisePilotPlan,
) -> list[TaskPilotAllocation]:
    tasks = sorted(set(proposal_tasks))
    if len(tasks) != plan.proposal_family_size:
        raise ValueError("proposal task count does not match the fixed family size")
    return [
        TaskPilotAllocation(
            task=task,
            batches=plan.required_unanimous_batches,
            episodes_per_batch=plan.episodes_per_batch,
        )
        for task in tasks
    ]


def uniform_task_allocation(
    all_tasks: list[str],
    total_batches: int,
    episodes_per_batch: int,
) -> list[TaskPilotAllocation]:
    """Allocate an integer batch budget across all tasks without outcome access."""

    tasks = sorted(set(all_tasks))
    if not tasks:
        raise ValueError("uniform allocation requires at least one task")
    if len(tasks) != len(all_tasks):
        raise ValueError("task names must be unique")
    if total_batches < 0:
        raise ValueError("total_batches cannot be negative")
    if episodes_per_batch <= 0:
        raise ValueError("episodes_per_batch must be positive")
    base, remainder = divmod(total_batches, len(tasks))
    return [
        TaskPilotAllocation(
            task=task,
            batches=base + (index < remainder),
            episodes_per_batch=episodes_per_batch,
        )
        for index, task in enumerate(tasks)
    ]


def total_policy_episodes(allocation: list[TaskPilotAllocation]) -> int:
    return sum(item.paired_policy_episodes for item in allocation)


def random_task_allocation(
    all_tasks: list[str],
    selected_task_count: int,
    plan: FamilywisePilotPlan,
    seed: int,
) -> list[TaskPilotAllocation]:
    """Sample tasks without outcomes and give each the complete gate budget."""

    tasks = sorted(set(all_tasks))
    if len(tasks) != len(all_tasks):
        raise ValueError("task names must be unique")
    if not 0 < selected_task_count <= len(tasks):
        raise ValueError("selected_task_count must lie between one and the task count")
    selected = sorted(random.Random(seed).sample(tasks, selected_task_count))
    return [
        TaskPilotAllocation(
            task=task,
            batches=plan.required_unanimous_batches,
            episodes_per_batch=plan.episodes_per_batch,
        )
        for task in selected
    ]


def fixed_budget_comparison(
    all_tasks: list[str],
    proposal_tasks: list[str],
    plan: FamilywisePilotPlan,
    random_allocation_seed: int = 20_260_715,
) -> dict:
    focused = proposal_focused_allocation(proposal_tasks, plan)
    focused_batches = sum(item.batches for item in focused)
    uniform = uniform_task_allocation(
        all_tasks=all_tasks,
        total_batches=focused_batches,
        episodes_per_batch=plan.episodes_per_batch,
    )
    random_tasks = random_task_allocation(
        all_tasks=all_tasks,
        selected_task_count=len(focused),
        plan=plan,
        seed=random_allocation_seed,
    )
    focused_cost = total_policy_episodes(focused)
    uniform_cost = total_policy_episodes(uniform)
    random_cost = total_policy_episodes(random_tasks)
    if focused_cost != uniform_cost or focused_cost != random_cost:
        raise AssertionError("budget-matched allocations changed total policy episodes")
    return {
        "scope": "Pre-specified allocation rules; no pilot or final outcomes are used.",
        "proposal_family_size": plan.proposal_family_size,
        "familywise_alpha": plan.familywise_alpha,
        "bonferroni_local_alpha": plan.local_alpha,
        "required_unanimous_batches": plan.required_unanimous_batches,
        "episodes_per_batch": plan.episodes_per_batch,
        "total_candidate_plus_fallback_episodes": focused_cost,
        "proposal_focused": [asdict(item) for item in focused],
        "uniform_all_tasks": [asdict(item) for item in uniform],
        "outcome_blind_random_tasks": {
            "seed": random_allocation_seed,
            "allocation": [asdict(item) for item in random_tasks],
        },
    }
