"""Familywise pilot verification for a fixed external proposal family.

This module is separate from the completed three-domain protocol.  It adds a
Bonferroni gate for the future external study without changing the source hash
or interpretation of the earlier per-proposal gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

from experiments.pilot_verifier import PilotGateParams, PilotGateResult, verify_promotion


@dataclass(frozen=True)
class FamilywisePilotPlan:
    proposal_family_size: int
    familywise_alpha: float = 0.05
    episodes_per_batch: int = 50
    min_mean_advantage: float = 0.0

    def __post_init__(self) -> None:
        if self.proposal_family_size <= 0:
            raise ValueError("proposal_family_size must be positive")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie strictly between zero and one")
        if self.episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")

    @property
    def local_alpha(self) -> float:
        return self.familywise_alpha / self.proposal_family_size

    @property
    def required_unanimous_batches(self) -> int:
        return ceil(log2(self.proposal_family_size / self.familywise_alpha))

    @property
    def paired_policy_episodes_per_proposal(self) -> int:
        return 2 * self.episodes_per_batch * self.required_unanimous_batches


def verify_familywise_promotion(
    batch_advantages: list[float],
    plan: FamilywisePilotPlan,
) -> PilotGateResult:
    """Apply the pre-specified Bonferroni gate to one fixed proposal."""

    required = plan.required_unanimous_batches
    if len(batch_advantages) != required:
        raise ValueError(
            f"familywise gate requires exactly {required} batches; "
            f"received {len(batch_advantages)}"
        )
    return verify_promotion(
        batch_advantages,
        PilotGateParams(
            alpha=plan.local_alpha,
            min_mean_advantage=plan.min_mean_advantage,
            min_nonzero_batches=required,
        ),
    )


def holm_rejections(p_values: dict[str, float], familywise_alpha: float = 0.05) -> set[str]:
    """Return Holm step-down rejections for a complete fixed family."""

    if not p_values:
        return set()
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rejected: set[str] = set()
    family_size = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {p_value}")
        if p_value > familywise_alpha / (family_size - index):
            break
        rejected.add(name)
    return rejected
