"""Independent pilot gate for task-conditional policy promotions."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class PilotGateParams:
    alpha: float = 0.01
    min_mean_advantage: float = 0.0
    min_nonzero_batches: int = 7


@dataclass(frozen=True)
class PilotGateResult:
    accepted: bool
    mean_advantage: float
    sign_test_p: float
    positive_batches: int
    negative_batches: int
    zero_batches: int
    reason: str


def one_sided_sign_test_p(positive: int, negative: int) -> float:
    """Exact upper-tail sign-test p-value after discarding ties."""

    n = positive + negative
    if n <= 0:
        return 1.0
    return sum(comb(n, successes) for successes in range(positive, n + 1)) / (2**n)


def verify_promotion(batch_advantages: list[float], params: PilotGateParams = PilotGateParams()) -> PilotGateResult:
    if not batch_advantages:
        raise ValueError("pilot verification requires batch advantages")
    positive = sum(value > 0.0 for value in batch_advantages)
    negative = sum(value < 0.0 for value in batch_advantages)
    zeros = len(batch_advantages) - positive - negative
    nonzero = positive + negative
    mean_advantage = sum(batch_advantages) / len(batch_advantages)
    sign_p = one_sided_sign_test_p(positive, negative)
    if nonzero < params.min_nonzero_batches:
        reason = "insufficient_nonzero_pilot_batches"
        accepted = False
    elif mean_advantage <= params.min_mean_advantage:
        reason = "pilot_mean_below_margin"
        accepted = False
    elif sign_p > params.alpha:
        reason = "pilot_sign_test_not_significant"
        accepted = False
    else:
        reason = "pilot_verified_positive_advantage"
        accepted = True
    return PilotGateResult(
        accepted=accepted,
        mean_advantage=mean_advantage,
        sign_test_p=sign_p,
        positive_batches=positive,
        negative_batches=negative,
        zero_batches=zeros,
        reason=reason,
    )
