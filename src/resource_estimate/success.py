import math

from ..field import Factorization
from ..validation import validate_probability


def single_run_success_lower_bound(
    q_factors: Factorization,
) -> float:
    """
    Shor-DLP の1-run 成功確率下限 (Mosca bound)
    Mosca, PhD thesis (1999), Corollary 19, p.58:
    回路は Mosca の構成 (n = ceil(log2(2q))+1) を前提とする
    """
    return (64.0 / math.pi**4) * math.prod(1.0 - 1.0 / p for p, _ in q_factors)


def run_success_lower_bound(
    p_alg: float,
    eps_syn: float,
    eps_factory: float,
    eps_data: float,
) -> float:
    """FT 実装誤差込みの 1-run 成功確率下限"""
    p = validate_probability(p_alg, "p_alg", endpoints=True)

    eps_impl = eps_syn + eps_factory + eps_data
    validate_probability(eps_impl, "eps_impl", endpoints=True)

    return max(0.0, p - eps_impl)


def repetitions_for_failure(
    p_single: float,
    failure_threshold: float,
) -> int:
    """(1 - p_single)**R <= failure_threshold を満たす最小 R。"""
    p = validate_probability(p_single, "p_single", endpoints=True)
    threshold = validate_probability(
        failure_threshold,
        "failure_threshold",
    )

    if p == 0:
        raise ValueError("unachievable: p_single is 0")
    if p == 1:
        return 1

    return math.ceil(math.log(threshold) / math.log1p(-p))


def total_failure_upper_bound(p_run: float, repetitions: int) -> float:
    """(1-p_run)**R"""
    p = validate_probability(p_run, "p_run", endpoints=True)
    if type(repetitions) is not int or repetitions < 0:
        raise ValueError("repetitions must be a nonnegative integer")
    if repetitions == 0 or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    return float(math.exp(repetitions * math.log1p(-p)))
