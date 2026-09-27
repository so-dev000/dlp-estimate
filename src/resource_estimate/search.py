import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from ..field import DLPInstance, Factorization, FieldElement, FieldSpec, PolynomialCoefficients
from ..shor import ShorDLP
from ..validation import validate_probability
from .configuration import search_physical_configuration
from .logical import (
    estimate_logical_resources,
    estimate_magic_state_demand,
    synthesize_rotations,
)
from .physical import QualtranPhysicalConfig
from .success import (
    repetitions_for_failure,
    run_success_lower_bound,
    single_run_success_lower_bound,
    total_failure_upper_bound,
)


@dataclass(frozen=True, kw_only=True)
class Params:
    """検証済みの DLP 入力。eval_point では重い体・位数・部分群検証を行わない。"""

    label: str = ""
    p: int
    r: int
    f: PolynomialCoefficients
    q: int
    q_factors: Factorization | None = None
    g: FieldElement
    h: FieldElement


class EvalRow(TypedDict):
    label: str
    field_bits: int  # ceil(log2(p^r))
    # Shor configuration
    exponent_bits: int  # 2つの指数レジスタ共通幅 Q=2**m > q の最小値
    # Physical-model inputs
    logical_qubits: int  # n_algo_qubits ancillaを含む最大同時使用数
    n_t_states: int  # T demang (native + 回転合成)
    n_ccz_states: int  # CCZ demand (toffoli + cswap + and_bloq)
    # Per-point choices
    data_block: str  # 探索で選ばれた data_block
    code_distance: int  # 探索で選ばれた data_d
    fifteen_to_one_dx: NotRequired[int]  # fifteen_to_one 行のみ
    fifteen_to_one_dz: NotRequired[int]  # fifteen_to_one 行のみ
    fifteen_to_one_dm: NotRequired[int]  # fifteen_to_one 行のみ
    ccz2t_l1_d: NotRequired[int]  # ccz2t 行のみ
    ccz2t_l2_d: NotRequired[int]  # ccz2t 行のみ
    # Physical qubits
    physical_qubits: int  # factory + data
    factory_physical_qubits: int  # factory 側の物理 qubit 数
    data_physical_qubits: int  # data 側の物理 qubit 数
    # QEC cycles
    n_cycles: int  # 単発の QEC cycle 数
    n_generation_cycles: int  # magic state生成律速の cycle 数
    n_consumption_cycles: int  # magic state消費律速の cycle 数
    # Runtime
    duration_hr: float  # 単発の実行時間 [時間]
    # Physical failure
    physical_failure_prob: float  # factory_error + data_error
    factory_error: float  # magic state factory 由来の単発失敗率
    data_error: float  # data block 由来の単発失敗率
    implementation_error: float  # eps_syn + factory_error + data_error (単発の実装誤差)
    # Algorithm success: p_alg (理想下限) と p_run (実装誤差込み下限)。
    # R は p_run と最終目標 delta_final から一本化して決める。
    single_run_success: float  # 16/pi^4 * phi(q)/q (因数分解不明時はさらに保守的な下限)
    combined_single_run_success: float  # max(0, p_alg - eps_impl) [1-shot 下限 p_run]
    repetitions: int  # (1-p_run)^R <= delta_final の最小R
    total_failure_prob: float  # (1-p_run)^R [R 回全体の上限]
    duration_hr_total: float  # duration_hr * R (逐次実行)
    n_cycles_total: int  # n_cycles * R (逐次実行)
    # Error message
    error: NotRequired[str]


def eval_point(
    params: Params,
    physical_config: QualtranPhysicalConfig,
    final_failure_threshold: float,
    synthesis_error_budget: float,
) -> EvalRow:
    validate_probability(final_failure_threshold, "final_failure_threshold")
    validate_probability(synthesis_error_budget, "synthesis_error_budget")
    spec = FieldSpec(p=params.p, r=params.r, f=params.f)
    instance = DLPInstance(
        spec=spec, q=params.q, q_factors=params.q_factors, g=params.g, h=params.h
    )
    shor = ShorDLP(instance=instance)
    logical = estimate_logical_resources(shor)
    discrete_gates = synthesize_rotations(logical, synthesis_error_budget)
    demand = estimate_magic_state_demand(discrete_gates)
    if instance.q_factors is None:
        raise ValueError("q_factors is required to bound single-run success")
    p_alg = single_run_success_lower_bound(instance.q_factors)
    # 失敗行にも、固定 factory に実際に渡す需要を記録する。
    n_t, n_ccz = demand.n_t_states, demand.n_ccz_states
    if physical_config.factory == "fifteen_to_one":
        n_t, n_ccz = n_t + 4 * n_ccz, 0
    base: dict[str, Any] = {
        "label": params.label,
        "field_bits": spec.field_bits,
        "exponent_bits": shor.exponent_bits,
        "logical_qubits": logical.logical_qubits,
        "n_t_states": n_t,
        "n_ccz_states": n_ccz,
        "single_run_success": p_alg,
    }
    try:
        _, physical = search_physical_configuration(
            logical,
            demand,
            physical_config,
        )
    except Exception as e:
        return EvalRow(base | {"error": f"{type(e).__name__}: {e}"})
    physical_row: dict[str, Any] = {
        "data_block": physical.data_block,
        "code_distance": physical.code_distance,
    }
    if physical.factory == "fifteen_to_one":
        physical_row |= {
            "fifteen_to_one_dx": physical.fifteen_to_one_dx,
            "fifteen_to_one_dz": physical.fifteen_to_one_dz,
            "fifteen_to_one_dm": physical.fifteen_to_one_dm,
        }
    else:
        physical_row |= {
            "ccz2t_l1_d": physical.ccz2t_l1_d,
            "ccz2t_l2_d": physical.ccz2t_l2_d,
        }
    physical_row |= {
        "physical_qubits": physical.physical_qubits,
        "factory_physical_qubits": physical.factory_physical_qubits,
        "data_physical_qubits": physical.data_physical_qubits,
        "n_cycles": physical.n_cycles,
        "n_generation_cycles": physical.n_generation_cycles,
        "n_consumption_cycles": physical.n_consumption_cycles,
        "duration_hr": physical.duration_hr,
        "physical_failure_prob": physical.physical_failure_prob,
        "factory_error": physical.factory_error,
        "data_error": physical.data_error,
    }
    p_run = run_success_lower_bound(
        p_alg,
        synthesis_error_budget,
        physical.factory_error,
        physical.data_error,
    )
    implementation_error = synthesis_error_budget + physical.factory_error + physical.data_error
    try:
        repetitions = repetitions_for_failure(p_run, final_failure_threshold)
    except Exception as e:
        return EvalRow(
            base
            | physical_row
            | {
                "implementation_error": implementation_error,
                "combined_single_run_success": p_run,
                "error": f"{type(e).__name__}: {e}",
            }
        )
    return EvalRow(
        base
        | physical_row
        | {
            "implementation_error": implementation_error,
            "combined_single_run_success": p_run,
            "repetitions": repetitions,
            "total_failure_prob": total_failure_upper_bound(p_run, repetitions),
            # 逐次R回実行の総費用。qubitsはR倍しない。
            "duration_hr_total": physical.duration_hr * repetitions,
            "n_cycles_total": physical.n_cycles * repetitions,
        }
    )


def sweep(
    points: list[Params],
    physical_config: QualtranPhysicalConfig,
    final_failure_threshold: float,
    on_row: Callable[[dict[str, Any]], None],
    synthesis_error_budget: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for params in points:
        try:
            row: dict[str, Any] = dict(
                eval_point(params, physical_config, final_failure_threshold, synthesis_error_budget)
            )
            print(f"Evaluated: {params.label}")
        except Exception as e:
            row = {
                "label": params.label,
                "error": f"{type(e).__name__}: {e}",
            }
            with contextlib.suppress(Exception):
                row["field_bits"] = FieldSpec(
                    p=params.p,
                    r=params.r,
                    f=params.f,
                ).field_bits
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows
