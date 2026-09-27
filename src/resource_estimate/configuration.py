import math
from collections.abc import Iterator

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import LogicalErrorModel

from .logical import LogicalResources, MagicStateDemand
from .physical import (
    FactoryName,
    PhysicalDistances,
    QualtranPhysicalConfig,
    QualtranPhysicalResources,
    estimate_physical_resources,
    make_factory,
    make_qec_scheme,
    to_algorithm_summary,
)
from .success import repetitions_for_failure, run_success_lower_bound

SEARCH_DATA_DS = tuple(range(3, 62, 2))
FIFTEEN_TO_ONE_DXS = tuple(range(9, 28, 2))
FIFTEEN_TO_ONE_DZS = tuple(range(7, 24, 2))
FIFTEEN_TO_ONE_DMS = tuple(range(5, 18, 2))
CCZ2T_L1_DS = tuple(range(11, 24, 2))
CCZ2T_L2_DS = tuple(range(23, 40, 2))
SWEPT_DATA_BLOCKS = ("compact", "intermediate", "fast")


def _distance_kwargs(factory: FactoryName, dists: tuple[int, ...]) -> dict[str, int]:
    if factory == "ccz2t":
        l1_d, l2_d = dists
        return {"ccz2t_l1_d": l1_d, "ccz2t_l2_d": l2_d}
    d_x, d_z, d_m = dists
    return {"fifteen_to_one_dx": d_x, "fifteen_to_one_dz": d_z, "fifteen_to_one_dm": d_m}


def _iter_distance_candidates(factory: FactoryName) -> Iterator[tuple[int, ...]]:
    if factory == "ccz2t":
        for l1_d in CCZ2T_L1_DS:
            for l2_d in CCZ2T_L2_DS:
                yield (l1_d, l2_d)
    else:
        for d_x in FIFTEEN_TO_ONE_DXS:
            for d_z in FIFTEEN_TO_ONE_DZS:
                for d_m in FIFTEEN_TO_ONE_DMS:
                    if d_x <= 3 * d_m:
                        yield (d_x, d_z, d_m)


def _factory_error_exceeds_budget(
    factory: FactoryName,
    dists: tuple[int, ...],
    gates_n_t: int,
    gates_n_ccz: int,
    config: QualtranPhysicalConfig,
) -> bool:
    probe = PhysicalDistances(
        factory=factory,
        data_block=SWEPT_DATA_BLOCKS[0],
        data_d=SEARCH_DATA_DS[0],
        **_distance_kwargs(factory, dists),
    )
    error_model = LogicalErrorModel(
        qec_scheme=make_qec_scheme(config.qec_scheme), physical_error=config.physical_error_rate
    )
    error = make_factory(probe).factory_error(
        GateCounts(t=gates_n_t, toffoli=gates_n_ccz), error_model
    )
    return not math.isfinite(error) or error > config.factory_error_budget


def search_physical_configuration(
    logical: LogicalResources,
    demand: MagicStateDemand,
    config: QualtranPhysicalConfig,
    *,
    p_alg: float,
    synthesis_error_budget: float,
    final_failure_threshold: float,
) -> tuple[PhysicalDistances, QualtranPhysicalResources]:
    best: tuple[PhysicalDistances, QualtranPhysicalResources] | None = None
    best_key: tuple[float, int, int, int, int, int, int] | None = None
    best_physical_failure = math.inf
    n_evaluated = 0

    def consider(distances: PhysicalDistances) -> None:
        nonlocal best, best_key, best_physical_failure, n_evaluated
        try:
            resources = estimate_physical_resources(logical, demand, config, distances)
        except OverflowError:
            return
        n_evaluated += 1
        physical_failure = resources.physical_failure_prob
        best_physical_failure = min(best_physical_failure, physical_failure)
        # aggregate だけでなく factory / data を個別にもチェックする
        # (Beverland 流の等分 budget)。
        if not (
            physical_failure <= config.physical_failure_threshold
            and resources.factory_error <= config.factory_error_budget
            and resources.data_error <= config.data_error_budget
        ):
            return
        p_run = run_success_lower_bound(
            p_alg,
            synthesis_error_budget,
            resources.factory_error,
            resources.data_error,
        )
        if p_run <= 0:
            return
        reps = repetitions_for_failure(p_run, final_failure_threshold)
        total_qubit_hours = resources.physical_qubits * resources.duration_hr * reps
        key = (
            total_qubit_hours,
            resources.physical_qubits,
            resources.n_cycles,
            SWEPT_DATA_BLOCKS.index(distances.data_block),
            distances.data_d,
            resources.factory_physical_qubits,
            resources.data_physical_qubits,
        )
        if best is None or (best_key is not None and key < best_key):
            best = (distances, resources)
            best_key = key

    summary = to_algorithm_summary(logical, demand, config.factory)
    gates = summary.n_logical_gates
    for dists in _iter_distance_candidates(config.factory):
        if _factory_error_exceeds_budget(
            config.factory, dists, int(gates.t), int(gates.toffoli), config
        ):
            continue
        for data_block in SWEPT_DATA_BLOCKS:
            for data_d in SEARCH_DATA_DS:
                consider(
                    PhysicalDistances(
                        factory=config.factory,
                        data_block=data_block,
                        data_d=data_d,
                        **_distance_kwargs(config.factory, dists),
                    )
                )
    if best is None:
        raise ValueError(
            "no configuration satisfies physical_failure_prob <="
            f" {config.physical_failure_threshold:g} with factory_error <="
            f" {config.factory_error_budget:g} and data_error <="
            f" {config.data_error_budget:g} (evaluated {n_evaluated} candidates,"
            f" best physical_failure_prob {best_physical_failure:g})"
        )
    return best
