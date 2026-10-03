import math
from collections.abc import Iterator

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import LogicalErrorModel
from tqdm import tqdm

from ..success import repetitions_for_failure, run_success_lower_bound
from .logical import LogicalResources, MagicStateDemand
from .physical import (
    DataBlockName,
    FactoryName,
    PhysicalDistances,
    QualtranPhysicalConfig,
    QualtranPhysicalResources,
    estimate_physical_resources,
    make_factory,
    make_qec_scheme,
    to_algorithm_summary,
)

SEARCH_DATA_DS = tuple(range(3, 32, 2))
FIFTEEN_TO_ONE_DXS = tuple(range(3, 34, 2))
FIFTEEN_TO_ONE_DZS = tuple(range(3, 28, 2))
FIFTEEN_TO_ONE_DMS = tuple(range(3, 22, 2))
CCZ2T_L1_DS = tuple(range(5, 28, 2))
CCZ2T_L2_DS = tuple(range(11, 48, 2))
SWEPT_DATA_BLOCKS: tuple[DataBlockName, ...] = ("compact", "intermediate", "fast")

FactoryDists = tuple[int, ...]
CandidateKey = tuple[int, ...]

_DATA_BLOCK_ORDER = {name: i for i, name in enumerate(SWEPT_DATA_BLOCKS)}


def _distance_kwargs(factory: FactoryName, dists: FactoryDists) -> dict[str, int]:
    if factory == "ccz2t":
        l1_d, l2_d = dists
        return {"ccz2t_l1_d": l1_d, "ccz2t_l2_d": l2_d}
    d_x, d_z, d_m = dists
    return {"fifteen_to_one_dx": d_x, "fifteen_to_one_dz": d_z, "fifteen_to_one_dm": d_m}


def _iter_factory_distances(factory: FactoryName) -> Iterator[FactoryDists]:
    if factory == "ccz2t":
        for l1_d in CCZ2T_L1_DS:
            for l2_d in CCZ2T_L2_DS:
                yield (l1_d, l2_d)
        return
    for d_x in FIFTEEN_TO_ONE_DXS:
        for d_z in FIFTEEN_TO_ONE_DZS:
            for d_m in FIFTEEN_TO_ONE_DMS:
                if d_x <= 3 * d_m:
                    yield (d_x, d_z, d_m)


def _factory_error_exceeds_budget(
    factory: FactoryName,
    dists: FactoryDists,
    gates: GateCounts,
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
        GateCounts(t=int(gates.t), toffoli=int(gates.toffoli)), error_model
    )
    return not math.isfinite(error) or error > config.factory_error_budget


def _meets_error_budgets(
    resources: QualtranPhysicalResources, config: QualtranPhysicalConfig
) -> bool:
    return (
        resources.physical_failure_prob <= config.physical_failure_threshold
        and resources.factory_error <= config.factory_error_budget
        and resources.data_error <= config.data_error_budget
    )


def _repetitions_for_candidate(
    p_alg: float,
    synthesis_error_budget: float,
    factory_error: float,
    data_error: float,
    final_failure_threshold: float,
) -> int | None:
    try:
        p_run = run_success_lower_bound(p_alg, synthesis_error_budget, factory_error, data_error)
        return repetitions_for_failure(p_run, final_failure_threshold)
    except ValueError:
        return None


def _candidate_key(
    distances: PhysicalDistances,
    resources: QualtranPhysicalResources,
    repetitions: int,
) -> CandidateKey:
    """総 space-time volume n_phys * R * C_run を主キーにする。"""
    per_shot = resources.physical_qubits * resources.n_cycles
    return (
        per_shot * repetitions,
        per_shot,
        repetitions,
        resources.physical_qubits,
        resources.n_cycles,
        _DATA_BLOCK_ORDER[distances.data_block],
        distances.data_d,
        resources.factory_physical_qubits,
        resources.data_physical_qubits,
    )


def _iter_data_candidates(factory: FactoryName, dists: FactoryDists) -> Iterator[PhysicalDistances]:
    kwargs = _distance_kwargs(factory, dists)
    for data_block in SWEPT_DATA_BLOCKS:
        for data_d in SEARCH_DATA_DS:
            yield PhysicalDistances(factory=factory, data_block=data_block, data_d=data_d, **kwargs)


def search_physical_configuration(
    logical: LogicalResources,
    demand: MagicStateDemand,
    config: QualtranPhysicalConfig,
    *,
    p_alg: float,
    synthesis_error_budget: float,
    final_failure_threshold: float,
) -> tuple[PhysicalDistances, QualtranPhysicalResources]:
    gates = to_algorithm_summary(logical, demand, config.factory).n_logical_gates

    best: tuple[PhysicalDistances, QualtranPhysicalResources] | None = None
    best_key: CandidateKey | None = None
    best_physical_failure = math.inf
    n_evaluated = 0

    factory_dists = list(_iter_factory_distances(config.factory))
    n_data = len(SWEPT_DATA_BLOCKS) * len(SEARCH_DATA_DS)
    pbar = tqdm(
        total=len(factory_dists) * n_data,
        desc="physical search",
        unit="cand",
        position=1,
        leave=False,
    )
    for dists in factory_dists:
        if _factory_error_exceeds_budget(config.factory, dists, gates, config):
            pbar.update(n_data)
            continue
        for distances in _iter_data_candidates(config.factory, dists):
            try:
                resources = estimate_physical_resources(logical, demand, config, distances)
            except OverflowError:
                continue
            finally:
                pbar.update(1)
            n_evaluated += 1
            best_physical_failure = min(best_physical_failure, resources.physical_failure_prob)
            if not _meets_error_budgets(resources, config):
                continue
            repetitions = _repetitions_for_candidate(
                p_alg,
                synthesis_error_budget,
                resources.factory_error,
                resources.data_error,
                final_failure_threshold,
            )
            if repetitions is None:
                continue
            key = _candidate_key(distances, resources, repetitions)
            if best_key is None or key < best_key:
                best, best_key = (distances, resources), key
    pbar.close()

    if best is None:
        raise ValueError(
            "no configuration satisfies physical_failure_prob <="
            f" {config.physical_failure_threshold:g} with factory_error <="
            f" {config.factory_error_budget:g} and data_error <="
            f" {config.data_error_budget:g} (evaluated {n_evaluated} candidates,"
            f" best physical_failure_prob {best_physical_failure:g})"
        )
    return best
