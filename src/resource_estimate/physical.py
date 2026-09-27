from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import (
    AlgorithmSummary,
    CompactDataBlock,
    DataBlock,
    FastDataBlock,
    FifteenToOne,
    IntermediateDataBlock,
    MagicStateFactory,
    PhysicalCostModel,
    PhysicalParameters,
    QECScheme,
)
from qualtran.surface_code.ccz2t_factory import CCZ2TFactory

from ..validation import validate_probability
from .logical import LogicalResources, MagicStateDemand

FactoryName = Literal["fifteen_to_one", "ccz2t"]
DataBlockName = Literal["fast", "intermediate", "compact"]


@dataclass(frozen=True, kw_only=True)
class QualtranPhysicalConfig:
    physical_error_rate: float
    cycle_time_us: float
    qec_scheme: Literal["gidney_fowler", "beverland"]
    physical_failure_threshold: float
    factory_error_budget: float
    data_error_budget: float
    factory: FactoryName

    def __post_init__(self) -> None:
        validate_probability(self.physical_error_rate, "physical_error_rate")
        if self.physical_error_rate >= 0.01:
            raise ValueError("physical_error_rate must be below the QEC model threshold 0.01")
        if isinstance(self.cycle_time_us, bool) or not isinstance(self.cycle_time_us, (int, float)):
            raise ValueError("cycle_time_us must be a number")
        if not math.isfinite(self.cycle_time_us) or not (self.cycle_time_us > 0.0):
            raise ValueError("cycle_time_us must be positive")
        if self.qec_scheme not in ("gidney_fowler", "beverland"):
            raise ValueError("qec_scheme must be gidney_fowler or beverland")
        validate_probability(self.physical_failure_threshold, "physical_failure_threshold")
        validate_probability(self.factory_error_budget, "factory_error_budget")
        validate_probability(self.data_error_budget, "data_error_budget")
        if self.factory not in ("fifteen_to_one", "ccz2t"):
            raise ValueError("factory must be fifteen_to_one or ccz2t")


@dataclass(frozen=True, kw_only=True)
class PhysicalDistances:
    factory: FactoryName
    data_block: DataBlockName
    data_d: int
    fifteen_to_one_dx: int | None = None
    fifteen_to_one_dz: int | None = None
    fifteen_to_one_dm: int | None = None
    ccz2t_l1_d: int | None = None
    ccz2t_l2_d: int | None = None

    def __post_init__(self) -> None:
        if self.factory not in ("fifteen_to_one", "ccz2t"):
            raise ValueError("factory must be fifteen_to_one or ccz2t")
        if self.data_block not in ("fast", "intermediate", "compact"):
            raise ValueError("data_block must be fast, intermediate, or compact")
        if isinstance(self.data_d, bool) or not isinstance(self.data_d, int):
            raise ValueError("data_d must be an int")
        if self.data_d < 2:
            raise ValueError("data_d must be >= 2")
        for name in (
            "fifteen_to_one_dx",
            "fifteen_to_one_dz",
            "fifteen_to_one_dm",
            "ccz2t_l1_d",
            "ccz2t_l2_d",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, kw_only=True)
class QualtranPhysicalResources:
    physical_qubits: int
    duration_hr: float
    physical_failure_prob: float
    n_cycles: int
    code_distance: int
    fifteen_to_one_dx: int | None
    fifteen_to_one_dz: int | None
    fifteen_to_one_dm: int | None
    ccz2t_l1_d: int | None
    ccz2t_l2_d: int | None
    factory_error: float
    data_error: float
    n_generation_cycles: int
    n_consumption_cycles: int
    factory_physical_qubits: int
    data_physical_qubits: int
    factory: str
    data_block: str


def to_algorithm_summary(
    logical: LogicalResources,
    demand: MagicStateDemand,
    factory: FactoryName,
) -> AlgorithmSummary:
    n_t = demand.n_t_states
    n_ccz = demand.n_ccz_states
    if factory == "fifteen_to_one":
        n_t, n_ccz = n_t + 4 * n_ccz, 0
    if n_t + n_ccz == 0:
        raise ValueError("magic-state throughput model requires non-Clifford gates")
    return AlgorithmSummary(
        n_algo_qubits=logical.logical_qubits,
        n_logical_gates=GateCounts(t=n_t, toffoli=n_ccz),
    )


def make_data_block(data_block: DataBlockName, data_d: int) -> DataBlock:
    match data_block:
        case "fast":
            return FastDataBlock(data_d=data_d)
        case "intermediate":
            return IntermediateDataBlock(data_d=data_d)
        case "compact":
            return CompactDataBlock(data_d=data_d)


def make_factory(distances: PhysicalDistances) -> MagicStateFactory:
    match distances.factory:
        case "fifteen_to_one":
            d_x, d_z, d_m = (
                distances.fifteen_to_one_dx,
                distances.fifteen_to_one_dz,
                distances.fifteen_to_one_dm,
            )
            if distances.ccz2t_l1_d is not None or distances.ccz2t_l2_d is not None:
                raise ValueError("ccz2t distances are unused by fifteen_to_one")
            if d_x is None or d_z is None or d_m is None:
                raise ValueError("fifteen_to_one requires dx, dz, dm")
            if not 0 < d_x <= 3 * d_m:
                raise ValueError("require 0 < d_x <= 3 * d_m")
            return FifteenToOne(d_X=d_x, d_Z=d_z, d_m=d_m)
        case "ccz2t":
            if any(
                value is not None
                for value in (
                    distances.fifteen_to_one_dx,
                    distances.fifteen_to_one_dz,
                    distances.fifteen_to_one_dm,
                )
            ):
                raise ValueError("fifteen_to_one distances are unused by ccz2t")
            if distances.ccz2t_l1_d is None or distances.ccz2t_l2_d is None:
                raise ValueError("ccz2t requires l1_d, l2_d")
            return CCZ2TFactory(
                distillation_l1_d=distances.ccz2t_l1_d,
                distillation_l2_d=distances.ccz2t_l2_d,
            )


def make_qec_scheme(name: Literal["gidney_fowler", "beverland"]) -> QECScheme:
    match name:
        case "gidney_fowler":
            return QECScheme.make_gidney_fowler()
        case "beverland":
            return QECScheme.make_beverland_et_al()


def estimate_physical_resources(
    logical: LogicalResources,
    demand: MagicStateDemand,
    config: QualtranPhysicalConfig,
    distances: PhysicalDistances,
) -> QualtranPhysicalResources:
    summary = to_algorithm_summary(logical, demand, distances.factory)
    model = PhysicalCostModel(
        physical_params=PhysicalParameters(
            physical_error=config.physical_error_rate,
            cycle_time_us=config.cycle_time_us,
        ),
        data_block=make_data_block(distances.data_block, distances.data_d),
        factory=make_factory(distances),
        qec_scheme=make_qec_scheme(config.qec_scheme),
    )
    logical_error_model = model.logical_error_model
    gates = summary.n_logical_gates
    n_cycles = int(model.n_cycles(summary))
    n_generation_cycles = int(model.factory.n_cycles(gates, logical_error_model))
    n_consumption_cycles = int(model.data_block.n_cycles(gates, logical_error_model))
    factory_error = float(model.factory.factory_error(gates, logical_error_model))
    data_error = float(
        model.data_block.data_error(summary.n_algo_qubits, n_cycles, logical_error_model)
    )
    if any(not math.isfinite(value) or value < 0 for value in (factory_error, data_error)):
        raise ArithmeticError("physical error model returned a non-finite or negative error")
    factory_physical_qubits = int(model.factory.n_physical_qubits())
    data_physical_qubits = int(model.data_block.n_physical_qubits(summary.n_algo_qubits))
    return QualtranPhysicalResources(
        physical_qubits=factory_physical_qubits + data_physical_qubits,
        duration_hr=float(model.duration_hr(summary)),
        physical_failure_prob=factory_error + data_error,
        n_cycles=n_cycles,
        code_distance=distances.data_d,
        fifteen_to_one_dx=distances.fifteen_to_one_dx,
        fifteen_to_one_dz=distances.fifteen_to_one_dz,
        fifteen_to_one_dm=distances.fifteen_to_one_dm,
        ccz2t_l1_d=distances.ccz2t_l1_d,
        ccz2t_l2_d=distances.ccz2t_l2_d,
        factory_error=factory_error,
        data_error=data_error,
        n_generation_cycles=n_generation_cycles,
        n_consumption_cycles=n_consumption_cycles,
        factory_physical_qubits=factory_physical_qubits,
        data_physical_qubits=data_physical_qubits,
        factory=distances.factory,
        data_block=distances.data_block,
    )
