from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from qualtran import Bloq
from qualtran.resource_counting import (
    GateCounts,
    QECGatesCost,
    QubitCount,
    get_cost_value,
)
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


@dataclass(frozen=True)
class LogicalResources:
    # GateCounts.tは回路中に明示的に現れるTゲートの数であることに注意
    gates: GateCounts
    logical_qubits: int

    @property
    def ccz(self) -> int:
        """
        Toffoli + CSwap + And
        """
        counts = self.gates.total_t_and_ccz_count(
            ts_per_rotation=0,
        )
        return int(counts["n_ccz"])

    @property
    def t_equiv_default(self) -> int:
        """
        Qualtran デフォルトの T-equivalent count。

        Toffoli  = 4 T
        CSwap    = 4 T
        And      = 4 T
        Rotation = 11 T
        """
        return int(self.gates.total_t_count())


def estimate_logical_resources(bloq: Bloq) -> LogicalResources:
    gates = get_cost_value(
        bloq,
        QECGatesCost(legacy_shims=False),
    )

    logical_qubits = int(
        get_cost_value(
            bloq,
            QubitCount(),
        )
    )

    return LogicalResources(
        gates=gates,
        logical_qubits=logical_qubits,
    )


@dataclass(frozen=True, kw_only=True)
class QualtranPhysicalConfig:
    physical_error_rate: float
    cycle_time_us: float
    data_d: int
    data_block: Literal["fast", "intermediate", "compact"]
    distillation_d_x: int
    distillation_d_z: int
    distillation_d_m: int
    qec_scheme: Literal["gidney_fowler", "beverland"]

    def __post_init__(self) -> None:
        if isinstance(self.physical_error_rate, bool) or not isinstance(
            self.physical_error_rate, (int, float)
        ):
            raise ValueError("physical_error_rate must be a number")
        if not math.isfinite(self.physical_error_rate) or not (
            0.0 < self.physical_error_rate < 1.0
        ):
            raise ValueError("physical_error_rate must satisfy 0 < p < 1")
        if isinstance(self.cycle_time_us, bool) or not isinstance(self.cycle_time_us, (int, float)):
            raise ValueError("cycle_time_us must be a number")
        if not math.isfinite(self.cycle_time_us) or not (self.cycle_time_us > 0.0):
            raise ValueError("cycle_time_us must be positive")
        if isinstance(self.data_d, bool) or not isinstance(self.data_d, int):
            raise ValueError("data_d must be an int")
        if self.data_d < 2:
            raise ValueError("data_d must be >= 2")
        if self.data_block not in ("fast", "intermediate", "compact"):
            raise ValueError("data_block must be fast, intermediate, or compact")
        if self.qec_scheme not in ("gidney_fowler", "beverland"):
            raise ValueError("qec_scheme must be gidney_fowler or beverland")
        for name in ("distillation_d_x", "distillation_d_z", "distillation_d_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, kw_only=True)
class QualtranPhysicalResources:
    """PhysicalCostModel の正規化結果"""

    physical_qubits: int
    duration_hr: float
    failure_prob: float
    n_cycles: int
    code_distance: int


def to_algorithm_summary(logical: LogicalResources) -> AlgorithmSummary:
    """LogicalResources を AlgorithmSummary へ変換する"""
    return AlgorithmSummary(
        n_algo_qubits=logical.logical_qubits,
        n_logical_gates=logical.gates,
    )


def _make_data_block(config: QualtranPhysicalConfig) -> DataBlock:
    """論理パッチ配置を返す"""
    match config.data_block:
        case "fast":
            return FastDataBlock(data_d=config.data_d)
        case "intermediate":
            return IntermediateDataBlock(data_d=config.data_d)
        case "compact":
            return CompactDataBlock(data_d=config.data_d)


def _make_factory(config: QualtranPhysicalConfig) -> MagicStateFactory:
    """15-to-1 factory を返す。"""
    d_x, d_z, d_m = (
        config.distillation_d_x,
        config.distillation_d_z,
        config.distillation_d_m,
    )
    if not 0 < d_x <= 3 * d_m:
        raise ValueError("require 0 < d_x <= 3 * d_m")
    return FifteenToOne(d_X=d_x, d_Z=d_z, d_m=d_m)


def _make_qec_scheme(config: QualtranPhysicalConfig) -> QECScheme:
    """論理誤り率モデルを返す。"""
    match config.qec_scheme:
        case "gidney_fowler":
            return QECScheme.make_gidney_fowler()
        case "beverland":
            return QECScheme.make_beverland_et_al()


def estimate_physical_resources(
    logical: LogicalResources,
    config: QualtranPhysicalConfig,
) -> QualtranPhysicalResources:
    """論理計数から physical qubits・実行時間・失敗確率を解析的に返す。"""
    summary = to_algorithm_summary(logical)
    model = PhysicalCostModel(
        physical_params=PhysicalParameters(
            physical_error=config.physical_error_rate,
            cycle_time_us=config.cycle_time_us,
        ),
        data_block=_make_data_block(config),
        factory=_make_factory(config),
        qec_scheme=_make_qec_scheme(config),
    )
    return QualtranPhysicalResources(
        physical_qubits=int(model.n_phys_qubits(summary)),
        duration_hr=float(model.duration_hr(summary)),
        failure_prob=float(model.error(summary)),
        n_cycles=int(model.n_cycles(summary)),
        code_distance=config.data_d,
    )
