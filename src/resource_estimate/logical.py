from collections.abc import Callable
from dataclasses import dataclass

from qualtran import Adjoint, Bloq, QUInt
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled
from qualtran.bloqs.mod_arithmetic import CModAdd, CModAddK, CtrlScaleModAdd
from qualtran.bloqs.qft import QFTTextBook
from qualtran.resource_counting import (
    GateCounts,
    QECGatesCost,
    QubitCount,
    get_cost_value,
)
from qualtran.surface_code import AlgorithmSummary, beverland_et_al_model
from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost
from qualtran.symbolics import SymbolicInt

from ..validation import validate_probability


@dataclass(frozen=True, kw_only=True)
class LogicalResources:
    logical_qubits: int
    t: int
    toffoli: int
    cswap: int
    and_bloq: int
    clifford: int
    rotation: int
    measurement: int


class FeedForwardQECGatesCost(QECGatesCost):
    def compute(
        self,
        bloq: Bloq,
        get_callee_cost: Callable[[Bloq], GateCounts],
    ) -> GateCounts:
        if isinstance(bloq, ClassicallyControlled):
            return get_callee_cost(bloq.subbloq)
        return super().compute(bloq, get_callee_cost)


class FastQubitCount(QubitCount):
    def compute(
        self,
        bloq: Bloq,
        get_callee_cost: Callable[[Bloq], SymbolicInt],
    ) -> SymbolicInt:
        if isinstance(bloq, (IntState, IntEffect)):
            return bloq.bitsize

        if isinstance(bloq, QFTTextBook):
            return bloq.bitsize

        if isinstance(bloq, Adjoint):
            return get_callee_cost(bloq.subbloq)

        if isinstance(bloq, CModAddK):
            return get_callee_cost(
                CModAdd(
                    QUInt(bloq.bitsize),
                    mod=bloq.mod,
                )
            )

        if isinstance(bloq, CtrlScaleModAdd):
            return (
                get_callee_cost(
                    CModAdd(
                        QUInt(bloq.bitsize),
                        mod=bloq.mod,
                    )
                )
                + bloq.bitsize
                + 1
            )

        if isinstance(bloq, ClassicallyControlled):
            return get_callee_cost(bloq.subbloq)

        return super().compute(bloq, get_callee_cost)


def estimate_logical_resources(bloq: Bloq) -> LogicalResources:
    gates = get_cost_value(
        bloq,
        FeedForwardQECGatesCost(legacy_shims=False),
    )

    logical_qubits = int(
        get_cost_value(
            bloq,
            FastQubitCount(),
        )
    )

    return LogicalResources(
        logical_qubits=logical_qubits,
        t=int(gates.t),
        toffoli=int(gates.toffoli),
        cswap=int(gates.cswap),
        and_bloq=int(gates.and_bloq),
        clifford=int(gates.clifford),
        rotation=int(gates.rotation),
        measurement=int(gates.measurement),
    )


def synthesize_rotations(
    logical: LogicalResources,
    synthesis_error_budget: float,
) -> GateCounts:
    validate_probability(synthesis_error_budget, "synthesis_error_budget")
    alg = AlgorithmSummary(
        n_algo_qubits=logical.logical_qubits,
        n_logical_gates=GateCounts(
            t=logical.t,
            toffoli=logical.toffoli,
            cswap=logical.cswap,
            and_bloq=logical.and_bloq,
            clifford=logical.clifford,
            rotation=logical.rotation,
            measurement=logical.measurement,
        ),
    )

    gates = beverland_et_al_model.n_discrete_logical_gates(
        eps_syn=synthesis_error_budget,
        alg=alg,
        rotation_model=BeverlandEtAlRotationCost,
    )

    assert gates.rotation == 0
    return gates


@dataclass(frozen=True, kw_only=True)
class MagicStateDemand:
    n_t_states: int
    n_ccz_states: int


def estimate_magic_state_demand(discrete_gates: GateCounts) -> MagicStateDemand:
    assert discrete_gates.rotation == 0
    counts = discrete_gates.total_t_and_ccz_count(ts_per_rotation=0)
    return MagicStateDemand(
        n_t_states=int(counts["n_t"]),
        n_ccz_states=int(counts["n_ccz"]),
    )
