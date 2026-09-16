from dataclasses import dataclass

from qualtran import Bloq
from qualtran.resource_counting import (
    GateCounts,
    QECGatesCost,
    QubitCount,
    get_cost_value,
)


@dataclass(frozen=True)
class LogicalResources:
    # https://qualtran.readthedocs.io/en/latest/reference/qualtran/resource_counting/GateCounts.html
    gates: GateCounts
    logical_qubits: int


def estimate_resources(bloq: Bloq) -> LogicalResources:
    gates = get_cost_value(
        bloq,
        QECGatesCost(legacy_shims=False),
    )

    logical_qubits = int(
        get_cost_value(
            bloq,
            # https://qualtran.readthedocs.io/en/latest/resource_counting/qubit_counts.html
            QubitCount(),
        )
    )

    return LogicalResources(
        gates=gates,
        logical_qubits=logical_qubits,
    )
