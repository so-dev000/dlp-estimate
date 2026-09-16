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


def estimate_resources(bloq: Bloq) -> LogicalResources:
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
