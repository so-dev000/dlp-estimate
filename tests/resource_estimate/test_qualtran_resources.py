from qualtran.bloqs import basic_gates as bg

from src.field import DLPInstance, FieldSpec
from src.resource_estimate.qualtran_resources import LogicalResources, estimate_resources
from src.shor import ShorDLP, make_shor_config


def test_estimate_single_qubit_gates() -> None:
    clifford = estimate_resources(bg.Hadamard())

    assert isinstance(clifford, LogicalResources)
    assert clifford.logical_qubits == 1
    assert clifford.ccz == 0
    assert clifford.t_equiv_default == 0
    assert int(clifford.gates.clifford) == 1

    t_gate = estimate_resources(bg.TGate())

    assert t_gate.logical_qubits == 1
    assert int(t_gate.gates.t) == 1
    assert t_gate.ccz == 0
    assert t_gate.t_equiv_default == 1


def test_ccz_counts_toffoli_cswap_and() -> None:
    toffoli = estimate_resources(bg.Toffoli())

    assert toffoli.logical_qubits == 3
    assert int(toffoli.gates.toffoli) == 1
    assert toffoli.ccz == 1
    assert toffoli.t_equiv_default == 4


def test_t_equiv_default_uses_qualtran_cost_model() -> None:
    rotation = estimate_resources(bg.Rz(0.5))

    assert int(rotation.gates.rotation) == 1
    assert rotation.ccz == 0
    # Qualtran default: Rotation = 11 T
    assert rotation.t_equiv_default == 11


def test_t_equiv_matches_weighted_sum() -> None:
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    instance = DLPInstance(spec=spec, q=2, g=(4,), h=(1,))
    bloq = ShorDLP(instance=instance, config=make_shor_config(instance))
    resource = estimate_resources(bloq)

    assert resource.logical_qubits > 0
    expected_ccz = (
        int(resource.gates.toffoli) + int(resource.gates.cswap) + int(resource.gates.and_bloq)
    )
    assert resource.ccz == expected_ccz
    expected_t_equiv = int(resource.gates.t) + 4 * expected_ccz + 11 * int(resource.gates.rotation)
    assert resource.t_equiv_default == expected_t_equiv
