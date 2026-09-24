import pytest
from qualtran.bloqs import basic_gates as bg

from src.field import DLPInstance, FieldSpec
from src.resource_estimate.qualtran_resources import (
    LogicalResources,
    QualtranPhysicalConfig,
    estimate_logical_resources,
    estimate_physical_resources,
)
from src.shor import ShorDLP, make_shor_config


def test_estimate_single_qubit_gates() -> None:
    clifford = estimate_logical_resources(bg.Hadamard())

    assert isinstance(clifford, LogicalResources)
    assert clifford.logical_qubits == 1
    assert clifford.ccz == 0
    assert clifford.t_equiv_default == 0
    assert int(clifford.gates.clifford) == 1

    t_gate = estimate_logical_resources(bg.TGate())

    assert t_gate.logical_qubits == 1
    assert int(t_gate.gates.t) == 1
    assert t_gate.ccz == 0
    assert t_gate.t_equiv_default == 1


def test_ccz_counts_toffoli_cswap_and() -> None:
    toffoli = estimate_logical_resources(bg.Toffoli())

    assert toffoli.logical_qubits == 3
    assert int(toffoli.gates.toffoli) == 1
    assert toffoli.ccz == 1
    assert toffoli.t_equiv_default == 4


def test_t_equiv_default_uses_qualtran_cost_model() -> None:
    rotation = estimate_logical_resources(bg.Rz(0.5))

    assert int(rotation.gates.rotation) == 1
    assert rotation.ccz == 0
    # Qualtran default: Rotation = 11 T
    assert rotation.t_equiv_default == 11


def test_t_equiv_matches_weighted_sum() -> None:
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    instance = DLPInstance(spec=spec, q=2, g=(4,), h=(1,))
    bloq = ShorDLP(instance=instance, config=make_shor_config(instance))
    resource = estimate_logical_resources(bloq)

    assert resource.logical_qubits > 0
    expected_ccz = (
        int(resource.gates.toffoli) + int(resource.gates.cswap) + int(resource.gates.and_bloq)
    )
    assert resource.ccz == expected_ccz
    expected_t_equiv = int(resource.gates.t) + 4 * expected_ccz + 11 * int(resource.gates.rotation)
    assert resource.t_equiv_default == expected_t_equiv


def _toy_logical() -> LogicalResources:
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    instance = DLPInstance(spec=spec, q=2, g=(4,), h=(1,))
    return estimate_logical_resources(ShorDLP(instance=instance, config=make_shor_config(instance)))


def test_physical_fast_block_config() -> None:
    config = QualtranPhysicalConfig(
        physical_error_rate=1e-3,
        cycle_time_us=0.01,
        data_d=7,
        data_block="fast",
        distillation_d_x=7,
        distillation_d_z=7,
        distillation_d_m=7,
        qec_scheme="gidney_fowler",
    )
    res = estimate_physical_resources(_toy_logical(), config)

    assert res.code_distance == 7
    assert res.physical_qubits > 0
    assert res.n_cycles > 0


def test_physical_block_space_ordering() -> None:
    logical = _toy_logical()
    phys = {
        block: estimate_physical_resources(
            logical,
            QualtranPhysicalConfig(
                physical_error_rate=1e-3,
                cycle_time_us=0.01,
                data_d=7,
                data_block=block,
                distillation_d_x=7,
                distillation_d_z=7,
                distillation_d_m=7,
                qec_scheme="gidney_fowler",
            ),
        ).physical_qubits
        for block in ("compact", "intermediate", "fast")
    }

    assert phys["compact"] < phys["intermediate"] < phys["fast"]


def test_physical_distillation_is_required() -> None:
    with pytest.raises(TypeError):
        QualtranPhysicalConfig(physical_error_rate=1e-3, cycle_time_us=0.01, data_d=7)


def test_physical_rejects_bad_block_and_distillation() -> None:
    with pytest.raises(ValueError, match="data_block"):
        QualtranPhysicalConfig(
            physical_error_rate=1e-3,
            cycle_time_us=0.01,
            data_d=7,
            data_block="simple",
            distillation_d_x=7,
            distillation_d_z=7,
            distillation_d_m=7,
            qec_scheme="gidney_fowler",
        )
    with pytest.raises(ValueError, match="distillation_d_x"):
        QualtranPhysicalConfig(
            physical_error_rate=1e-3,
            cycle_time_us=0.01,
            data_d=7,
            data_block="fast",
            distillation_d_x=0,
            distillation_d_z=7,
            distillation_d_m=7,
            qec_scheme="gidney_fowler",
        )
    with pytest.raises(ValueError, match="d_x"):
        estimate_physical_resources(
            _toy_logical(),
            QualtranPhysicalConfig(
                physical_error_rate=1e-3,
                cycle_time_us=0.01,
                data_d=7,
                data_block="fast",
                distillation_d_x=7,
                distillation_d_z=7,
                distillation_d_m=2,
                qec_scheme="gidney_fowler",
            ),
        )


def test_physical_qec_scheme_switch_changes_error() -> None:
    logical = _toy_logical()
    base = dict(
        physical_error_rate=1e-3,
        cycle_time_us=0.01,
        data_d=7,
        data_block="fast",
        distillation_d_x=7,
        distillation_d_z=7,
        distillation_d_m=7,
        qec_scheme="gidney_fowler",
    )
    gidney = estimate_physical_resources(logical, QualtranPhysicalConfig(**base))
    beverland = estimate_physical_resources(
        logical, QualtranPhysicalConfig(**{**base, "qec_scheme": "beverland"})
    )

    assert gidney.physical_qubits == beverland.physical_qubits
    assert beverland.failure_prob < gidney.failure_prob
    with pytest.raises(ValueError, match="qec_scheme"):
        QualtranPhysicalConfig(**{**base, "qec_scheme": "surface_gamma"})
