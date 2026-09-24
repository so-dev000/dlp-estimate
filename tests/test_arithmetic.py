from itertools import product

import numpy as np
import pytest
from qualtran import Bloq, Controlled
from qualtran.bloqs.gf_arithmetic import GF2MulK
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
from qualtran.resource_counting._call_graph import get_bloq_callee_counts
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.arithmetic import (
    ControlledConstMul,
    ControlledGF2ConstMul,
    ControlledLinearMapAdd,
    get_controlled_const_mul,
)
from src.field import FieldSpec, FiniteField


def _as_tuple(value: ClassicalValT) -> tuple[int, ...]:
    """call_classically の戻り値(ClassicalValT)を係数tupleに正規化する。"""
    return tuple(int(v) for v in np.asarray(value))


@pytest.fixture
def spec() -> FieldSpec:
    """最小の素体 GF(5)。"""
    return FieldSpec(p=5, r=1, f=(0, 1))


def test_linear_map_add_applies_matrix_only_when_ctrl_is_one(spec: FieldSpec) -> None:
    bloq = ControlledLinearMapAdd(spec=spec, matrix=((2,),))  # y += ctrl * 2 * x

    _, _, y_on = bloq.call_classically(ctrl=1, x=np.array([3]), y=np.array([1]))
    _, _, y_off = bloq.call_classically(ctrl=0, x=np.array([3]), y=np.array([1]))

    assert np.asarray(y_on).tolist() == [2]  # 2*3+1 = 7 = 2 mod 5
    assert np.asarray(y_off).tolist() == [1]


def test_linear_map_add_rejects_bad_matrix(spec: FieldSpec) -> None:
    with pytest.raises(ValueError):
        ControlledLinearMapAdd(spec=spec, matrix=((1, 2), (3, 4)))  # 2x2 は r=1 に不一致
    with pytest.raises(ValueError):
        ControlledLinearMapAdd(spec=spec, matrix=((5,),))  # 係数は [0, p) の範囲外


def test_linear_map_add_decomposition_matches_classical_action(spec: FieldSpec) -> None:
    bloq = ControlledLinearMapAdd(spec=spec, matrix=((2,),))

    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(bloq, ctrl=[0, 1], x=[[1], [3]], y=[[0], [1]])


def test_const_mul_applies_power_only_when_ctrl_is_one(spec: FieldSpec) -> None:
    bloq = ControlledConstMul(spec=spec, c=(2,))  # x -> 2^ctrl * x

    _, x_on = bloq.call_classically(ctrl=1, x=np.array([3]))
    _, x_off = bloq.call_classically(ctrl=0, x=np.array([3]))

    assert np.asarray(x_on).tolist() == [1]  # 2*3 = 6 = 1 mod 5
    assert np.asarray(x_off).tolist() == [3]


def test_const_mul_rejects_zero_and_inverts_via_adjoint(spec: FieldSpec) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        ControlledConstMul(spec=spec, c=(0,))

    assert ControlledConstMul(spec=spec, c=(2,)).adjoint() == ControlledConstMul(
        spec=spec,
        c=(3,),  # 2 * 3 = 1 mod 5
    )


def test_const_mul_decomposition_matches_classical_action(spec: FieldSpec) -> None:
    bloq = ControlledConstMul(spec=spec, c=(2,))

    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(bloq, ctrl=[0, 1], x=[[1], [3]])


@pytest.fixture(params=[(1, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1)], ids=["gf4", "gf8", "gf8-alt"])
def gf2_spec(request: pytest.FixtureRequest) -> FieldSpec:
    return FieldSpec(p=2, r=len(request.param) - 1, f=request.param)


@pytest.mark.parametrize("polynomial", [(0, 1), (1, 1, 1), (1, 0, 1, 1)])
def test_identity_needs_no_operations(polynomial: tuple[int, ...]) -> None:
    spec = FieldSpec(p=2, r=len(polynomial) - 1, f=polynomial)
    bloq = ControlledGF2ConstMul(spec=spec, c=(1,) + (0,) * (spec.r - 1))

    decomposition = assert_valid_bloq_decomposition(bloq)

    assert not decomposition.bloq_instances
    assert get_cost_value(bloq, QubitCount()) == spec.r + 1
    assert_consistent_classical_action(bloq, ctrl=[0, 1], x=list(product((0, 1), repeat=spec.r)))


def test_decomposition_preserves_field_constant_and_inverse(gf2_spec: FieldSpec) -> None:
    spec = gf2_spec
    elements = list(product((0, 1), repeat=spec.r))
    field = FiniteField(spec)
    for constant in elements:
        if constant in (field.zero, field.one):
            continue
        bloq = ControlledGF2ConstMul(spec=spec, c=constant)
        decomposition = assert_valid_bloq_decomposition(bloq)
        multiplications = [
            instance.bloq
            for instance in decomposition.bloq_instances
            if isinstance(instance.bloq, Controlled)
        ]
        assert len(multiplications) == 1
        multiplication = multiplications[0].subbloq
        assert isinstance(multiplication, GF2MulK)
        poly = multiplication.dtype.irreducible_poly
        assert poly is not None
        assert tuple(int(a) for a in reversed(poly.coeffs)) == spec.f
        assert multiplication.const == sum(a << j for j, a in enumerate(constant))

        assert_consistent_classical_action(bloq, ctrl=[0, 1], x=elements)
        # 暗黙の置換を検出するため末端まで分解する。
        fully_decomposed = decomposition.flatten()
        inverse = bloq.adjoint().decompose_bloq().flatten()
        for ctrl, element in product((0, 1), elements):
            ctrl_out, x_out = fully_decomposed.call_classically(ctrl=ctrl, x=np.array(element))
            expected = field.mul(constant, element) if ctrl else element
            assert ctrl_out == ctrl
            assert _as_tuple(x_out) == expected
            ctrl_back, x_back = inverse.call_classically(ctrl=ctrl_out, x=x_out)
            assert ctrl_back == ctrl
            assert _as_tuple(x_back) == element


def test_gf4_multiply_by_x_matches_hand_calculation() -> None:
    bloq = ControlledGF2ConstMul(spec=FieldSpec(p=2, r=2, f=(1, 1, 1)), c=(0, 1))
    decomposition = bloq.decompose_bloq().flatten()

    for ctrl, a, b in product((0, 1), repeat=3):
        ctrl_out, x_out = decomposition.call_classically(ctrl=ctrl, x=np.array([a, b]))
        assert ctrl_out == ctrl
        assert _as_tuple(x_out) == ((b, a ^ b) if ctrl else (a, b))


def test_gf4_controlled_decomposition_preserves_full_unitary() -> None:
    bloq = ControlledGF2ConstMul(spec=FieldSpec(p=2, r=2, f=(1, 1, 1)), c=(0, 1))
    expected = np.zeros((8, 8), dtype=complex)
    for ctrl, a, b in product((0, 1), repeat=3):
        out_a, out_b = (b, a ^ b) if ctrl else (a, b)
        expected[4 * ctrl + 2 * out_a + out_b, 4 * ctrl + 2 * a + b] = 1
    actual = bloq.decompose_bloq().flatten().tensor_contract()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_resources_match_controlled_multiplication_with_explicit_swaps(gf2_spec: FieldSpec) -> None:
    spec = gf2_spec
    bloq = ControlledGF2ConstMul(spec=spec, c=(0, 1) + (0,) * (spec.r - 2))
    standard = next(
        instance.bloq
        for instance in bloq.decompose_bloq().bloq_instances
        if isinstance(instance.bloq, Controlled)
    )

    assert get_cost_value(bloq, QECGatesCost()) == get_cost_value(standard, QECGatesCost())
    assert get_cost_value(bloq, QubitCount()) == get_cost_value(standard, QubitCount())


def test_gf4_resources_include_controlled_permutation() -> None:
    bloq = ControlledGF2ConstMul(spec=FieldSpec(p=2, r=2, f=(1, 1, 1)), c=(0, 1))
    costs = get_cost_value(bloq, QECGatesCost(legacy_shims=False))
    assert costs.toffoli == 1
    assert costs.cswap == 1


@pytest.mark.parametrize(
    ("spec", "constant"),
    [
        (FieldSpec(p=2, r=2, f=(1, 1, 1)), (0, 1)),
        (FieldSpec(p=2, r=2, f=(1, 1, 1)), (1, 1)),
        (FieldSpec(p=2, r=4, f=(1, 1, 0, 0, 1)), (0, 1, 0, 0)),
        (FieldSpec(p=2, r=4, f=(1, 1, 0, 0, 1)), (1, 0, 1, 1)),
    ],
)
def test_call_graph_matches_decomposition_counts(
    spec: FieldSpec, constant: tuple[int, ...]
) -> None:
    """手書き build_call_graph が分解由来の CNOT・SWAP 数と一致する。"""
    bloq = ControlledGF2ConstMul(spec=spec, c=constant)
    inner = next(
        instance.bloq
        for instance in bloq.decompose_bloq().bloq_instances
        if isinstance(instance.bloq, Controlled)
    ).subbloq

    reference = {
        type(callee).__name__: count
        for callee, count in dict(get_bloq_callee_counts(inner.decompose_bloq())).items()
    }
    actual = {
        type(callee).__name__: count for callee, count in inner.build_call_graph(None).items()
    }

    assert actual.get("CNOT", 0) == reference.get("CNOT", 0)
    assert actual.get("TwoBitSwap", 0) == reference.get("TwoBitSwap", 0)
    assert set(actual) <= {"CNOT", "TwoBitSwap"}


@pytest.mark.parametrize(
    ("spec", "constant", "bloq_type"),
    [
        (FieldSpec(p=2, r=1, f=(0, 1)), (1,), ControlledGF2ConstMul),
        (FieldSpec(p=2, r=2, f=(1, 1, 1)), (0, 1), ControlledGF2ConstMul),
        (FieldSpec(p=5, r=1, f=(0, 1)), (2,), ControlledConstMul),
    ],
)
def test_common_entry_matches_method_and_encoding(
    spec: FieldSpec, constant: tuple[int, ...], bloq_type: type[Bloq]
) -> None:
    bloq = get_controlled_const_mul(spec, constant)
    assert isinstance(bloq, bloq_type)
    assert bloq.signature.get_left("x").bitsize == spec.coefficient_bits
    assert bloq.signature.get_left("x").shape == (spec.r,)


def test_bloqs_reject_wrong_characteristic() -> None:
    binary_spec = FieldSpec(p=2, r=2, f=(1, 1, 1))
    with pytest.raises(ValueError, match="not supported"):
        ControlledConstMul(spec=binary_spec, c=(0, 1))
    with pytest.raises(ValueError, match="not supported"):
        ControlledLinearMapAdd(spec=binary_spec, matrix=((1, 0), (0, 1)))
    with pytest.raises(ValueError, match=r"GF\(2\) is required"):
        ControlledGF2ConstMul(spec=FieldSpec(p=5, r=1, f=(0, 1)), c=(1,))


def test_extension_field_linear_add_matches_hand_calculation() -> None:
    spec = FieldSpec(p=5, r=2, f=(2, 0, 1))
    bloq = ControlledLinearMapAdd(spec=spec, matrix=((0, 3), (1, 0)))
    decomposition = assert_valid_bloq_decomposition(bloq)
    for ctrl, expected in [(0, (1, 2)), (1, (4, 0))]:
        ctrl_out, x_out, y_out = decomposition.call_classically(
            ctrl=ctrl, x=np.array([3, 1]), y=np.array([1, 2])
        )
        assert ctrl_out == ctrl
        assert _as_tuple(x_out) == (3, 1)
        assert _as_tuple(y_out) == expected


@pytest.mark.parametrize("constant", [(1, 0), (0, 1), (2, 1)])
def test_extension_field_const_mul_restores_workspace_and_inverts(
    constant: tuple[int, ...],
) -> None:
    spec = FieldSpec(p=5, r=2, f=(2, 0, 1))
    bloq = ControlledConstMul(spec=spec, c=constant)
    decomposition = assert_valid_bloq_decomposition(bloq)
    inverse = bloq.adjoint().decompose_bloq()
    if constant == (1, 0):
        assert not decomposition.bloq_instances

    # 分解内のFreeが作業領域0を検査する。期待値はX^2=-2を使って独立に計算する。
    for ctrl, a, b in product((0, 1), range(5), range(5)):
        ctrl_out, x_out = decomposition.call_classically(ctrl=ctrl, x=np.array([a, b]))
        u, v = constant
        expected = ((u * a - 2 * v * b) % 5, (u * b + v * a) % 5) if ctrl else (a, b)
        assert ctrl_out == ctrl
        assert _as_tuple(x_out) == expected
        ctrl_back, x_back = inverse.call_classically(ctrl=ctrl_out, x=x_out)
        assert ctrl_back == ctrl
        assert _as_tuple(x_back) == (a, b)


@pytest.mark.parametrize("constant", [(0, 0), (1,), (1, 0, 0), (2, 0)])
def test_gf2_rejects_invalid_constants(constant: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        ControlledGF2ConstMul(spec=FieldSpec(p=2, r=2, f=(1, 1, 1)), c=constant)


@pytest.mark.parametrize(
    ("ctrl", "x"), [(2, [0, 1]), (-1, [0, 1]), (1, [0]), (1, [0, 2]), (0, [-1, 0])]
)
def test_gf2_rejects_invalid_classical_inputs(ctrl: int, x: list[int]) -> None:
    bloq = ControlledGF2ConstMul(spec=FieldSpec(p=2, r=2, f=(1, 1, 1)), c=(0, 1))
    with pytest.raises(ValueError):
        bloq.call_classically(ctrl=ctrl, x=np.array(x))
