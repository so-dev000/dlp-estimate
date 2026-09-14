import numpy as np
import pytest
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.arithmetic import ControlledConstMul, ControlledLinearMapAdd
from src.field import FieldSpec


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
