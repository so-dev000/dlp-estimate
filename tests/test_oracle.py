import numpy as np
import pytest
from qualtran import Adjoint
from qualtran.testing import assert_consistent_classical_action, assert_valid_bloq_decomposition

from src.field import DLPInstance, FieldSpec
from src.oracle import DeprecatedDLPOracle


@pytest.fixture
def instance() -> DLPInstance:
    spec = FieldSpec(p=5, r=1, f=(1, 0))
    return DLPInstance(spec=spec, q=2, q_factors=((2, 1),), g=(4,), h=(1,))


def test_oracle_multiplies_directly_on_target(instance: DLPInstance) -> None:
    # N&C 型: |a,b,x> -> |a,b,x * h^a g^b>。h=1, g=4。
    bloq = DeprecatedDLPOracle(instance=instance, exponent_bits=2)

    _, _, y = bloq.call_classically(a=1, b=0, y=np.array([1]))
    assert np.asarray(y).tolist() == [1]  # 1 * 1^1 * 4^0 = 1

    _, _, y = bloq.call_classically(a=1, b=1, y=np.array([1]))
    assert np.asarray(y).tolist() == [4]  # 1 * 1^1 * 4^1 = 4

    _, _, y = bloq.call_classically(a=1, b=1, y=np.array([2]))
    assert np.asarray(y).tolist() == [3]  # 2 * 1^1 * 4^1 = 8 = 3 mod 5


def test_oracle_is_not_self_inverse(instance: DLPInstance) -> None:
    bloq = DeprecatedDLPOracle(instance=instance, exponent_bits=2)
    assert isinstance(bloq.adjoint(), Adjoint)


def test_oracle_decomposition_is_valid(instance: DLPInstance) -> None:
    assert_valid_bloq_decomposition(DeprecatedDLPOracle(instance=instance, exponent_bits=2))


def test_oracle_decomposition_matches_classical_action(instance: DLPInstance) -> None:
    bloq = DeprecatedDLPOracle(instance=instance, exponent_bits=2)
    assert_consistent_classical_action(bloq, a=[0, 1, 3], b=[0, 1, 3], y=[[1], [2]])
