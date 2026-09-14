import numpy as np
import pytest
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.field import DLPInstance, FieldSpec
from src.shor import DLPOracle, FieldExponentiation, make_shor_config


@pytest.fixture
def instance() -> DLPInstance:
    """GF(5) の部分群 {1, 4} で 4^d = 1 を解く最小入力。"""
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    return DLPInstance(spec=spec, q=2, q_factors=((2, 1),), g=(4,), h=(1,))


def test_make_shor_config_defaults_to_minimum_bits(instance: DLPInstance) -> None:
    assert make_shor_config(instance).exponent_bits == (2 * 2).bit_length()

    with pytest.raises(ValueError, match="at least"):
        make_shor_config(instance, exponent_bits=1)


def test_exponentiation_multiplies_by_base_to_the_exponent(instance: DLPInstance) -> None:
    bloq = FieldExponentiation(spec=instance.spec, base=(2,), exponent_bits=2)

    _, x = bloq.call_classically(exponent=3, x=np.array([2]))

    assert np.asarray(x).tolist() == [1]  # 2^3 * 2 = 16 = 1 mod 5


def test_exponentiation_rejects_zero_base_and_inverts_via_adjoint(
    instance: DLPInstance,
) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        FieldExponentiation(spec=instance.spec, base=(0,), exponent_bits=2)

    assert FieldExponentiation(
        spec=instance.spec, base=(2,), exponent_bits=2
    ).adjoint() == FieldExponentiation(spec=instance.spec, base=(3,), exponent_bits=2)


def test_exponentiation_decomposition_matches_classical_action(
    instance: DLPInstance,
) -> None:
    bloq = FieldExponentiation(spec=instance.spec, base=(2,), exponent_bits=2)

    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(bloq, exponent=[0, 1, 3], x=[[1], [2]])


def test_oracle_xors_encoded_function_value(instance: DLPInstance) -> None:
    bloq = DLPOracle(instance=instance, exponent_bits=2)  # h^1 g^0 = 1 -> (001)

    _, _, y = bloq.call_classically(a=1, b=0, y=np.array([0, 0, 0]))

    assert np.asarray(y).tolist() == [0, 0, 1]
    assert bloq.adjoint() is bloq  # 自己逆元


def test_oracle_decomposition_is_valid(instance: DLPInstance) -> None:
    assert_valid_bloq_decomposition(DLPOracle(instance=instance, exponent_bits=2))
