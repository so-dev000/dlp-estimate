import numpy as np
import pytest
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.exponentiation import DeprecatedFieldExponentiation, _squared_constants
from src.field import DLPInstance, FieldSpec, shared_field


@pytest.fixture
def instance() -> DLPInstance:
    spec = FieldSpec(p=5, r=1, f=(1, 0))
    return DLPInstance(spec=spec, q=2, q_factors=((2, 1),), g=(4,), h=(1,))


def test_exponentiation_respects_bit_order(instance: DLPInstance) -> None:
    bloq = DeprecatedFieldExponentiation(spec=instance.spec, base=(2,), exponent_bits=3)

    _, x = bloq.call_classically(exponent=3, x=np.array([1]))

    assert np.asarray(x).tolist() == [3]  # 2^3 * 1 = 8 = 3 mod 5


def test_exponentiation_rejects_zero_base_and_inverts_via_adjoint(
    instance: DLPInstance,
) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        DeprecatedFieldExponentiation(spec=instance.spec, base=(0,), exponent_bits=2)

    assert DeprecatedFieldExponentiation(
        spec=instance.spec, base=(2,), exponent_bits=2
    ).adjoint() == DeprecatedFieldExponentiation(spec=instance.spec, base=(3,), exponent_bits=2)


def test_exponentiation_decomposition_matches_classical_action(
    instance: DLPInstance,
) -> None:
    bloq = DeprecatedFieldExponentiation(spec=instance.spec, base=(2,), exponent_bits=2)

    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(bloq, exponent=[0, 1, 3], x=[[1], [2]])


def test_squared_constants_are_msb_first(instance: DLPInstance) -> None:
    assert _squared_constants(shared_field(instance.spec), (2,), 3) == ((1,), (4,), (2,))


@pytest.mark.parametrize(
    ("spec", "base"),
    [
        (FieldSpec(p=2, r=3, f=(1, 0, 1, 1)), (1, 0, 0)),
        (FieldSpec(p=5, r=2, f=(1, 0, 2)), (1, 2)),
    ],
)
def test_decomposed_exponentiation_matches_msb_bit_order(spec, base):
    field = shared_field(spec)
    bloq = DeprecatedFieldExponentiation(spec=spec, base=base, exponent_bits=3)
    assert_consistent_classical_action(
        bloq, exponent=list(range(8)), x=[field.zero, field.one, base]
    )
