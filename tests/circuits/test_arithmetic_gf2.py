from itertools import product

from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.circuits.arithmetic_gf2 import ControlledGF2ConstMul
from src.field import FieldSpec


def test_decomposed_const_mul_preserves_coefficient_order():
    spec = FieldSpec(p=2, r=3, f=(1, 0, 1, 1))
    bloq = ControlledGF2ConstMul(spec=spec, c=(1, 0, 0))
    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(
        bloq, ctrl=[0, 1], x=list(product(range(spec.p), repeat=spec.r))
    )
