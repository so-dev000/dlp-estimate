from itertools import product

from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.circuits.arithmetic_modp import ControlledConstMul, ControlledLinearMapAdd
from src.field import FieldSpec


def test_decomposed_const_mul_preserves_coefficient_order():
    spec = FieldSpec(p=5, r=2, f=(1, 0, 2))
    bloq = ControlledConstMul(spec=spec, c=(1, 2))
    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(
        bloq, ctrl=[0, 1], x=list(product(range(spec.p), repeat=spec.r))
    )


def test_linear_map_add_matches_classical_action():
    spec = FieldSpec(p=5, r=2, f=(1, 0, 2))
    matrix = ((1, 2), (3, 4))
    bloq = ControlledLinearMapAdd(spec=spec, matrix=matrix)
    assert_valid_bloq_decomposition(bloq)
    elements = list(product(range(spec.p), repeat=spec.r))
    assert_consistent_classical_action(bloq, ctrl=[0, 1], x=elements, y=elements)
