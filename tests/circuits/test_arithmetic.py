from itertools import product

import pytest
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.circuits.arithmetic import ControlledLinearMapAdd, get_controlled_const_mul
from src.field import FieldSpec


@pytest.mark.parametrize(
    ("spec", "constant"),
    [
        (FieldSpec(p=2, r=3, f=(1, 0, 1, 1)), (1, 0, 0)),
        (FieldSpec(p=5, r=2, f=(1, 0, 2)), (1, 2)),
    ],
)
def test_decomposed_const_mul_preserves_coefficient_order(spec, constant):
    bloq = get_controlled_const_mul(spec, constant)
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
