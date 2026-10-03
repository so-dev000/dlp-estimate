from qualtran.testing import assert_consistent_classical_action, assert_valid_bloq_decomposition

from src.circuits.oracle import DeprecatedDLPOracle
from src.field import DLPInstance, FieldSpec


def _gf5_instance() -> DLPInstance:
    spec = FieldSpec(p=5, r=1, f=(1, 0))
    return DLPInstance(spec=spec, q=2, q_factors=((2, 1),), g=(4,), h=(1,))


def test_oracle_decomposition_matches_classical_action() -> None:
    bloq = DeprecatedDLPOracle(instance=_gf5_instance(), exponent_bits=2)
    assert_valid_bloq_decomposition(bloq)
    assert_consistent_classical_action(bloq, a=[0, 1, 3], b=[0, 1, 3], y=[[1], [2]])
