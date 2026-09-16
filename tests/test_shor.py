from collections import Counter

import numpy as np
import pytest
from qualtran import CBit, QBit, Side
from qualtran._infra.adjoint import Adjoint
from qualtran.bloqs.qft import QFTTextBook
from qualtran.testing import (
    assert_consistent_classical_action,
    assert_valid_bloq_decomposition,
)

from src.field import DLPInstance, FieldSpec
from src.shor import DLPOracle, FieldExponentiation, ShorDLP, make_shor_config


@pytest.fixture
def instance() -> DLPInstance:
    """GF(5) の部分群 {1, 4} で 4^d = 1 を解く最小入力。"""
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    return DLPInstance(spec=spec, q=2, g=(4,), h=(1,))


def test_make_shor_config_defaults_to_minimum_bits(instance: DLPInstance) -> None:
    assert make_shor_config(instance).exponent_bits == (2 * 2).bit_length()

    with pytest.raises(ValueError, match="at least"):
        make_shor_config(instance, exponent_bits=1)


def test_exponentiation_respects_bit_order(instance: DLPInstance) -> None:
    # base=2 の二乗定数は (2, 4, 1)。指数 3 = 0b011 は第0・第1定数だけを使う。
    # ビット順が逆だと (1, 4) の積になり、結果が変わる。
    bloq = FieldExponentiation(spec=instance.spec, base=(2,), exponent_bits=3)

    _, x = bloq.call_classically(exponent=3, x=np.array([1]))

    assert np.asarray(x).tolist() == [3]  # 2^3 * 1 = 8 = 3 mod 5


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


def test_oracle_is_self_inverse_on_arbitrary_y(instance: DLPInstance) -> None:
    bloq = DLPOracle(instance=instance, exponent_bits=2)  # h^1 g^1 = 4 -> (100)
    y_in = np.array([1, 0, 1])

    _, _, y_mid = bloq.call_classically(a=1, b=1, y=y_in)
    assert np.asarray(y_mid).tolist() == [0, 0, 1]

    a_out, b_out, y_out = bloq.call_classically(a=1, b=1, y=np.asarray(y_mid))
    assert (a_out, b_out) == (1, 1)
    assert np.asarray(y_out).tolist() == y_in.tolist()


def test_oracle_decomposition_is_valid(instance: DLPInstance) -> None:
    assert_valid_bloq_decomposition(DLPOracle(instance=instance, exponent_bits=2))


@pytest.fixture
def toy_instance() -> DLPInstance:
    """GF(5)、g=2、h=3、q=4 の toy 入力。既知解は 3。"""
    spec = FieldSpec(p=5, r=1, f=(0, 1))
    return DLPInstance(spec=spec, q=4, g=(2,), h=(3,))


def test_shor_dlp_signature(toy_instance: DLPInstance) -> None:
    config = make_shor_config(toy_instance)
    bloq = ShorDLP(instance=toy_instance, config=config)
    regs = {reg.name: reg for reg in bloq.signature}
    width = toy_instance.spec.r * toy_instance.spec.coefficient_bits

    assert set(regs) == {"a", "b", "y"}
    for name in ("a", "b"):
        assert regs[name].dtype == CBit()
        assert regs[name].shape == (config.exponent_bits,)
        assert regs[name].side == Side.RIGHT
    assert regs["y"].dtype == QBit()
    assert regs["y"].shape == (width,)
    assert regs["y"].side == Side.RIGHT


def test_shor_dlp_decomposition_structure(toy_instance: DLPInstance) -> None:
    # Qualtran 0.7.0 の assert_valid_bloq_decomposition は CBit の RIGHT 出力を
    # 拒否するため(CompositeBloq の接続検査が num_qubits > 0 を要求)、
    # ShorDLP については構築成功と部品構成で検証する。
    config = make_shor_config(toy_instance)
    bloq = ShorDLP(instance=toy_instance, config=config)
    bloqs = [binst.bloq for binst in bloq.decompose_bloq().bloq_instances]
    counts = Counter(type(b).__name__ for b in bloqs)

    assert counts["DLPOracle"] == 1  # oracle は1回
    assert counts["Hadamard"] == 2 * config.exponent_bits
    assert counts["MeasureZ"] == 2 * config.exponent_bits
    inverse_qfts = [b for b in bloqs if isinstance(b, Adjoint)]
    assert len(inverse_qfts) == 2  # a・b への逆QFT
    assert all(isinstance(b.subbloq, QFTTextBook) for b in inverse_qfts)
