from dataclasses import replace

import pytest

from src import classical
from src.classical import (
    _chinese_remainder_theorem,
    solve_pohlig_hellman,
)
from src.field import DLPInstance, FieldElement, FieldSpec, FiniteField


@pytest.fixture
def composite_order_instance() -> DLPInstance:
    """GF(101) で 2 の位数は 100 = 2² * 5²、2⁷³ = 48。"""
    return DLPInstance(
        spec=FieldSpec(p=101, r=1, f=(0, 1)),
        q=100,
        q_factors=((2, 2), (5, 2)),
        g=(2,),
        h=(48,),
    )


@pytest.fixture
def gf23() -> FiniteField:
    """GF(23) の部分群 <2> は位数 11。2³ = 8。"""
    return FiniteField(FieldSpec(p=23, r=1, f=(0, 1)))


@pytest.mark.parametrize(
    ("target", "expected_log"),
    [
        pytest.param((1,), 0, id="identity"),
        pytest.param((2,), 1, id="generator"),
        pytest.param((48,), 73, id="combine-prime-power-subgroups"),
    ],
)
def test_pohlig_hellman_recovers_known_logarithm(
    composite_order_instance: DLPInstance, target: FieldElement, expected_log: int
) -> None:
    instance = replace(composite_order_instance, h=target)

    assert solve_pohlig_hellman(instance, seed=0) == expected_log


def test_pohlig_hellman_solves_proper_subgroup_in_extension_field() -> None:
    # X² = 3 なので X³ = 3X。X の位数は 8、体の乗法群の位数は 24。
    instance = DLPInstance(
        spec=FieldSpec(p=5, r=2, f=(2, 0, 1)),
        q=8,
        q_factors=((2, 3),),
        g=(0, 1),
        h=(0, 3),
    )

    assert solve_pohlig_hellman(instance, seed=0) == 3


def test_pohlig_hellman_rejects_incorrect_recombined_answer(
    composite_order_instance: DLPInstance, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(classical, "_chinese_remainder_theorem", lambda _: 0)

    with pytest.raises(RuntimeError, match="does not satisfy"):
        solve_pohlig_hellman(composite_order_instance, seed=0)


def test_chinese_remainder_theorem_combines_congruences() -> None:
    # 23 mod 3 = 2、23 mod 5 = 3、23 mod 7 = 2。
    assert _chinese_remainder_theorem([(2, 3), (3, 5), (2, 7)]) == 23
