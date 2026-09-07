import random
from dataclasses import replace

import pytest

from src import classical
from src.classical import (
    PollardRhoState,
    _chinese_remainder_theorem,
    _pollard_rho,
    _rho_step,
    solve_pohlig_hellman,
)
from src.field import DLPInstance, FieldElement, FieldSpec, FiniteField


@pytest.fixture
def composite_order_instance() -> DLPInstance:
    """GF(101) で 2 の位数は 100 = 2² * 5²、2⁷³ = 48。"""
    return DLPInstance(
        field=FieldSpec(p=101, r=1, f=(0, 1)),
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
        field=FieldSpec(p=5, r=2, f=(2, 0, 1)),
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


@pytest.mark.parametrize(
    ("target", "expected_log"),
    [
        pytest.param((1,), 0, id="identity"),
        pytest.param((2,), 1, id="generator"),
        pytest.param((8,), 3, id="nontrivial-logarithm"),
    ],
)
def test_pollard_rho_recovers_known_logarithm(
    gf23: FiniteField, target: FieldElement, expected_log: int
) -> None:
    result = _pollard_rho(gf23, (2,), target, 11, rng=random.Random(0))

    assert result == expected_log


def test_pollard_rho_reports_exhausted_restarts(
    gf23: FiniteField, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 常に Xg の分岐に進むと b が変わらず、有効な衝突を得られない。
    monkeypatch.setattr(classical, "hash", lambda _: 2, raising=False)

    with pytest.raises(RuntimeError, match="after max_restarts"):
        _pollard_rho(gf23, (2,), (8,), 11, rng=random.Random(0), max_restarts=2)


@pytest.mark.parametrize(
    ("bucket", "expected_state"),
    [
        pytest.param(0, ((12,), 10, 0), id="multiply-by-target"),
        pytest.param(1, ((8,), 9, 9), id="square"),
        pytest.param(2, ((3,), 0, 10), id="multiply-by-generator"),
    ],
)
def test_rho_step_updates_element_and_wraps_exponents(
    gf23: FiniteField,
    monkeypatch: pytest.MonkeyPatch,
    bucket: int,
    expected_state: PollardRhoState,
) -> None:
    # Python の hash 値に依存せず、各分岐を 1 回ずつ検証する。
    monkeypatch.setattr(classical, "hash", lambda _: bucket, raising=False)
    initial_state = ((13,), 10, 10)  # 2¹⁰ * 8¹⁰ = 13 mod 23

    assert _rho_step(gf23, (2,), (8,), 11, initial_state) == expected_state


@pytest.mark.xfail(
    all(hash(element) % 3 == 2 for element in [(1,), (5,), (25,)]),
    reason="位数 3 の部分群 {1, 5, 25} がすべて Xg の分岐に入り、再試行しても解けない",
    raises=RuntimeError,
    strict=True,
)
def test_pohlig_hellman_solves_subgroup_with_degenerate_hash_partition() -> None:
    # 3² = 9 mod 31。hash の分割方法が改善されたら xfail を外す。
    instance = DLPInstance(
        field=FieldSpec(p=31, r=1, f=(0, 1)),
        q=30,
        q_factors=((2, 1), (3, 1), (5, 1)),
        g=(3,),
        h=(9,),
    )

    assert solve_pohlig_hellman(instance, seed=0) == 2
