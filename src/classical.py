import random

from field import DLPInstance, FieldElement, FiniteField

_RHO_MAX_STEPS = 100_000


def solve_pohlig_hellman(instance: DLPInstance, *, seed: int = 0) -> int:
    """Pohlig-Hellman法で離散対数を解く。"""
    field = FiniteField(instance.field)
    if instance.h == field.one:
        return 0
    if instance.h == instance.g:
        return 1

    rng = random.Random(seed)
    congruences = []
    for prime, exponent in instance.q_factors:
        sub_log = _solve_prime_power(field, instance, prime, exponent, rng=rng)
        sub_order = prime**exponent
        congruences.append((sub_log, sub_order))

    discrete_log = _chinese_remainder_theorem(congruences)
    # 検算
    if field.pow(instance.g, discrete_log) != instance.h:
        raise RuntimeError("computed discrete logarithm does not satisfy g**d = h")
    return discrete_log


def _solve_prime_power(
    field: FiniteField, instance: DLPInstance, prime: int, exponent: int, *, rng: random.Random
) -> int:
    """位数が素数べきの部分群における離散対数を解く。"""
    g = instance.g
    h = instance.h
    q = instance.q

    sub_order = prime**exponent
    sub_group_exponent = q // sub_order  # gの位数を部分群の位数で割ったもの
    g_i = field.pow(g, sub_group_exponent)
    h_i = field.pow(h, sub_group_exponent)

    gamma = field.pow(g_i, prime ** (exponent - 1))

    x = 0
    for i in range(exponent):
        residual = field.mul(h_i, field.inv(field.pow(g_i, x)))
        c = field.pow(residual, prime ** (exponent - 1 - i))
        d = _pollard_rho(field, gamma, c, prime, rng=rng)
        x += d * (prime**i)
    return x


def _chinese_remainder_theorem(congruences: list[tuple[int, int]]) -> int:
    """中国剰余定理で合同式の解を求める。"""
    pass


def _pollard_rho(
    field: FiniteField, g: FieldElement, h: FieldElement, order: int, *, rng: random.Random
) -> int:
    pass


def _rho_step(
    field: FiniteField,
    g: FieldElement,
    h: FieldElement,
    order: int,
    rng: random.Random,
    state: tuple[FieldElement, int, int],
) -> tuple[FieldElement, int, int]:
    pass
