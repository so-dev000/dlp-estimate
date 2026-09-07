import math
import random

from .field import DLPInstance, FieldElement, FiniteField

type PollardRhoState = tuple[FieldElement, int, int]


def solve_pohlig_hellman(instance: DLPInstance, *, seed: int = 0) -> int:
    """Pohlig-Hellman法で離散対数問題を部分群ごとに解き、CRTで組み合わせる。"""
    field = FiniteField(instance.field)
    h = instance.h
    g = instance.g

    if h == field.one:
        return 0
    if h == g:
        return 1

    rng = random.Random(seed)
    congruences = []
    for prime, exponent in instance.q_factors:
        sub_log = _solve_prime_power(field, instance, prime, exponent, rng=rng)
        sub_order = prime**exponent
        congruences.append((sub_log, sub_order))

    discrete_log = _chinese_remainder_theorem(congruences)
    # 検算
    if field.pow(g, discrete_log) != h:
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
    total_order = 1
    # N = n1 * n2 * ... * nk
    for _, order in congruences:
        total_order *= order

    x = 0
    for remainder, order in congruences:
        partial_order = total_order // order  # N_i = N / n_i
        inverse = pow(partial_order, -1, order)  # M_i = N_i^(-1) mod n_i
        x += remainder * partial_order * inverse  # x = Σ a_i * N_i * M_i
    return x % total_order


def _pollard_rho(
    field: FiniteField,
    g: FieldElement,
    h: FieldElement,
    order: int,
    *,
    rng: random.Random,
    max_restarts: int = 32,
    step_factor: int = 10,
) -> int:
    """Pollard rho法で離散対数を解く。"""
    if h == field.one:
        return 0
    elif h == g:
        return 1

    # _rho_stepのwrapper
    def step(state: PollardRhoState) -> PollardRhoState:
        return _rho_step(field, g, h, order, state)

    max_steps = math.ceil(step_factor * math.sqrt(order))

    for _ in range(max_restarts):
        a = rng.randrange(order)
        b = rng.randrange(order)
        x_0 = field.mul(field.pow(g, a), field.pow(h, b))
        initial_state = (x_0, a, b)

        # Floydのサイクル検出法で衝突を探す
        tortoise = step(initial_state)
        hare = step(step(initial_state))

        for _ in range(max_steps):
            element_t, a_t, b_t = tortoise
            element_h, a_h, b_h = hare

            # 衝突が見つかった場合
            if element_h == element_t:
                numerator = (a_t - a_h) % order
                denominator = (b_h - b_t) % order
                if denominator == 0:
                    break
                candidate = (numerator * pow(denominator, -1, order)) % order
                # 検算
                if field.pow(g, candidate) == h:
                    return candidate
                break

            tortoise = step(tortoise)
            hare = step(step(hare))

    raise RuntimeError("Pollard rho failed to find a discrete logarithm after max_restarts")


def _rho_step(
    field: FiniteField,
    g: FieldElement,
    h: FieldElement,
    order: int,
    state: PollardRhoState,
) -> PollardRhoState:
    """Pollard rho法の状態を1ステップ進める。"""
    element, a, b = state
    # 3つの部分集合に分ける
    bucket = hash(element) % 3

    if bucket == 0:
        # X' = Xh
        return (field.mul(element, h), a, (b + 1) % order)
    elif bucket == 1:
        # X' = X²
        return (field.mul(element, element), (2 * a) % order, (2 * b) % order)
    else:
        # X' = Xg
        return (field.mul(element, g), (a + 1) % order, b)
