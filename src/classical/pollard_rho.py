import math
import random

from src.field import FieldElement, FiniteField

type PollardRhoState = tuple[FieldElement, int, int]


def pollard_rho(
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
    """X = g**a * h**b を保ちながら、状態を1ステップ進める。"""
    element, a, b = state
    # 整数表現 sum(c_i * p**i) の mod 3 を、高次数の係数から計算する。
    bucket = 0
    for coefficient in reversed(element):
        bucket = (bucket * field.spec.p + coefficient) % 3

    if bucket == 0:
        # S_0: X' = Xh
        return (field.mul(element, h), a, (b + 1) % order)
    elif bucket == 1:
        # S_1: X' = X²
        return (field.mul(element, element), (2 * a) % order, (2 * b) % order)
    else:
        # S_2: X' = Xg
        return (field.mul(element, g), (a + 1) % order, b)
