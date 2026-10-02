from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import sympy as sp

from src.field import FieldSpec, FiniteField

OUT = Path(__file__).resolve().parents[1] / "src" / "literature_data"
SCHEMA = 1


def _coeffs(expr, x, n, p):
    poly = sp.Poly(expr, x, modulus=p)
    return tuple(int(poly.nth(i)) % p for i in range(n, -1, -1))


def _quad(field, c, t):
    def const(a):
        return (0,) * (field.spec.r - 1) + (a,)

    return field.add(
        const(c[0]),
        field.mul(t, field.add(const(c[1]), field.mul(t, const(c[2])))),
    )


def _dense(values):
    return {
        "encoding": "dense_coefficients_high_to_low",
        "values": list(map(str, values)),
    }


def generate_dgp21():
    p, c = 135066410865995223349603927, 64417723306991464419622353
    known_log = 7627280816875322297766747970138378530353852976315498
    a = (
        31415926535897932384626433,
        83279502884197169399375105,
        82097494459230781640628620,
    )
    b = (
        89986280348253421170679821,
        48086513282306647093844609,
        55058223172535940812848111,
    )

    t, x = sp.symbols("t X")
    base = t**3 - t + 1
    ext = (x - t) ** 2 + c * (x - t) + 1

    f = _coeffs(sp.resultant(base, ext, t), x, 6, p)
    if f[0] != 1:
        inv = pow(f[0], -1, p)
        f = tuple(inv * v % p for v in f)

    gb = sp.groebner([base, ext], t, x, order="lex", modulus=p)
    rel = next(z.as_expr() for z in gb.polys if sp.degree(z.as_expr(), t) == 1)

    ct = int(sp.Poly(rel, t, x, modulus=p).coeff_monomial(t)) % p
    if ct == 0:
        raise RuntimeError("DGP21 failed to solve t as a polynomial in X")

    t_flat = _coeffs(
        -pow(ct, -1, p) * sp.expand(rel - ct * t),
        x,
        5,
        p,
    )

    field = FiniteField(FieldSpec(p=p, r=6, f=f))
    x_flat = (0, 0, 0, 0, 1, 0)

    target = field.add(
        _quad(field, a, t_flat),
        field.mul(
            field.sub(x_flat, t_flat),
            _quad(field, b, t_flat),
        ),
    )

    q = p * p - p + 1
    if (p**6 - 1) % q != 0:
        raise RuntimeError("DGP21 q does not divide p^6 - 1")

    cofactor = (p**6 - 1) // q
    g = field.pow(x_flat, cofactor)
    h = field.pow(target, cofactor)

    if not sp.isprime(q) or g == field.one or field.pow(g, q) != field.one:
        raise RuntimeError("DGP21 order verification failed")

    if field.pow(g, known_log) != h:
        raise RuntimeError("DGP21 known-log verification failed")

    return {
        "schema_version": SCHEMA,
        "label": "dgp21",
        "field": {
            "p": str(p),
            "r": 6,
            "modulus": _dense(f),
        },
        "dlp": {
            "q": str(q),
            "q_factors": [[str(q), 1]],
            "g": _dense(g),
            "h": _dense(h),
            "known_log": str(known_log),
            "order_status": "proven_prime_order_subgroup",
        },
    }


def _xor(*groups):
    support = set()
    for group in groups:
        support.symmetric_difference_update(group)
    return tuple(sorted(support))


def _u(power):
    return tuple(1025 * k + power - k for k in range(power + 1) if math.comb(power, k) & 1)


def _reduce(value, modulus, degree):
    while value.bit_length() - 1 >= degree:
        value ^= modulus << (value.bit_length() - 1 - degree)
    return value


def _mul_sparse(value, exponents, modulus, degree):
    product = 0
    for exponent in exponents:
        product ^= value << exponent
    return _reduce(product, modulus, degree)


def generate_gklwz21():
    r = 30_750

    u10, u20, u30 = _u(10), _u(20), _u(30)
    f_exponents = _xor(u30, u20, u10, (1025, 1), (0,))
    t_exponents = _xor(u10, (0,))
    g_exponents = _xor((1,), _u(3))

    if f_exponents[-1] != r:
        raise RuntimeError("GKLWZ21 modulus verification failed")

    if g_exponents != (1, 3, 1027, 2051, 3075):
        raise RuntimeError("GKLWZ21 generator verification failed")

    modulus = sum(1 << e for e in f_exponents)

    t_powers = [1]
    for _ in range(29):
        t_powers.append(
            _mul_sparse(
                t_powers[-1],
                t_exponents,
                modulus,
                r,
            )
        )

    pi_scaled = int(
        sp.N(
            sp.pi * sp.Integer(2) ** r,
            9_500,
        )
    )

    h = 0
    for e in range(r):
        if (pi_scaled >> (r - 1 - e)) & 1:
            h ^= t_powers[29 - e % 30] << (e // 30)

    h = _reduce(h, modulus, r)

    digest = hashlib.sha256(h.to_bytes((r + 7) // 8, "little")).hexdigest()

    expected = "ec6f487c2a2b9b6aba259d2851aae888c127ea386bd6e9e584e81bcfbab14c9d"

    if digest != expected:
        raise RuntimeError("GKLWZ21 target verification failed")

    return {
        "schema_version": SCHEMA,
        "label": "gklwz21",
        "field": {
            "p": "2",
            "r": r,
            "modulus": {
                "encoding": "gf2_nonzero_exponents",
                "exponents": list(f_exponents),
            },
        },
        "dlp": {
            "q": {
                "encoding": "mersenne",
                "exponent": r,
            },
            "q_factors": None,
            "g": {
                "encoding": "gf2_nonzero_exponents",
                "length": r,
                "exponents": list(g_exponents),
            },
            "h": {
                "encoding": "gf2_lsb0_hex",
                "length": r,
                "hex": f"{h:0{(r + 3) // 4}x}",
                "sha256_little_endian": digest,
            },
            "known_log": None,
            "order_status": "paper_presumed_generator_full_multiplicative_group",
        },
    }


def _gf2_mul(a, b, modulus, degree):
    product = 0
    while b:
        if b & 1:
            product ^= a
        a <<= 1
        b >>= 1
    return _reduce(product, modulus, degree)


def _gf2_pow(base, exponent, modulus, degree):
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf2_mul(result, base, modulus, degree)
        base = _gf2_mul(base, base, modulus, degree)
        exponent >>= 1
    return result


def _gf2_pow_x_2n(n, modulus, degree):
    """x^(2^n) mod modulus を n 回の平方で求める。"""
    result = 0b10
    for _ in range(n):
        result = _gf2_mul(result, result, modulus, degree)
    return result


def generate_kleinjung14():
    r = 1279

    # h1(x^q)*x + h0(x^q) を x で割ったものが定義多項式。
    # h1 = x+1, h0 = x^5+x^3+x より
    # f = x^1279 + x^767 + x^256 + x^255 + 1 (標数2では - = +)。
    f_exponents = (0, 255, 256, 767, r)

    modulus = sum(1 << e for e in f_exponents)

    if modulus.bit_length() - 1 != r:
        raise RuntimeError("KLEINJUNG14 modulus verification failed")

    # Rabin test: r は素数のため、x^(2^r) = x かつ線形因子なしで既約。
    if _gf2_pow_x_2n(r, modulus, r) != 0b10:
        raise RuntimeError("KLEINJUNG14 irreducibility verification failed")

    if modulus & 1 != 1 or bin(modulus).count("1") % 2 != 1:
        raise RuntimeError("KLEINJUNG14 linear-factor verification failed")

    order = (1 << r) - 1

    if not sp.isprime(order):
        raise RuntimeError("KLEINJUNG14 order primality verification failed")

    x = 0b10

    if x == 1 or _gf2_pow(x, order, modulus, r) != 1:
        raise RuntimeError("KLEINJUNG14 generator verification failed")

    pi_scaled = int(sp.N(sp.pi * sp.Integer(2) ** r, 2000))

    if pi_scaled.bit_length() != r + 2:
        raise RuntimeError("KLEINJUNG14 pi precision verification failed")

    h = 0
    for i in range(r):
        if (pi_scaled >> (r - 1 - i)) & 1:
            h |= 1 << i

    h = _reduce(h, modulus, r)

    digest = hashlib.sha256(h.to_bytes((r + 7) // 8, "little")).hexdigest()

    known_log = int(
        "3212750760383542442717887844353225418270190233889477506520509005251151805661482432193924349687140554198064504993379500428095843726914531339996055760370853427597658839547030087071391545204047791193885994409524243018423092634151430844517137778559194148975494771537228921138598346875362703070651041102748164857763667856599890811247759947699602938086144581217406940091918470212637857540496"
    )

    if _gf2_pow(x, known_log, modulus, r) != h:
        raise RuntimeError("KLEINJUNG14 known-log verification failed")

    return {
        "schema_version": SCHEMA,
        "label": "kleinjung14",
        "field": {
            "p": "2",
            "r": r,
            "modulus": {
                "encoding": "gf2_nonzero_exponents",
                "exponents": list(f_exponents),
            },
        },
        "dlp": {
            "q": str(order),
            "q_factors": [[str(order), 1]],
            "g": {
                "encoding": "gf2_nonzero_exponents",
                "length": r,
                "exponents": [1],
            },
            "h": {
                "encoding": "gf2_lsb0_hex",
                "length": r,
                "hex": f"{h:0{(r + 3) // 4}x}",
                "sha256_little_endian": digest,
            },
            "known_log": str(known_log),
            "order_status": "proven_generator_full_multiplicative_group",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    generated = {
        "dgp21.json": generate_dgp21(),
        "gklwz21.json": generate_gklwz21(),
        "kleinjung14.json": generate_kleinjung14(),
    }

    OUT.mkdir(parents=True, exist_ok=True)

    for name, data in generated.items():
        path = OUT / name
        text = json.dumps(data, indent=2, ensure_ascii=True) + "\n"

        if args.check:
            if not path.is_file():
                raise FileNotFoundError(path)

            if path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"generated data differs from {path}")

            print(f"OK: {path}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
