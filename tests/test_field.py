from itertools import product

import galois
import numpy as np
import pytest

from src.field import FieldSpec, FiniteField, format_polynomial


@pytest.fixture(
    params=[
        FieldSpec(p=2, r=3, f=(1, 0, 1, 1)),
        FieldSpec(p=5, r=2, f=(1, 0, 2)),
    ]
)
def fields(request):
    spec = request.param
    field = FiniteField(spec)
    field.validate()
    reference = galois.GF(
        spec.p**spec.r,
        irreducible_poly=galois.Poly(spec.f, field=galois.GF(spec.p)),
        compile="python-calculate",
    )
    return field, reference


def as_integer(coefficients, p):
    value = 0
    for coefficient in coefficients:
        value = value * p + coefficient
    return value


def test_arithmetic_matches_galois_in_high_degree_order(fields):
    field, reference = fields
    spec = field.spec
    elements = list(product(range(spec.p), repeat=spec.r))

    assert field.one == (0,) * (spec.r - 1) + (1,)
    for a, b in product(elements, repeat=2):
        left = reference(as_integer(a, spec.p))
        right = reference(as_integer(b, spec.p))
        assert field.add(a, b) == tuple((left + right).vector().tolist())
        assert field.sub(a, b) == tuple((left - right).vector().tolist())
        assert field.mul(a, b) == tuple((left * right).vector().tolist())


def test_powers_and_inverses_match_galois(fields):
    field, reference = fields
    spec = field.spec
    for a in product(range(spec.p), repeat=spec.r):
        expected = reference(as_integer(a, spec.p))
        for exponent in (-3, -1, 0, 1, 2, 5):
            if not any(a) and exponent < 0:
                continue
            assert field.pow(a, exponent) == tuple((expected**exponent).vector().tolist())
        if any(a):
            assert field.inv(a) == tuple((expected**-1).vector().tolist())
    with pytest.raises(ZeroDivisionError):
        field.inv(field.zero)


def test_constant_multiplication_matrix_uses_high_degree_basis(fields):
    field, reference = fields
    spec = field.spec
    elements = list(product(range(spec.p), repeat=spec.r))
    for c in elements:
        matrix = np.array(field.const_mul_matrix(c), dtype=object)
        constant = reference(as_integer(c, spec.p))
        for a in elements:
            expected = constant * reference(as_integer(a, spec.p))
            assert tuple((matrix @ a % spec.p).tolist()) == tuple(expected.vector().tolist())


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ((1, 0, 2), "X^2 + 2"),
        ((0, 0), "0"),
    ],
)
def test_format_polynomial(coefficients, expected):
    assert format_polynomial(coefficients) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (FieldSpec(p=2, r=3, f=(1, 0, 1, 1)), 3),
        (FieldSpec(p=5, r=1, f=(1, 0)), 3),
        (FieldSpec(p=5, r=2, f=(1, 0, 2)), 6),
    ],
)
def test_register_bits_counts_padded_encoding(spec, expected):
    assert spec.register_bits == expected
    assert spec.register_bits == spec.r * spec.coefficient_bits


def test_prime_field_modulus_is_high_degree_first():
    field = FiniteField(FieldSpec(p=5, r=1, f=(1, 0)))
    field.validate()
    assert field.mul((2,), (3,)) == (1,)
    with pytest.raises(ValueError, match="monic"):
        FieldSpec(p=5, r=1, f=(0, 1))
