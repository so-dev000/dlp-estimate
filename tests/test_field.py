from dataclasses import replace

import pytest

from src.field import (
    Bits,
    DLPInstance,
    Factorization,
    FieldElement,
    FieldSpec,
    FiniteField,
    PolynomialCoefficients,
    decode,
    embedding_degree,
    encode,
)


@pytest.fixture
def gf25() -> FiniteField:
    """GF(5)[X] / (X² + 2)。期待値の計算には X² = 3 を使う。"""
    return FiniteField(FieldSpec(p=5, r=2, f=(2, 0, 1)))


@pytest.mark.parametrize(
    ("p", "r", "f"),
    [
        pytest.param(1, 1, (0, 1), id="characteristic-too-small"),
        pytest.param(5, 0, (1,), id="degree-too-small"),
        pytest.param(5, 2, (0, 1), id="wrong-polynomial-degree"),
        pytest.param(5, 2, (5, 0, 1), id="coefficient-out-of-range"),
        pytest.param(5, 2, (2, 0, 2), id="non-monic-polynomial"),
        pytest.param(5, 1, (1, 1), id="prime-field-polynomial-must-be-x"),
    ],
)
def test_field_spec_rejects_invalid_values(p: int, r: int, f: PolynomialCoefficients) -> None:
    with pytest.raises(ValueError):
        FieldSpec(p=p, r=r, f=f)


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(FieldSpec(p=4, r=1, f=(0, 1)), id="composite-characteristic"),
        # X² + 4 = (X - 1)(X + 1) over GF(5)。
        pytest.param(FieldSpec(p=5, r=2, f=(4, 0, 1)), id="reducible-polynomial"),
    ],
)
def test_field_construction_checks_primality_and_irreducibility(spec: FieldSpec) -> None:
    with pytest.raises(ValueError):
        FiniteField(spec)


def test_prime_field_arithmetic() -> None:
    field = FiniteField(FieldSpec(p=5, r=1, f=(0, 1)))

    assert field.zero == (0,)
    assert field.one == (1,)
    assert field.add((2,), (4,)) == (1,)  # 6 mod 5
    assert field.sub((2,), (4,)) == (3,)  # -2 mod 5
    assert field.neg((2,)) == (3,)
    assert field.mul((2,), (4,)) == (3,)  # 8 mod 5
    assert field.inv((2,)) == (3,)  # 2 * 3 = 1 mod 5
    assert field.pow((2,), 4) == (1,)


def test_extension_field_arithmetic(gf25: FiniteField) -> None:
    assert gf25.spec.order == 25
    assert gf25.zero == (0, 0)
    assert gf25.one == (1, 0)
    assert gf25.add((3, 4), (4, 2)) == (2, 1)
    assert gf25.sub((3, 1), (4, 2)) == (4, 4)
    assert gf25.neg((3, 1)) == (2, 4)
    assert gf25.mul((3, 1), (0, 1)) == (3, 3)  # (3 + X)X = 3 + 3X
    assert gf25.inv((0, 1)) == (0, 2)  # X * 2X = 2X² = 1
    assert gf25.pow((0, 1), -2) == (2, 0)  # (2X)² = 4X² = 2


def test_constant_multiplication_matrix_uses_basis_images_as_columns(gf25: FiniteField) -> None:
    # 第 1 列は X * 1 = X、第 2 列は X * X = 3 の係数列。
    assert gf25.const_mul_matrix((0, 1)) == ((0, 3), (1, 0))


def test_galois_conversion_preserves_coefficient_order(gf25: FiniteField) -> None:
    scalar = gf25.to_galois((3, 1))

    assert scalar.ndim == 0
    assert scalar.vector().tolist() == [1, 3]  # galois は高次数順。
    assert gf25.from_galois(scalar) == (3, 1)


def test_from_galois_requires_scalar_from_same_field(gf25: FiniteField) -> None:
    scalar = gf25.to_galois((3, 1))
    vector = type(scalar)([scalar])
    other_field = FiniteField(FieldSpec(p=5, r=1, f=(0, 1)))

    with pytest.raises(ValueError, match="galois scalar belonging to this field"):
        gf25.from_galois(vector)
    with pytest.raises(ValueError, match="galois scalar belonging to this field"):
        gf25.from_galois(other_field.to_galois((1,)))


def test_zero_powers_and_inverse(gf25: FiniteField) -> None:
    assert gf25.pow(gf25.zero, 0) == gf25.one
    assert gf25.pow(gf25.zero, 2) == gf25.zero

    with pytest.raises(ZeroDivisionError):
        gf25.inv(gf25.zero)
    with pytest.raises(ZeroDivisionError):
        gf25.pow(gf25.zero, -1)


@pytest.mark.parametrize(
    "element",
    [(1,), (5, 0)],
    ids=["wrong-length", "coefficient-out-of-range"],
)
def test_conversion_and_encoding_reject_invalid_elements(
    gf25: FiniteField, element: FieldElement
) -> None:
    with pytest.raises(ValueError):
        gf25.to_galois(element)
    with pytest.raises(ValueError):
        encode(element, gf25.spec)


def test_encoding_uses_low_degree_order_and_big_endian_bits(gf25: FiniteField) -> None:
    bits = (0, 1, 1, 0, 0, 1)  # (3, 1) → (011, 001)

    assert gf25.spec.coefficient_bits == 3
    assert encode((3, 1), gf25.spec) == bits
    assert decode(bits, gf25.spec) == (3, 1)


def test_binary_field_uses_two_bits_per_coefficient() -> None:
    field = FiniteField(FieldSpec(p=2, r=1, f=(0, 1)))

    assert field.add((1,), (1,)) == (0,)
    assert field.spec.coefficient_bits == 2
    assert encode((1,), field.spec) == (0, 1)
    assert decode((0, 1), field.spec) == (1,)
    with pytest.raises(ValueError, match="decoded coefficients"):
        decode((1, 0), field.spec)


@pytest.mark.parametrize(
    ("bits", "message"),
    [
        pytest.param((0, 0, 0), "tuple of length 6", id="wrong-length"),
        pytest.param((0, 0, 2, 0, 0, 0), "only the integers 0 and 1", id="non-bit"),
        pytest.param((1, 0, 1, 0, 0, 0), "coefficients must be less than 5", id="out-of-range"),
    ],
)
def test_decode_rejects_invalid_bits(gf25: FiniteField, bits: Bits, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode(bits, gf25.spec)


@pytest.mark.parametrize(
    ("spec", "order", "factors", "generator", "expected_degree"),
    [
        pytest.param(FieldSpec(p=5, r=1, f=(0, 1)), 4, ((2, 2),), (2,), 1, id="prime-field"),
        pytest.param(
            FieldSpec(p=5, r=2, f=(2, 0, 1)), 8, ((2, 3),), (0, 1), 2, id="full-extension"
        ),
        pytest.param(
            FieldSpec(p=3, r=4, f=(2, 1, 0, 0, 1)),
            2,
            ((2, 1),),
            (2, 0, 0, 0),
            1,
            id="prime-subfield-with-repeated-degree-reduction",
        ),
    ],
)
def test_embedding_degree_is_smallest_degree_containing_subgroup(
    spec: FieldSpec,
    order: int,
    factors: Factorization,
    generator: FieldElement,
    expected_degree: int,
) -> None:
    instance = DLPInstance(
        spec=spec, q=order, q_factors=factors, g=generator, h=FiniteField(spec).one
    )

    assert embedding_degree(instance) == expected_degree


@pytest.fixture
def order_two_instance() -> DLPInstance:
    """GF(5) の部分群 {1, 4} で 4ᵈ = 1 を解く入力。"""
    return DLPInstance(
        spec=FieldSpec(p=5, r=1, f=(0, 1)),
        q=2,
        q_factors=((2, 1),),
        g=(4,),
        h=(1,),
    )


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        pytest.param(((2,),), "each factor must be", id="malformed-pair"),
        pytest.param(((4, 1),), "distinct primes in increasing order", id="non-prime"),
        pytest.param(((2, 0),), "positive integers", id="invalid-exponent"),
        pytest.param(((2, 2),), "complete factorization", id="excess-factor"),
        pytest.param((), "complete factorization", id="missing-factor"),
    ],
)
def test_dlp_rejects_invalid_factorization(
    order_two_instance: DLPInstance, factors: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(order_two_instance, q_factors=factors)


def test_dlp_requires_order_greater_than_one(order_two_instance: DLPInstance) -> None:
    with pytest.raises(ValueError, match="q must be an integer > 1"):
        replace(order_two_instance, q=1)


@pytest.mark.parametrize("generator", [(2,), (1,)], ids=["order-too-large", "order-too-small"])
def test_dlp_requires_exact_generator_order(
    order_two_instance: DLPInstance, generator: FieldElement
) -> None:
    with pytest.raises(ValueError, match="q must be the order of g"):
        replace(order_two_instance, g=generator)


def test_dlp_rejects_target_outside_generated_subgroup(order_two_instance: DLPInstance) -> None:
    with pytest.raises(ValueError, match="h must belong to the subgroup"):
        replace(order_two_instance, h=(2,))  # 2 は {1, 4} に属さない。
