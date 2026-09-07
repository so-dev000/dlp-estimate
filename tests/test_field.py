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
    coefficient_bits,
    decode,
    embedding_degree,
    encode,
)


@pytest.fixture
def gf25() -> FiniteField:
    return FiniteField(FieldSpec(p=5, r=2, f=(2, 0, 1)))  # GF(5²) = GF(25) with X² + 2 = 0


@pytest.mark.parametrize(
    ("p", "r", "f"),
    [
        (1, 1, (0, 1)),  # p < 2
        (5, 0, (1,)),  # r < 1
        (5, 2, (0, 1)),  # 次数の不一致
        (5, 2, (5, 0, 1)),  # 係数の範囲外
        (5, 2, (2, 0, 2)),  # 非モニック
        (5, 1, (1, 1)),  # 素体の f は X に固定
    ],
)
def test_field_spec_rejects_invalid_values(p: int, r: int, f: PolynomialCoefficients) -> None:
    with pytest.raises(ValueError):
        FieldSpec(p=p, r=r, f=f)


@pytest.mark.parametrize(
    "spec",
    [
        FieldSpec(p=4, r=1, f=(0, 1)),  # p が素数でない
        FieldSpec(p=5, r=2, f=(4, 0, 1)),  # f が既約でない
    ],
    ids=["composite-characteristic", "reducible-polynomial"],
)
def test_field_construction_rejects_invalid_spec(spec: FieldSpec) -> None:
    with pytest.raises(ValueError):
        FiniteField(spec)


def test_prime_field_arithmetic() -> None:
    field = FiniteField(FieldSpec(p=5, r=1, f=(0, 1)))
    assert field.add((2,), (4,)) == (1,)  # 2 + 4 = 6 ≡ 1 (mod 5)
    assert field.sub((2,), (4,)) == (3,)  # 2 - 4 = -2 ≡ 3 (mod 5)
    assert field.neg((2,)) == (3,)  # -2 ≡ 3 (mod 5)
    assert field.mul((2,), (4,)) == (3,)  # 2 * 4 = 8 ≡ 3 (mod 5)
    assert field.inv((2,)) == (3,)  # 2⁻¹ ≡ 3 (mod 5)
    assert field.pow((2,), 4) == (1,)  # 2⁴ = 16 ≡ 1 (mod 5)


def test_extension_field_arithmetic(gf25: FiniteField) -> None:
    # X² + 2 = 0 より X² = 3、X⁻¹ = 2X。
    assert gf25.spec.order == 25
    assert gf25.zero == (0, 0)
    assert gf25.one == (1, 0)
    assert gf25.add((3, 1), (0, 1)) == (3, 2)
    assert gf25.sub((3, 1), (0, 1)) == (3, 0)
    assert gf25.neg((3, 1)) == (2, 4)
    assert gf25.mul((3, 1), (0, 1)) == (3, 3)
    assert gf25.inv((0, 1)) == (0, 2)
    assert gf25.pow((0, 1), -1) == (0, 2)
    # 列は X·1 と X·X の係数列。
    assert gf25.const_mul_matrix((0, 1)) == ((0, 3), (1, 0))


def test_galois_conversion_preserves_coefficient_order(gf25: FiniteField) -> None:
    scalar = gf25.to_galois((3, 1))
    assert scalar.ndim == 0
    assert scalar.vector().tolist() == [1, 3]  # galois は高次数順
    assert gf25.from_galois(scalar) == (3, 1)
    vector = type(scalar)(scalar, ndmin=1)
    with pytest.raises(ValueError):
        gf25.from_galois(vector)


def test_zero_powers_and_inverse(gf25: FiniteField) -> None:
    assert gf25.pow(gf25.zero, 0) == gf25.one
    with pytest.raises(ZeroDivisionError):
        gf25.inv(gf25.zero)
    with pytest.raises(ZeroDivisionError):
        gf25.pow(gf25.zero, -1)


@pytest.mark.parametrize("element", [(1,), (5, 0)])
def test_rejects_invalid_element(gf25: FiniteField, element: FieldElement) -> None:
    with pytest.raises(ValueError):
        gf25.to_galois(element)
    with pytest.raises(ValueError):
        encode(element, gf25.spec)


def test_encoding_uses_low_degree_coefficients_and_big_endian_bits(gf25: FiniteField) -> None:
    bits = (0, 1, 1, 0, 0, 1)  # (3, 1) → (011, 001)
    assert coefficient_bits(gf25.spec) == 3
    assert encode((3, 1), gf25.spec) == bits
    assert decode(bits, gf25.spec) == (3, 1)


def test_binary_field_uses_two_bits_per_coefficient() -> None:
    field = FiniteField(FieldSpec(p=2, r=1, f=(0, 1)))
    assert field.add((1,), (1,)) == (0,)
    assert coefficient_bits(field.spec) == 2
    assert encode((1,), field.spec) == (0, 1)
    assert decode((0, 1), field.spec) == (1,)
    with pytest.raises(ValueError):
        decode((1, 0), field.spec)


@pytest.mark.parametrize(
    "bits",
    [(0, 0, 0), (0, 0, 2, 0, 0, 0), (1, 0, 1, 0, 0, 0)],
    ids=["wrong-length", "non-bit", "coefficient-out-of-range"],
)
def test_decode_rejects_invalid_bits(gf25: FiniteField, bits: Bits) -> None:
    with pytest.raises(ValueError):
        decode(bits, gf25.spec)


@pytest.mark.parametrize(
    ("spec", "q", "factors", "g", "h", "log", "degree"),
    [
        (FieldSpec(p=5, r=1, f=(0, 1)), 4, ((2, 2),), (2,), (3,), 3, 1),
        (FieldSpec(p=2, r=2, f=(1, 1, 1)), 3, ((3, 1),), (0, 1), (1, 1), 2, 2),
        (FieldSpec(p=5, r=2, f=(2, 0, 1)), 8, ((2, 3),), (0, 1), (0, 3), 3, 2),
        (FieldSpec(p=5, r=2, f=(2, 0, 1)), 4, ((2, 2),), (2, 0), (3, 0), 3, 1),
    ],
    ids=["gf5", "gf4", "gf25", "gf25-prime-subfield"],
)
def test_dlp_and_embedding_degree(
    spec: FieldSpec,
    q: int,
    factors: Factorization,
    g: FieldElement,
    h: FieldElement,
    log: int,
    degree: int,
) -> None:
    instance = DLPInstance(field=spec, q=q, q_factors=factors, g=g, h=h)
    assert FiniteField(spec).pow(g, log) == h
    assert embedding_degree(instance) == degree


def test_dlp_validates_on_construction() -> None:
    instance = DLPInstance(
        field=FieldSpec(p=5, r=1, f=(0, 1)), q=2, q_factors=((2, 1),), g=(4,), h=(1,)
    )
    with pytest.raises(ValueError, match="complete factorization"):
        replace(instance, q_factors=((2, 2),))
    with pytest.raises(ValueError, match="order of g"):
        replace(instance, g=(2,))  # g² ≠ 1
    with pytest.raises(ValueError, match="order of g"):
        replace(instance, g=(1,))  # g² = 1 でも位数は 1
    with pytest.raises(ValueError, match="subgroup"):
        replace(instance, h=(2,))  # 2 ∉ {1, 4}
