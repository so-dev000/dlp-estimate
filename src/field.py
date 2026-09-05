import galois
from sympy import isprime

type FieldElement = tuple[int, ...]
type PolynomialCoefficients = tuple[int, ...]


class FieldSpec:
    p: int
    r: int
    f: PolynomialCoefficients

    def __init__(self, p: int, r: int, f: PolynomialCoefficients) -> None:
        self.p = p
        self.r = r
        self.f = f
        if self.r == 1:
            self.f = (0, 1)  # 0 + 1*X = X
        self.validate()

    def validate(self) -> None:
        if isprime(self.p) is False:
            raise ValueError(f"p must be prime, got {self.p}")
        if self.r < 1:
            raise ValueError(f"r must be positive, got {self.r}")

    @property
    def order(self) -> int:
        return self.p**self.r


class FiniteField:
    spec: FieldSpec
    _galois_field: type[galois.FieldArray]

    def __init__(self, spec: FieldSpec) -> None:
        self.spec = spec
        if spec.r == 1:
            self._galois_field = galois.GF(spec.p)
        else:
            gf_p = galois.GF(spec.p)
            poly = galois.Poly(spec.f, field=gf_p, order="asc")
            self._galois_field = galois.GF(spec.p, spec.r, irreducible_poly=poly)

    # 加法単位元
    @property
    def zero(self) -> FieldElement:
        return (0,) * self.spec.r  # (0, 0, ..., 0) : 0 + 0*X + ... + 0*X^(r-1) = 0

    # 乗法単位元
    @property
    def one(self) -> FieldElement:
        return (1,) + (0,) * (self.spec.r - 1)  # (1, 0, ..., 0) : 1 + 0*X + ... + 0*X^(r-1) = 1

    def to_galois(self, a: FieldElement) -> galois.FieldArray:
        return self._galois_field.Vector(list(reversed(a)))

    def from_galois(self, x: galois.FieldArray) -> FieldElement:
        return tuple(int(c) for c in reversed(x.vector()))

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = self.to_galois(b)
        c_galois = a_galois + b_galois
        return self.from_galois(c_galois)

    def sub(self, a: FieldElement, b: FieldElement) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = self.to_galois(b)
        c_galois = a_galois - b_galois
        return self.from_galois(c_galois)

    def neg(self, a: FieldElement) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = -a_galois
        return self.from_galois(b_galois)

    def mul(self, a: FieldElement, b: FieldElement) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = self.to_galois(b)
        c_galois = a_galois * b_galois
        return self.from_galois(c_galois)

    def inv(self, a: FieldElement) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = a_galois**-1
        return self.from_galois(b_galois)

    def pow(self, a: FieldElement, exponent: int) -> FieldElement:
        a_galois = self.to_galois(a)
        b_galois = a_galois**exponent
        return self.from_galois(b_galois)
