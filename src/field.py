from galois import GF, FieldArray
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

    def __init__(self, spec: FieldSpec) -> None:
        self.spec = spec

    @property
    def zero(self) -> FieldElement:
        return (0,)

    @property
    def one(self) -> FieldElement:
        return (1,)

    def to_galois(self, a: FieldElement) -> FieldArray:
        return GF(self.spec.p**self.spec.r, irreducible_poly=self.spec.f)(a)

    def from_galois(self, x: FieldArray) -> FieldElement:
        return tuple(int(xi) for xi in x)

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
