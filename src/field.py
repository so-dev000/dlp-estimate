from dataclasses import dataclass
from functools import lru_cache

import galois
from sympy import factorint, isprime

type FieldElement = tuple[int, ...]  # (a_0, a_1, ..., a_{r-1})
type PolynomialCoefficients = tuple[int, ...]  # (c_0, c_1, ..., c_r)
type FieldMatrix = tuple[FieldElement, ...]
type Bits = tuple[int, ...]
type Factorization = tuple[tuple[int, int], ...]  # (prime, exponent)


@dataclass(frozen=True, kw_only=True)
class FieldSpec:
    """GF(p^r)の仕様。素数性・既約性の検証は体の構築時に行う。"""

    p: int
    r: int
    f: PolynomialCoefficients

    def __post_init__(self) -> None:
        if type(self.p) is not int or self.p < 2:
            raise ValueError("p must be an integer >= 2")
        if type(self.r) is not int or self.r < 1:
            raise ValueError("r must be a positive integer")
        if not isinstance(self.f, tuple) or len(self.f) != self.r + 1:
            raise ValueError(f"f must be a coefficient tuple of length {self.r + 1}")
        if any(type(c) is not int or not 0 <= c < self.p for c in self.f):
            raise ValueError(f"f coefficients must be integers in [0, {self.p})")
        if self.f[-1] != 1:
            raise ValueError("f must be monic")
        if self.r == 1 and self.f != (0, 1):
            raise ValueError("f must be (0, 1) for a prime field")

    @property
    def order(self) -> int:
        """体の要素数 p**r を返す。"""
        return self.p**self.r


@lru_cache(maxsize=32)
def _galois_field(spec: FieldSpec) -> type[galois.FieldArray]:
    """pの素数性を検証し、既約性の検証をgaloisに任せて体を構築する。"""
    if not isprime(spec.p):
        raise ValueError(f"p must be prime, got {spec.p}")
    prime_field = galois.GF(spec.p, 1, compile="python-calculate")
    if spec.r == 1:
        return prime_field

    poly = galois.Poly(spec.f, field=prime_field, order="asc")
    return galois.GF(spec.p, spec.r, irreducible_poly=poly, compile="python-calculate")


def _validate_element(a: FieldElement, spec: FieldSpec) -> None:
    if not isinstance(a, tuple) or len(a) != spec.r:
        raise ValueError(f"field element must be a tuple of length {spec.r}")
    if any(type(c) is not int or not 0 <= c < spec.p for c in a):
        raise ValueError(f"coefficients must be integers in [0, {spec.p})")


class FiniteField:
    """要素を長さr、係数が[0, p)の低次数順tupleで扱う有限体。"""

    def __init__(self, spec: FieldSpec) -> None:
        """pの素数性とfの既約性を検証して体を構築する。"""
        self.spec = spec
        self._galois_field = _galois_field(spec)

    @property
    def zero(self) -> FieldElement:
        """加法単位元を返す。"""
        return (0,) * self.spec.r  # (0, 0, ..., 0) : 0 + 0*X + ... + 0*X^(r-1) = 0

    @property
    def one(self) -> FieldElement:
        """乗法単位元を返す。"""
        return (1,) + (0,) * (self.spec.r - 1)  # (1, 0, ..., 0) : 1 + 0*X + ... + 0*X^(r-1) = 1

    def to_galois(self, a: FieldElement) -> galois.FieldArray:
        """低次数順の係数tupleを、この体のgaloisスカラーへ変換する。"""
        _validate_element(a, self.spec)
        return self._galois_field.Vector(list(reversed(a)))

    def from_galois(self, x: galois.FieldArray) -> FieldElement:
        """この体のgaloisスカラーを低次数順の係数tupleへ変換する。"""
        if not isinstance(x, self._galois_field) or x.ndim != 0:
            raise ValueError("x must be a galois scalar belonging to this field")
        return tuple(int(c) for c in reversed(x.vector()))

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """和 a + b を返す。"""
        return self.from_galois(self.to_galois(a) + self.to_galois(b))

    def sub(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """差 a - b を返す。"""
        return self.from_galois(self.to_galois(a) - self.to_galois(b))

    def neg(self, a: FieldElement) -> FieldElement:
        """加法逆元 -a を返す。"""
        return self.from_galois(-self.to_galois(a))

    def mul(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """積 a * b を返す。"""
        return self.from_galois(self.to_galois(a) * self.to_galois(b))

    def inv(self, a: FieldElement) -> FieldElement:
        """乗法逆元を返す。"""
        return self.from_galois(self.to_galois(a) ** -1)

    def pow(self, a: FieldElement, exponent: int) -> FieldElement:
        """整数乗を返す。負の指数は非零要素に限り、0**0は1とする。"""
        if type(exponent) is not int:
            raise ValueError("exponent must be an integer")
        value = self.to_galois(a)
        return self.from_galois(value**exponent)

    def const_mul_matrix(self, c: FieldElement) -> FieldMatrix:
        """GF(p)上の定数乗算行列を返す。第j列は c * X^j mod f の係数列。"""
        value = self.to_galois(c)
        r = self.spec.r
        columns = []
        for j in range(r):
            # 高次数順のVectorで、X^jに対応する位置だけ1にする。
            basis = self._galois_field.Vector([0] * (r - j - 1) + [1] + [0] * j)
            columns.append(self.from_galois(value * basis))
        return tuple(tuple(column[i] for column in columns) for i in range(r))


def coefficient_bits(spec: FieldSpec) -> int:
    """1係数のビット幅を返す。"""
    return spec.p.bit_length()


def encode(element: FieldElement, spec: FieldSpec) -> Bits:
    """
    係数を低次数順に、各係数をp.bit_length()ビットのbig-endianで符号化する。
    例: p=5, r=3, element=(3, 0, 4) の場合、(3, 0, 4) -> (011, 000, 100) -> (0,1,1,0,0,0,1,0,0)
    """
    n = coefficient_bits(spec)
    _validate_element(element, spec)
    return tuple((a >> shift) & 1 for a in element for shift in range(n - 1, -1, -1))


def decode(bits: Bits, spec: FieldSpec) -> FieldElement:
    """encodeされた符号を復元する。"""
    n = coefficient_bits(spec)
    if not isinstance(bits, tuple) or len(bits) != spec.r * n:
        raise ValueError(f"bits must be a tuple of length {spec.r * n}")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only the integers 0 and 1")

    coefficients = []
    for start in range(0, len(bits), n):
        value = 0
        for bit in bits[start : start + n]:
            value = (value << 1) | bit
        if value >= spec.p:
            raise ValueError(f"decoded coefficients must be less than {spec.p}")
        coefficients.append(value)
    return tuple(coefficients)


def _validate_factorization(q: int, factors: Factorization) -> None:
    remaining = q
    previous_prime = 1
    for pair in factors:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each factor must be a (prime, exponent) tuple")
        prime, exponent = pair
        if type(prime) is not int or prime <= previous_prime or not isprime(prime):
            raise ValueError("q_factors must contain distinct primes in increasing order")
        if type(exponent) is not int or exponent < 1:
            raise ValueError("factor exponents must be positive integers")
        for _ in range(exponent):
            remaining, remainder = divmod(remaining, prime)
            if remainder:
                raise ValueError("q_factors must be the complete factorization of q")
        previous_prime = prime
    if remaining != 1:
        raise ValueError("q_factors must be the complete factorization of q")


@dataclass(frozen=True, kw_only=True)
class DLPInstance:
    """g**d = hを解く入力。生成時に体・位数・部分群所属を検証する。"""

    field: FieldSpec
    q: int  # gの位数
    q_factors: Factorization
    g: FieldElement
    h: FieldElement

    def __post_init__(self) -> None:
        """体・qの因数分解・gの位数q・hの部分群所属を検証する。"""
        if type(self.q) is not int or self.q <= 1:
            raise ValueError("q must be an integer > 1")
        _validate_factorization(self.q, self.q_factors)

        field = FiniteField(self.field)
        g = field.to_galois(self.g)
        h = field.to_galois(self.h)
        if g**self.q != 1:
            raise ValueError("q must be the order of g")
        for prime, _ in self.q_factors:
            if g ** (self.q // prime) == 1:
                raise ValueError("q must be the order of g")
        if h**self.q != 1:
            raise ValueError("h must belong to the subgroup generated by g")


def embedding_degree(instance: DLPInstance) -> int:
    """
    p^k ≡ 1 (mod q) を満たす最小の正整数 k (i.e. pのmod qでの位数) を返す。
    p^r = 1 (mod q) が成り立つことは DLPInstance の生成時に検証済みなので、
    k は r の約数であることを利用する。
    """
    p, r, q = instance.field.p, instance.field.r, instance.q

    degree = r
    for prime in factorint(r):
        prime = int(prime)
        while degree % prime == 0 and pow(p, degree // prime, q) == 1:
            degree //= prime
    return degree
