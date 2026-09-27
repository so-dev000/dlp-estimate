from dataclasses import dataclass
from functools import cache, lru_cache

from sympy import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import (
    gf_gcdex,
    gf_irreducible_p,
    gf_mul,
    gf_pow_mod,
    gf_rem,
    gf_strip,
)

type FieldElement = tuple[int, ...]  # (a_0, a_1, ..., a_{r-1})
type PolynomialCoefficients = tuple[int, ...]  # (c_0, c_1, ..., c_r)
type FieldMatrix = tuple[FieldElement, ...]
type Bits = tuple[int, ...]
type Factorization = tuple[tuple[int, int], ...]  # (prime, exponent)


@dataclass(frozen=True, kw_only=True)
class FieldSpec:
    """GF(p^r)の仕様。素数性・既約性はFiniteField.validate()で別途検証する。"""

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

    @property
    def coefficient_bits(self) -> int:
        """標数2では1ビット、奇標数では法pも表現できるp.bit_length()ビットを返す。"""
        return 1 if self.p == 2 else self.p.bit_length()

    @property
    def field_bits(self) -> int:
        """ceil(log2(p**r)) を返す。"""
        return (self.order - 1).bit_length()


def format_polynomial(coeffs: PolynomialCoefficients) -> str:
    """低次数順の係数tupleを多項式文字列表現にする。例: (2, 0, 1) -> "2 + X^2"。"""
    terms = []
    for power, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        if power == 0:
            terms.append(str(coeff))
        elif power == 1:
            terms.append("X" if coeff == 1 else f"{coeff}*X")
        else:
            terms.append(f"X^{power}" if coeff == 1 else f"{coeff}*X^{power}")
    return " + ".join(terms) if terms else "0"


def _validate_element(a: FieldElement, spec: FieldSpec) -> None:
    if not isinstance(a, tuple) or len(a) != spec.r:
        raise ValueError(f"field element must be a tuple of length {spec.r}")
    if any(type(c) is not int or not 0 <= c < spec.p for c in a):
        raise ValueError(f"coefficients must be integers in [0, {spec.p})")


class FiniteField:
    """検証済みのp/fを用い、原始元探索なしで剰余演算する有限体。"""

    def __init__(self, spec: FieldSpec) -> None:
        """軽量に構築する。体の検証は推定入口で明示的に行う。"""
        self.spec = spec
        self._modulus = list(reversed(spec.f))

    def validate(self) -> None:
        """pの素数性とfの既約性を必要時に明示的に検証する。"""
        if not isprime(self.spec.p):
            raise ValueError(f"p must be prime, got {self.spec.p}")
        if not gf_irreducible_p(self._modulus, self.spec.p, ZZ):
            raise ValueError("f must be irreducible")

    def _poly(self, a: FieldElement) -> list[int]:
        """public演算で検証済みの要素をSymPy形式へ変換する。"""
        return gf_strip(list(reversed(a)))

    def _from_poly(self, a: list[int]) -> FieldElement:
        return tuple(reversed(a)) + (0,) * (self.spec.r - len(a))

    def is_zero(self, a: FieldElement) -> bool:
        """体の構築を伴わず、係数の形と零かどうかを確認する。"""
        _validate_element(a, self.spec)
        return not any(a)

    @property
    def zero(self) -> FieldElement:
        """加法単位元を返す。"""
        return (0,) * self.spec.r  # (0, 0, ..., 0) : 0 + 0*X + ... + 0*X^(r-1) = 0

    @property
    def one(self) -> FieldElement:
        """乗法単位元を返す。"""
        return (1,) + (0,) * (self.spec.r - 1)  # (1, 0, ..., 0) : 1 + 0*X + ... + 0*X^(r-1) = 1

    def add(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """和 a + b を返す。"""
        _validate_element(a, self.spec)
        _validate_element(b, self.spec)
        return tuple((x + y) % self.spec.p for x, y in zip(a, b, strict=True))

    def sub(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """差 a - b を返す。"""
        _validate_element(a, self.spec)
        _validate_element(b, self.spec)
        return tuple((x - y) % self.spec.p for x, y in zip(a, b, strict=True))

    def neg(self, a: FieldElement) -> FieldElement:
        """加法逆元 -a を返す。"""
        _validate_element(a, self.spec)
        return tuple(-x % self.spec.p for x in a)

    def mul(self, a: FieldElement, b: FieldElement) -> FieldElement:
        """積 a * b を返す。"""
        _validate_element(a, self.spec)
        _validate_element(b, self.spec)
        if self.spec.r == 1:
            return (a[0] * b[0] % self.spec.p,)
        product = gf_mul(self._poly(a), self._poly(b), self.spec.p, ZZ)
        return self._from_poly(gf_rem(product, self._modulus, self.spec.p, ZZ))

    @lru_cache(maxsize=4096)  # noqa: B019 - bounded; shared_field also retains these objects.
    def inv(self, a: FieldElement) -> FieldElement:
        """乗法逆元を返す。"""
        if self.is_zero(a):
            raise ZeroDivisionError("zero has no multiplicative inverse")
        if self.spec.r == 1:
            return (pow(a[0], -1, self.spec.p),)
        # 拡張Euclid法で求める。全群位数の計算や原始元は不要。
        inverse, _, gcd = gf_gcdex(self._poly(a), self._modulus, self.spec.p, ZZ)
        if gcd != [1]:
            raise ZeroDivisionError("element has no multiplicative inverse modulo f")
        return self._from_poly(inverse)

    def pow(self, a: FieldElement, exponent: int) -> FieldElement:
        """整数乗を返す。負の指数は非零要素に限り、0**0は1とする。"""
        if type(exponent) is not int:
            raise ValueError("exponent must be an integer")
        _validate_element(a, self.spec)
        if exponent < 0:
            a, exponent = self.inv(a), -exponent
        if self.spec.r == 1:
            return (pow(a[0], exponent, self.spec.p),)
        return self._from_poly(gf_pow_mod(self._poly(a), exponent, self._modulus, self.spec.p, ZZ))

    @lru_cache(maxsize=4096)  # noqa: B019 - bounded; shared_field also retains these objects.
    def const_mul_matrix(self, c: FieldElement) -> FieldMatrix:
        """GF(p)上の定数乗算行列を返す。第j列は c * X^j mod f の係数列。"""
        _validate_element(c, self.spec)
        r = self.spec.r
        columns = [c]
        # c*X^jからXを一つ掛けて次の列を得る。一般の乗算をr回繰り返さない。
        for _ in range(1, r):
            previous = columns[-1]
            shifted = (0, *previous[:-1])
            columns.append(
                tuple((shifted[i] - previous[-1] * self.spec.f[i]) % self.spec.p for i in range(r))
            )
        return tuple(tuple(column[i] for column in columns) for i in range(r))


@cache
def shared_field(spec: FieldSpec) -> FiniteField:
    return FiniteField(spec)


def encode(element: FieldElement, spec: FieldSpec) -> Bits:
    """
    係数を低次数順に、各係数をspec.coefficient_bitsビットのbig-endianで符号化する。
    例: p=5, r=3, element=(3, 0, 4) の場合、(3, 0, 4) -> (011, 000, 100) -> (0,1,1,0,0,0,1,0,0)
    """
    n = spec.coefficient_bits
    _validate_element(element, spec)
    return tuple((a >> shift) & 1 for a in element for shift in range(n - 1, -1, -1))


def _validate_factorization(
    q_factors: Factorization | None,
    q: int,
) -> None:
    """q_factors が与えられた場合、q の完全素因数分解であることを確認する。"""
    if q_factors is None:
        return

    if not isinstance(q_factors, tuple) or not q_factors:
        raise ValueError("q_factors must be a non-empty tuple or None")

    product = 1
    previous = 0

    for item in q_factors:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("q_factors entries must be (prime, exponent) tuples")

        prime, exponent = item

        if type(prime) is not int or type(exponent) is not int:
            raise ValueError("q_factors prime and exponent must be integers")
        if exponent < 1:
            raise ValueError("q_factors exponents must be positive")
        if prime <= previous:
            raise ValueError("q_factors primes must be ascending without duplicates")
        if not isprime(prime):
            raise ValueError(f"q_factors {prime} is not prime")

        previous = prime
        product *= prime**exponent

    if product != q:
        raise ValueError("q_factors must multiply to q")


@dataclass(frozen=True, kw_only=True)
class DLPInstance:
    """
    g**d = h を解くDLP入力。

    q_factors が与えられている場合は q の完全因数分解として扱い、
    g の位数が厳密に q であることを検証できる。

    q_factors=None の場合は完全因数分解が未知であることを表し、
    exact-order 検証を省略する。
    """

    spec: FieldSpec
    q: int
    g: FieldElement
    h: FieldElement
    q_factors: Factorization | None = None

    def __post_init__(self) -> None:
        """型・形・非零条件と、与えられた素因数分解を軽量に検証する。"""
        if type(self.q) is not int or self.q <= 1:
            raise ValueError("q must be an integer > 1")

        _validate_factorization(self.q_factors, self.q)

        _validate_element(self.g, self.spec)
        _validate_element(self.h, self.spec)

        if not any(self.g) or not any(self.h):
            raise ValueError("g and h must be nonzero")

    def validate(self) -> None:
        """
        gの位数・hの部分群所属を検証する。

        q_factors が与えられている場合は exact order まで検証する。
        q_factors=None の場合は q の内部因数分解を行わず、
        exact-order 検証を省略する。
        """
        field = shared_field(self.spec)

        if field.pow(self.g, self.q) != field.one:
            raise ValueError("q must be the order of g")

        if self.q_factors is not None:
            for prime, _ in self.q_factors:
                if field.pow(self.g, self.q // prime) == field.one:
                    raise ValueError("q must be the order of g")

        if field.pow(self.h, self.q) != field.one:
            raise ValueError("h must belong to the subgroup generated by g")
