from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, CBit, QBit, QUInt, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, MeasureZ, XGate
from qualtran.bloqs.qft import QFTTextBook
from qualtran.simulation.classical_sim import ClassicalValT

from .arithmetic import get_controlled_const_mul
from .field import Bits, DLPInstance, FieldElement, FieldSpec, FiniteField, encode


@attrs.frozen(kw_only=True)
class ShorConfig:
    exponent_bits: int  # 2つの指数レジスタに共通のビット幅

    def __attrs_post_init__(self) -> None:
        if type(self.exponent_bits) is not int or self.exponent_bits < 1:
            raise ValueError("exponent_bits must be a positive integer")


def make_shor_config(
    instance: DLPInstance,
    *,
    exponent_bits: int | None = None,
) -> ShorConfig:
    minimum_bits = (instance.q * instance.q).bit_length()
    if exponent_bits is None:
        exponent_bits = minimum_bits

    config = ShorConfig(exponent_bits=exponent_bits)
    if config.exponent_bits < minimum_bits:
        raise ValueError(f"exponent_bits must be at least {minimum_bits} to satisfy 2**m > q**2")
    return config


def _validate_exponentiation_inputs(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> None:
    if type(exponent_bits) is not int or exponent_bits < 1:
        raise ValueError("exponent_bits must be a positive integer")
    if field.to_galois(base) == 0:
        raise ValueError("base must be nonzero")


def _squared_constants(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> tuple[FieldElement, ...]:
    constants = []
    constant = base
    for _ in range(exponent_bits):
        constants.append(constant)
        constant = field.mul(constant, constant)
    return tuple(constants)


@attrs.frozen(kw_only=True)
class FieldExponentiation(Bloq):
    """
    |a>|x> -> |a>|x * base^a>
    指数 a を
        a = Σ a_i 2^i,  a_i ∈ {0, 1}
    と2進展開し、
        base^a = Π (base^(2^i))^a_i
    を用いて、各ビット a_i = 1 のときだけ
    x に定数 base^(2^i) を乗算する。
    |a>|x>
      -> |a>|x * base^(a_0 2^0)>
      -> |a>|x * base^(a_0 2^0 + a_1 2^1)>
      -> ...
      -> |a>|x * base^a>
    """

    spec: FieldSpec
    base: FieldElement
    exponent_bits: int

    def __attrs_post_init__(self) -> None:
        _ = self.field
        _validate_exponentiation_inputs(self.field, self.base, self.exponent_bits)

    @cached_property
    def field(self) -> FiniteField:
        return FiniteField(self.spec)

    @property
    def signature(self) -> Signature:
        n = self.spec.coefficient_bits
        return Signature(
            [
                Register("exponent", QUInt(self.exponent_bits)),
                Register("x", QUInt(n), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        bits = bb.split(soqs["exponent"])
        x = soqs["x"]
        assert isinstance(bits, np.ndarray)

        constants = _squared_constants(self.field, self.base, self.exponent_bits)

        for i, constant in enumerate(constants):
            # 指数はbig-endianなので、第i定数 base**(2**i) は bits[-1-i] で指定する
            bits[-1 - i], x = bb.add_t(
                get_controlled_const_mul(self.spec, constant),
                ctrl=bits[-1 - i],
                x=x,
            )

        return {
            "exponent": bb.join(bits, dtype=QUInt(self.exponent_bits)),
            "x": x,
        }

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        r, p = self.spec.r, self.spec.p
        exponent = int(vals["exponent"])

        # チェック
        if not 0 <= exponent < 2**self.exponent_bits:
            raise ValueError(f"exponent must be in [0, {2**self.exponent_bits})")

        x = np.asarray(vals["x"])
        if x.shape != (r,):
            raise ValueError(f"x must have shape ({r},), got {x.shape}")
        coefficients = tuple(int(v) for v in x)
        if any(not 0 <= coefficient < p for coefficient in coefficients):
            raise ValueError(f"x coefficients must be in [0, {p})")

        # 古典計算
        power = self.field.pow(self.base, exponent)
        result = self.field.mul(power, coefficients)
        return {"exponent": exponent, "x": np.array(result, dtype=object)}

    def adjoint(self) -> FieldExponentiation:
        return FieldExponentiation(
            spec=self.spec,
            base=self.field.inv(self.base),
            exponent_bits=self.exponent_bits,
        )


def _x_encoded_bits(bb: BloqBuilder, w: np.ndarray, encoded: Bits, bitsize: int) -> np.ndarray:
    """係数レジスタ列wのうち、encodedが1のビット位置にXを適用する。"""
    for position, bit in enumerate(encoded):
        if not bit:
            continue
        j, k = divmod(position, bitsize)
        bits = bb.split(w[j])
        bits[k] = bb.add(XGate(), q=bits[k])
        w[j] = bb.join(bits, dtype=QUInt(bitsize))
    return w


@attrs.frozen(kw_only=True)
class DLPOracle(Bloq):
    """
    |a>|b>|y> -> |a>|b>|y ⊕ encode(h^a g^b)>
    wを作業レジスタとして使用
    |a>|b>|w=0>|y>
      -> |a>|b>|w=1>|y>
      -> |a>|b>|w=h^a>|y>
      -> |a>|b>|w=h^a g^b>|y>
      -> |a>|b>|w=h^a g^b>|y ⊕ encode(h^a g^b)>
      -> |a>|b>|w=h^a>|y ⊕ encode(h^a g^b)>
      -> |a>|b>|w=1>|y ⊕ encode(h^a g^b)>
      -> |a>|b>|w=0>|y ⊕ encode(h^a g^b)>
    """

    instance: DLPInstance
    exponent_bits: int

    def __attrs_post_init__(self) -> None:
        _ = self.field
        if type(self.exponent_bits) is not int or self.exponent_bits < 1:
            raise ValueError("exponent_bits must be a positive integer")

    @cached_property
    def field(self) -> FiniteField:
        return FiniteField(self.instance.spec)

    @property
    def signature(self) -> Signature:
        n = self.instance.spec.coefficient_bits
        return Signature(
            [
                Register("a", QUInt(self.exponent_bits)),
                Register("b", QUInt(self.exponent_bits)),
                Register("y", QBit(), (self.instance.spec.r * n,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        a = soqs["a"]
        b = soqs["b"]
        y = soqs["y"]
        assert isinstance(y, np.ndarray)

        spec = self.instance.spec
        n = spec.coefficient_bits
        r = spec.r
        m = self.exponent_bits

        # 作業レジスタw を0で確保
        w = np.array([bb.allocate(dtype=QUInt(n)) for _ in range(r)], dtype=object)
        encoded_one = encode(
            self.field.one, spec
        )  # (1, 0, 0, ..., 0) -> (001, 000, ..., 000) -> (0,0,1,0,0,0,...,0,0,0)
        w = _x_encoded_bits(bb, w, encoded_one, n)

        # |a>|w> -> |a>|h^a> を計算する
        a, w = bb.add_t(
            FieldExponentiation(spec=spec, base=self.instance.h, exponent_bits=m),
            exponent=a,
            x=w,
        )
        # |b>|w> -> |b>|w * g^b> = |b>|h^a g^b> を計算する
        b, w = bb.add_t(
            FieldExponentiation(spec=spec, base=self.instance.g, exponent_bits=m),
            exponent=b,
            x=w,
        )

        assert isinstance(w, np.ndarray)

        # |w>|y> -> |w>|y ⊕ encode(w)>を計算する
        for j in range(r):
            bits = bb.split(w[j])
            for k in range(n):
                bits[k], y[j * n + k] = bb.add_t(
                    CNOT(),
                    ctrl=bits[k],
                    target=y[j * n + k],
                )
            w[j] = bb.join(bits, dtype=QUInt(n))

        # uncompute: |b>|w=h^a g^b> -> |b>|h^a>を計算する
        b, w = bb.add_t(
            FieldExponentiation(spec=spec, base=self.instance.g, exponent_bits=m).adjoint(),
            exponent=b,
            x=w,
        )
        # uncompute: |a>|w=h^a> -> |a>|1>を計算する
        a, w = bb.add_t(
            FieldExponentiation(spec=spec, base=self.instance.h, exponent_bits=m).adjoint(),
            exponent=a,
            x=w,
        )

        assert isinstance(w, np.ndarray)

        # uncompute: |w=1> -> |w=0>を計算する
        w = _x_encoded_bits(bb, w, encoded_one, n)
        for soq in w:
            bb.free(soq)

        return {"a": a, "b": b, "y": y}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        m = self.exponent_bits
        r, n = self.instance.spec.r, self.instance.spec.coefficient_bits
        a = int(vals["a"])
        b = int(vals["b"])

        # チェック
        if not 0 <= a < 2**m:
            raise ValueError(f"a must be in [0, {2**m})")
        if not 0 <= b < 2**m:
            raise ValueError(f"b must be in [0, {2**m})")

        y = np.asarray(vals["y"])
        if y.shape != (r * n,):
            raise ValueError(f"y must have shape ({r * n},), got {y.shape}")
        y_bits = tuple(int(v) for v in y)
        if any(bit not in (0, 1) for bit in y_bits):
            raise ValueError("y must contain only the integers 0 and 1")

        # 古典計算
        function_value = self.field.mul(
            self.field.pow(self.instance.h, a),
            self.field.pow(self.instance.g, b),
        )
        encoded = encode(function_value, self.instance.spec)
        result = tuple(bit ^ mask for bit, mask in zip(y_bits, encoded, strict=True))  # XOR
        return {"a": a, "b": b, "y": np.array(result, dtype=object)}

    def adjoint(self) -> DLPOracle:
        return self


@attrs.frozen(kw_only=True)
class ShorDLP(Bloq):
    """
    Nielsen-Chuang5.4.2節のShor-DLP実装

    |a=0>|b=0>|y=0>
          -> H^{⊗m}をa, bに適用
          -> (1/2^m) Σ_{a,b} |a>|b>|y=0>
          -> DLPOracleを適用
          -> (1/2^m) Σ_{a,b} |a>|b>|y=encode(h^a g^b)>
          -> QFT^{-1}をa, bに適用
          -> a, bを測定
          -> |A>|B>|y>
    """

    instance: DLPInstance
    config: ShorConfig

    @cached_property
    def field(self) -> FiniteField:
        return FiniteField(self.instance.spec)

    @property
    def signature(self) -> Signature:
        m = self.config.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r
        return Signature(
            [
                Register("a", CBit(), (m,), Side.RIGHT),
                Register("b", CBit(), (m,), Side.RIGHT),
                Register("y", QBit(), (r * n,), Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        m = self.config.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r

        # 初期状態 |0>|0>|0>
        a = bb.allocate(dtype=QUInt(m))
        b = bb.allocate(dtype=QUInt(m))
        y = np.array(
            [bb.allocate(dtype=QBit()) for _ in range(r * n)],
            dtype=object,
        )

        # 一様重ね合わせ |0>|0>|0> -> (1/2^m) Σ_{a,b} |a>|b>|0>
        a_bits = bb.split(a)
        b_bits = bb.split(b)
        for i in range(m):
            a_bits[i] = bb.add(Hadamard(), q=a_bits[i])
            b_bits[i] = bb.add(Hadamard(), q=b_bits[i])
        a = bb.join(a_bits, dtype=QUInt(m))
        b = bb.join(b_bits, dtype=QUInt(m))

        # DLPOracleを適用 |a>|b>|y=0> -> |a>|b>|encode(h^a g^b)>
        a, b, y = bb.add_t(DLPOracle(instance=self.instance, exponent_bits=m), a=a, b=b, y=y)

        # 逆QFTを適用
        (a,) = bb.add_t(QFTTextBook(bitsize=m, with_reverse=True).adjoint(), q=a)
        (b,) = bb.add_t(QFTTextBook(bitsize=m, with_reverse=True).adjoint(), q=b)

        # Z測定
        a_bits = bb.split(a)
        b_bits = bb.split(b)
        a_clas = np.empty(m, dtype=object)
        b_clas = np.empty(m, dtype=object)
        for i in range(m):
            a_clas[i] = bb.add(MeasureZ(), q=a_bits[i])
            b_clas[i] = bb.add(MeasureZ(), q=b_bits[i])
        return {"a": a_clas, "b": b_clas, "y": y}
