from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.simulation.classical_sim import ClassicalValT

from .exponentiation import FieldExponentiation
from .field import Bits, DLPInstance, FiniteField, encode, shared_field


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
        if type(self.exponent_bits) is not int or self.exponent_bits < 1:
            raise ValueError("exponent_bits must be a positive integer")

    @cached_property
    def field(self) -> FiniteField:
        return shared_field(self.instance.spec)

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
