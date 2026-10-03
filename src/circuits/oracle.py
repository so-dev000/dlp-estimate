from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QUInt, Register, Signature, SoquetT
from qualtran.simulation.classical_sim import ClassicalValT

from ..field import DLPInstance, FiniteField, shared_field
from .exponentiation import DeprecatedFieldExponentiation


@attrs.frozen(kw_only=True)
class DeprecatedDLPOracle(Bloq):
    """
    Deprecated: 旧構成 (教科書版 ShorDLP 用)
    |a>|b>|y> -> |a>|b>|y * h^a g^b>。
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
                Register("y", QUInt(n), (self.instance.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        a = soqs["a"]
        b = soqs["b"]
        y = soqs["y"]

        m = self.exponent_bits
        spec = self.instance.spec

        # |a>|y> -> |a>|y * h^a> を直接 y 上で計算する
        a, y = bb.add_t(
            DeprecatedFieldExponentiation(spec=spec, base=self.instance.h, exponent_bits=m),
            exponent=a,
            x=y,
        )
        # |b>|y> -> |b>|y * g^b> = |b>|x * h^a g^b> を直接 y 上で計算する
        b, y = bb.add_t(
            DeprecatedFieldExponentiation(spec=spec, base=self.instance.g, exponent_bits=m),
            exponent=b,
            x=y,
        )

        return {"a": a, "b": b, "y": y}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        m = self.exponent_bits
        r, p = self.instance.spec.r, self.instance.spec.p
        a = int(vals["a"])
        b = int(vals["b"])

        # チェック
        if not 0 <= a < 2**m:
            raise ValueError(f"a must be in [0, {2**m})")
        if not 0 <= b < 2**m:
            raise ValueError(f"b must be in [0, {2**m})")

        y = np.asarray(vals["y"])
        if y.shape != (r,):
            raise ValueError(f"y must have shape ({r},), got {y.shape}")
        coefficients = tuple(int(v) for v in y)
        if any(not 0 <= coefficient < p for coefficient in coefficients):
            raise ValueError(f"y coefficients must be in [0, {p})")

        # 古典計算: y * h^a g^b
        function_value = self.field.mul(
            self.field.pow(self.instance.h, a),
            self.field.pow(self.instance.g, b),
        )
        result = self.field.mul(coefficients, function_value)
        return {"a": a, "b": b, "y": np.array(result, dtype=object)}
