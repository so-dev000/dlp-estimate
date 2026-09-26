from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QUInt, Register, Signature, SoquetT
from qualtran.simulation.classical_sim import ClassicalValT

from .arithmetic import get_controlled_const_mul
from .field import FieldElement, FieldSpec, FiniteField, shared_field


def _validate_exponentiation_inputs(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> None:
    if type(exponent_bits) is not int or exponent_bits < 1:
        raise ValueError("exponent_bits must be a positive integer")
    if field.is_zero(base):
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
        _validate_exponentiation_inputs(self.field, self.base, self.exponent_bits)

    @cached_property
    def field(self) -> FiniteField:
        return shared_field(self.spec)

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
