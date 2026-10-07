from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, BQUInt, QUInt, Register, Signature, SoquetT
from qualtran.simulation.classical_sim import ClassicalValT

from src.field import FieldElement, FieldSpec, FiniteField, shared_field


@attrs.frozen(kw_only=True)
class WindowedProductAdd(Bloq):
    """
    |e>|x>|y> -> |e>|x>|y + k_e*x>  k_e = factors[e]
    """

    spec: FieldSpec
    factors: tuple[FieldElement, ...]
    mul_window_size: int

    def __attrs_post_init__(self) -> None:
        if self.spec.p == 2:
            raise ValueError("GF(2) is not supported in this class currently")
        if len(self.factors) < 2 or len(self.factors) & (len(self.factors) - 1):
            raise ValueError("number of factors must be a positive power of two")
        if not 1 <= self.mul_window_size <= self.spec.register_bits:
            raise ValueError("invalid mul_window_size")

    @cached_property
    def field(self) -> FiniteField:
        return shared_field(self.spec)

    @property
    def exp_window_size(self) -> int:
        return (len(self.factors) - 1).bit_length()

    @property
    def signature(self) -> Signature:
        n = self.spec.coefficient_bits
        return Signature(
            [
                Register(
                    "exp_window",
                    BQUInt((len(self.factors) - 1).bit_length(), len(self.factors)),
                ),
                Register("x", QUInt(n), shape=(self.spec.r,)),
                Register("y", QUInt(n), shape=(self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        pass

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        exp_window = int(vals["exp_window"])
        x = tuple(int(v) for v in np.asarray(vals["x"], dtype=object))
        y = tuple(int(v) for v in np.asarray(vals["y"], dtype=object))
        return {
            "exp_window": exp_window,
            "x": np.asarray(x, dtype=object),
            "y": np.asarray(
                self.field.add(y, self.field.mul(self.factors[exp_window], x)), dtype=object
            ),
        }
