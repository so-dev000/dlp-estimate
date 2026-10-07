import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, BQUInt, QUInt, Register, Signature, SoquetT
from qualtran.simulation.classical_sim import ClassicalValT

from src.circuits.windowed_product_add import WindowedProductAdd
from src.field import FieldElement, FieldSpec, shared_field


@attrs.frozen(kw_only=True)
class WindowedConstMul(Bloq):
    """
    |e>|x> -> |e>|k_e*x>, k_e = selected_factors[e]
    """

    spec: FieldSpec
    window_constants: tuple[FieldElement, ...]
    mul_window_size: int

    def __attrs_post_init__(self) -> None:
        if self.spec.p == 2:
            raise ValueError("GF(2) is not supported in this class currently")
        if not self.window_constants:
            raise ValueError("window_constants must be nonempty")
        if any(self.field.is_zero(c) for c in self.window_constants):
            raise ValueError("window_constants must be nonzero")
        if not 1 <= self.mul_window_size <= self.spec.register_bits:
            raise ValueError("invalid mul_window_size")

    @property
    def field(self):
        return shared_field(self.spec)

    @property
    def selected_factors(self) -> tuple[FieldElement, ...]:
        """windowing論文kes  window_constants[-1] = k^(2**i)"""
        base_pow_2i = self.window_constants[-1]
        return tuple(self.field.pow(base_pow_2i, x) for x in range(2 ** len(self.window_constants)))

    @property
    def signature(self) -> Signature:
        width = len(self.window_constants)
        return Signature(
            [
                Register("exp_window", BQUInt(width, 2**width)),
                Register("x", QUInt(self.spec.coefficient_bits), shape=(self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        exp_window, x = soqs["exp_window"], soqs["x"]
        n = self.spec.coefficient_bits

        # working register |y> = |0>
        y = np.array([bb.allocate(dtype=QUInt(n)) for _ in range(self.spec.r)], dtype=object)

        # forward: |e>|x>|0> -> |e>|x>|k_e*x>, k_e = selected_factors[e]
        exp_window, x, y = bb.add_t(
            WindowedProductAdd(
                spec=self.spec, factors=self.selected_factors, mul_window_size=self.mul_window_size
            ),
            exp_window=exp_window,
            x=x,
            y=y,
        )
        x, y = y, x  # relabel

        # cleanup: relabel後の |k_e*a>|a> -> |k_e*a>|0>, y += -k_e^-1 * x
        # windowing論文のkes_inv
        negative_inverse_factors = tuple(
            self.field.neg(self.field.inv(factor)) for factor in self.selected_factors
        )
        exp_window, x, y = bb.add_t(
            WindowedProductAdd(
                spec=self.spec,
                factors=negative_inverse_factors,
                mul_window_size=self.mul_window_size,
            ),
            exp_window=exp_window,
            x=x,
            y=y,
        )

        # free working register
        for soq in y:
            bb.free(soq)
        return {"exp_window": exp_window, "x": x}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        exp_window = int(vals["exp_window"])
        x = tuple(int(v) for v in np.asarray(vals["x"], dtype=object))
        return {
            "exp_window": exp_window,
            "x": np.asarray(self.field.mul(self.selected_factors[exp_window], x), dtype=object),
        }

    def adjoint(self):
        return WindowedConstMul(
            spec=self.spec,
            window_constants=tuple(self.field.inv(c) for c in self.window_constants),
            mul_window_size=self.mul_window_size,
        )
