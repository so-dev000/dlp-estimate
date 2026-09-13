from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.mod_arithmetic import CtrlScaleModAdd
from qualtran.simulation.classical_sim import ClassicalValT

from .field import FieldElement, FieldMatrix, FieldSpec, FiniteField


@attrs.frozen(kw_only=True)
class ControlledLinearMapAdd(Bloq):
    """
    (ctrl, x, y) -> (ctrl, x, y + ctrl*M*x)
    Figure 14 of https://arxiv.org/abs/quant-ph/0301163v1
    """

    spec: FieldSpec
    matrix: FieldMatrix

    def __attrs_post_init__(self) -> None:
        _ = self.field
        r, p = self.spec.r, self.spec.p
        if len(self.matrix) != r:
            raise ValueError(f"matrix must have {r} rows")
        for row in self.matrix:
            if len(row) != r:
                raise ValueError(f"matrix rows must have {r} columns")
            if any(type(entry) is not int or not 0 <= entry < p for entry in row):
                raise ValueError(f"matrix entries must be integers in [0, {p})")

    @cached_property
    def field(self) -> FiniteField:
        """素数性・既約性を検証済みの有限体。"""
        return FiniteField(self.spec)

    @property
    def signature(self) -> Signature:
        n = self.spec.coefficient_bits
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("x", QUInt(n), (self.spec.r,)),
                Register("y", QUInt(n), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        ctrl = soqs["ctrl"]
        x = soqs["x"]
        y = soqs["y"]
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        n = self.spec.coefficient_bits
        p = self.spec.p

        for j in range(self.spec.r):
            for i in range(self.spec.r):
                coefficient = self.matrix[i][j]
                if coefficient == 0:
                    continue
                ctrl, x[j], y[i] = bb.add_t(
                    # y[i] += ctrl * coefficient * x[j] mod p
                    CtrlScaleModAdd(bitsize=n, mod=p, k=coefficient),
                    ctrl=ctrl,
                    x=x[j],
                    y=y[i],
                )

        return {"ctrl": ctrl, "x": x, "y": y}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        r, p = self.spec.r, self.spec.p
        ctrl = int(vals["ctrl"])
        if ctrl not in (0, 1):
            raise ValueError(f"ctrl must be 0 or 1, got {ctrl}")

        def to_coefficients(name: str) -> np.ndarray:
            a = np.asarray(vals[name])
            if a.shape != (r,):
                raise ValueError(f"{name} must have shape ({r},), got {a.shape}")
            coefficients = [int(v) for v in a]
            if any(not 0 <= c < p for c in coefficients):
                raise ValueError(f"{name} coefficients must be in [0, {p})")
            return np.array(coefficients, dtype=object)

        x = to_coefficients("x")
        y = to_coefficients("y")

        if ctrl:
            y = (np.array(self.matrix, dtype=object) @ x + y) % p

        return {"ctrl": ctrl, "x": x, "y": y}


@attrs.frozen(kw_only=True)
class ControlledConstMul(Bloq):
    """
    (ctrl, x) -> (ctrl, c^ctrl * x)
    Figure 11  of https://arxiv.org/abs/quant-ph/0301163v1
    """

    spec: FieldSpec
    c: FieldElement

    def __attrs_post_init__(self) -> None:
        if self.field.to_galois(self.c) == 0:
            raise ValueError("c must be nonzero")

    @cached_property
    def field(self) -> FiniteField:
        """素数性・既約性を検証済みの有限体。"""
        return FiniteField(self.spec)

    @property
    def signature(self) -> Signature:
        n = self.spec.coefficient_bits
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("x", QUInt(n), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        ctrl = soqs["ctrl"]
        x = soqs["x"]

        if self.c == self.field.one:
            return {"ctrl": ctrl, "x": x}

        n = self.spec.coefficient_bits
        p = self.spec.p
        r = self.spec.r

        # working register
        y = np.array(
            [bb.allocate(dtype=QUInt(n)) for _ in range(r)],
            dtype=object,
        )

        matrix = self.field.const_mul_matrix(self.c)
        ctrl, x, y = bb.add_t(
            ControlledLinearMapAdd(
                spec=self.spec,
                matrix=matrix,
            ),
            ctrl=ctrl,
            x=x,
            y=y,
        )

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        # Controlled-SWAP
        for i in range(r):
            ctrl, x[i], y[i] = bb.add_t(
                CSwap(bitsize=n),
                ctrl=ctrl,
                x=x[i],
                y=y[i],
            )

        # Uncompute y
        inverse_matrix = self.field.const_mul_matrix(self.field.inv(self.c))
        negative_inverse_matrix = tuple(
            tuple((-entry) % p for entry in row) for row in inverse_matrix
        )

        ctrl, x, y = bb.add_t(
            ControlledLinearMapAdd(
                spec=self.spec,
                matrix=negative_inverse_matrix,
            ),
            ctrl=ctrl,
            x=x,
            y=y,
        )

        assert isinstance(y, np.ndarray)

        # free working register
        for soq in y:
            bb.free(soq)

        return {"ctrl": ctrl, "x": x}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        r, p = self.spec.r, self.spec.p
        ctrl = int(vals["ctrl"])
        if ctrl not in (0, 1):
            raise ValueError(f"ctrl must be 0 or 1, got {ctrl}")

        x = np.asarray(vals["x"])
        if x.shape != (r,):
            raise ValueError(f"x must have shape ({r},), got {x.shape}")

        coefficients = tuple(int(v) for v in x)
        if any(not 0 <= coefficient < p for coefficient in coefficients):
            raise ValueError(f"x coefficients must be in [0, {p})")

        if ctrl:
            coefficients = self.field.mul(self.c, coefficients)

        return {
            "ctrl": ctrl,
            "x": np.array(coefficients, dtype=object),
        }

    def adjoint(self) -> ControlledConstMul:
        return ControlledConstMul(
            spec=self.spec,
            c=self.field.inv(self.c),
        )
