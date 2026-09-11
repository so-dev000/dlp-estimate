from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.mod_arithmetic import CtrlScaleModAdd
from qualtran.simulation.classical_sim import ClassicalValT

from .field import FieldElement, FieldMatrix, FieldSpec, FiniteField


@attrs.frozen(kw_only=True)
class ControlledLinearMapAdd(Bloq):
    """(ctrl, x, y) -> (ctrl, x, y + ctrl*M*x)"""

    spec: FieldSpec
    matrix: FieldMatrix

    def __attrs_post_init__(self) -> None:
        # 体を構築して素数性・既約性を検証する。結果は field にキャッシュされる。
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
                # attrs が生成する shape 引数を型検査器が認識しないため位置引数で渡す。
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
        for i, row in enumerate(self.matrix):
            for j, coefficient in enumerate(row):
                if coefficient == 0:
                    continue
                ctrl, x[j], y[i] = bb.add_t(
                    CtrlScaleModAdd(bitsize=n, mod=p, k=coefficient), ctrl=ctrl, x=x[j], y=y[i]
                )
        return {"ctrl": ctrl, "x": x, "y": y}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        return {}


@attrs.frozen(kw_only=True)
class ControlledConstMul(Bloq):
    """(ctrl, x) -> (ctrl, c^ctrl * x)"""

    spec: FieldSpec
    c: FieldElement

    def __attrs_post_init__(self) -> None:
        # to_galoisがcの長さ・係数範囲を検証し、非零性をここで確認する。
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
                Register("y", QUInt(n), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        return {}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        return {}

    def adjoint(self) -> ControlledConstMul:
        raise NotImplementedError
