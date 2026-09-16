from functools import cached_property

import attrs
import numpy as np
from qualtran import QGF, Bloq, BloqBuilder, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.gf_arithmetic import GF2MulK
from qualtran.bloqs.mod_arithmetic import CtrlScaleModAdd
from qualtran.simulation.classical_sim import ClassicalValT

from .field import FieldElement, FieldMatrix, FieldSpec, FiniteField


@attrs.frozen(kw_only=True)
class ControlledLinearMapAdd(Bloq):
    """
    |ctrl>|x>|y> -> |ctrl>|x>|y + ctrl*M*x>
    Figure 14 of https://arxiv.org/abs/quant-ph/0301163v1
    """

    spec: FieldSpec
    matrix: FieldMatrix

    def __attrs_post_init__(self) -> None:
        _ = self.field
        r, p = self.spec.r, self.spec.p
        if p == 2:
            raise ValueError("GF(2) is not supported in this class")
        if len(self.matrix) != r:
            raise ValueError(f"matrix must have {r} rows")
        for row in self.matrix:
            if len(row) != r:
                raise ValueError(f"matrix rows must have {r} columns")
            if any(type(entry) is not int or not 0 <= entry < p for entry in row):
                raise ValueError(f"matrix entries must be integers in [0, {p})")

    @cached_property
    def field(self) -> FiniteField:
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

        # ===== O(r^2)の呼び出しで非効率. Windowingという方法で減らせる? =====
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

        # チェック
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

        # 計算
        if ctrl:
            y = (np.array(self.matrix, dtype=object) @ x + y) % p

        return {"ctrl": ctrl, "x": x, "y": y}


@attrs.frozen(kw_only=True)
class ControlledConstMul(Bloq):
    """
    |ctrl>|x> -> |ctrl>|c^ctrl * x>
    Figure 11  of https://arxiv.org/abs/quant-ph/0301163v1
    """

    spec: FieldSpec
    c: FieldElement

    def __attrs_post_init__(self) -> None:
        if self.spec.p == 2:
            raise ValueError("GF(2) is not supported in this class")
        if self.field.to_galois(self.c) == 0:
            raise ValueError("c must be nonzero")

    @cached_property
    def field(self) -> FiniteField:
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

        # チェック
        if ctrl not in (0, 1):
            raise ValueError(f"ctrl must be 0 or 1, got {ctrl}")

        x = np.asarray(vals["x"])
        if x.shape != (r,):
            raise ValueError(f"x must have shape ({r},), got {x.shape}")

        coefficients = tuple(int(v) for v in x)
        if any(not 0 <= coefficient < p for coefficient in coefficients):
            raise ValueError(f"x coefficients must be in [0, {p})")

        # 計算
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


@attrs.frozen(kw_only=True)
class ControlledGF2ConstMul(Bloq):
    """
    |ctrl>|x> -> |ctrl>|c^ctrl * x>
    https://qualtran.readthedocs.io/en/latest/bloqs/gf_arithmetic/gf2_multiplication.html#gf2mulk
    https://arxiv.org/abs/1910.02849v2 Algorithm 1
    """

    spec: FieldSpec
    c: FieldElement

    def __attrs_post_init__(self) -> None:
        if self.spec.p != 2:
            raise ValueError("GF(2) is required in this class")
        if self.field.to_galois(self.c) == 0:
            raise ValueError("c must be nonzero")

    @cached_property
    def field(self) -> FiniteField:
        return FiniteField(self.spec)

    @property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("ctrl", QBit()),
                Register("x", QUInt(1), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        ctrl = soqs["ctrl"]
        x = soqs["x"]

        if self.c == self.field.one:
            return {"ctrl": ctrl, "x": x}

        constant = self.field.to_galois(self.c)
        polynomial = type(constant).irreducible_poly
        qgf = QGF(2, self.spec.r, polynomial)
        multiplication = GF2MulK(dtype=qgf, const=int(constant)).controlled()

        assert isinstance(x, np.ndarray)

        # QGFのビット列は高次数順なので、外部の係数ビットを反転してjoinし、出力をsplitして再び反転
        bits = np.array([bb.split(coefficient)[0] for coefficient in x], dtype=object)
        g = bb.join(bits[::-1], dtype=qgf)
        ctrl, g = bb.add_t(multiplication, ctrl=ctrl, g=g)

        bits = bb.split(g)[::-1]
        x = np.array([bb.join([bit], dtype=QUInt(1)) for bit in bits], dtype=object)
        return {"ctrl": ctrl, "x": x}

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        r, p = self.spec.r, self.spec.p
        ctrl = int(vals["ctrl"])

        # チェック
        if ctrl not in (0, 1):
            raise ValueError(f"ctrl must be 0 or 1, got {ctrl}")

        x = np.asarray(vals["x"])
        if x.shape != (r,):
            raise ValueError(f"x must have shape ({r},), got {x.shape}")

        coefficients = tuple(int(v) for v in x)
        if any(coefficient not in (0, 1) for coefficient in coefficients):
            raise ValueError(f"x coefficients must be in [0, {p})")

        if ctrl:
            coefficients = self.field.mul(self.c, coefficients)

        return {
            "ctrl": ctrl,
            "x": np.array(coefficients, dtype=object),
        }

    def adjoint(self) -> ControlledGF2ConstMul:
        return ControlledGF2ConstMul(
            spec=self.spec,
            c=self.field.inv(self.c),
        )


def get_controlled_const_mul(spec: FieldSpec, c: FieldElement) -> Bloq:
    """標数2ではGF(2^r)専用、それ以外では奇標数用の制御付き定数乗算を返す。"""
    if spec.p == 2:
        return ControlledGF2ConstMul(spec=spec, c=c)
    return ControlledConstMul(spec=spec, c=c)
