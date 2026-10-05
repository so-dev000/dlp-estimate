from functools import cache, cached_property

import attrs
import numpy as np
from qualtran import QGF, Bloq, BloqBuilder, QBit, QUInt, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, TwoBitSwap
from qualtran.bloqs.gf_arithmetic import GF2MulK
from qualtran.bloqs.gf_arithmetic.gf2_multiplication import SynthesizeLRCircuit
from qualtran.simulation.classical_sim import ClassicalValT

from ..field import FieldElement, FieldSpec, FiniteField, shared_field


@cache
def qgf_for_spec(spec: FieldSpec) -> QGF:
    return QGF(2, spec.r, tuple(spec.r - i for i, coeff in enumerate(spec.f) if coeff))


class GF2MulKWithExplicitSwaps(GF2MulK):
    """Qualtran 0.7.0 で制御 OFF 時にも残る暗黙の置換を、明示的な SWAP に置き換える。
    元のGF2MulKへ戻すとctrl=0でも値が変わる
    Issue作成済み
    """

    def build_composite_bloq(self, bb: BloqBuilder, g: SoquetT) -> dict[str, SoquetT]:
        bits = bb.split(g)[::-1]
        lower, upper, permutation = SynthesizeLRCircuit(self.reduction_matrix_q).lup
        n = len(bits)
        for i in range(n):
            for j in range(i + 1, n):
                if upper[i, j]:
                    bits[j], bits[i] = bb.add_t(CNOT(), ctrl=bits[j], target=bits[i])
        for i in reversed(range(n)):
            for j in reversed(range(i)):
                if lower[i, j]:
                    bits[j], bits[i] = bb.add_t(CNOT(), ctrl=bits[j], target=bits[i])
        columns = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[i, columns[j]]:
                    bits[i], bits[j] = bb.add_t(TwoBitSwap(), x=bits[i], y=bits[j])  # 明示的にSWAP
                    columns[i], columns[j] = columns[j], columns[i]
        return {"g": bb.join(bits[::-1], dtype=self.dtype)}

    def build_call_graph(self, ssa):
        # build_composite_bloq と同じ LUP 由来の数を、分解なしで返す。
        # 合成回路は CNOT と TwoBitSwap だけからなる。
        lower, upper, permutation = SynthesizeLRCircuit(self.reduction_matrix_q).lup
        n = int(self.n)
        n_cnot = 0
        for i in range(n):
            for j in range(i + 1, n):
                if upper[i, j]:
                    n_cnot += 1
        for i in range(n):
            for j in range(i):
                if lower[i, j]:
                    n_cnot += 1
        columns = list(range(n))
        n_swap = 0
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[i, columns[j]]:
                    n_swap += 1
                    columns[i], columns[j] = columns[j], columns[i]
        counts: dict[Bloq, int] = {}
        if n_cnot:
            counts[CNOT()] = n_cnot
        if n_swap:
            counts[TwoBitSwap()] = n_swap
        return counts


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
        if self.field.is_zero(self.c):
            raise ValueError("c must be nonzero")

    @cached_property
    def field(self) -> FiniteField:
        return shared_field(self.spec)

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

        constant = sum(bit << (self.spec.r - 1 - i) for i, bit in enumerate(self.c))
        qgf = qgf_for_spec(self.spec)
        multiplication = GF2MulKWithExplicitSwaps(dtype=qgf, const=constant).controlled()

        assert isinstance(x, np.ndarray)

        # 係数列とQGFのビット列はともに高次数順。
        bits = np.array([bb.split(coefficient)[0] for coefficient in x], dtype=object)
        g = bb.join(bits, dtype=qgf)
        ctrl, g = bb.add_t(multiplication, ctrl=ctrl, g=g)

        bits = bb.split(g)
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

    def adjoint(self):
        return ControlledGF2ConstMul(
            spec=self.spec,
            c=self.field.inv(self.c),
        )
