import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, CBit, QBit, QUInt, Register, Side, Signature, SoquetT
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import Hadamard

from src.circuits.arithmetic import get_controlled_const_mul
from src.circuits.semiclassical_qft import semiclassical_qft_step

from ..field import DLPInstance, FieldElement, FiniteField, shared_field


def _squared_constants(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> tuple[FieldElement, ...]:
    """
    baseの累乗定数をMSB順に返す
    例: 3ビットなら (base^4, base^2, base)
    """
    constants = [base] * exponent_bits
    for i in range(exponent_bits - 2, -1, -1):
        constants[i] = field.mul(constants[i + 1], constants[i + 1])
    return tuple(constants)


@attrs.frozen(kw_only=True)
class ShorDLP(Bloq):
    """
    Overall Algorithm: https://arxiv.org/abs/1905.09749
    Qubit Recycling: https://arxiv.org/abs/quant-ph/0001066
    """

    instance: DLPInstance

    @property
    def field(self):
        return shared_field(self.instance.spec)

    @property
    def exponent_bits(self) -> int:
        """2つの指数レジスタに共通のビット幅。Mosca構成 n = ceil(log2(2q))+1"""
        return (2 * self.instance.q - 1).bit_length() + 1

    @property
    def signature(self) -> Signature:
        m = self.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r
        return Signature(
            [
                Register("a", CBit(), shape=(m,), side=Side.RIGHT),
                Register("b", CBit(), shape=(m,), side=Side.RIGHT),
                Register("y", QUInt(n), shape=(r,), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        m = self.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r

        # working register |y>の初期化
        y = np.array(
            [bb.allocate(dtype=QUInt(n)) for _ in range(r)],
            dtype=object,
        )
        y[-1] = bb.add(XorK(dtype=QUInt(n), k=1), x=y[-1])

        outputs: dict[str, SoquetT] = {}

        for name, base in (("a", self.instance.h), ("b", self.instance.g)):
            constants = _squared_constants(self.field, base, m)
            history: list[SoquetT] = []

            for constant in constants:
                # リサイクルするcontrol qubit
                bit = bb.allocate(dtype=QBit())
                bit = bb.add(Hadamard(), q=bit)
                bit, y = bb.add_t(
                    get_controlled_const_mul(self.instance.spec, constant),
                    ctrl=bit,
                    x=y,
                )
                history = semiclassical_qft_step(bb, bit, history, inverse=True)
            outputs[name] = np.asarray(history[::-1], dtype=object)

        outputs["y"] = y
        return outputs

    def adjoint(self):
        raise NotImplementedError("This Bloq contains measurements.")
