import attrs
import numpy as np
from qualtran import (
    Bloq,
    BloqBuilder,
    BQUInt,
    CBit,
    QBit,
    QUInt,
    Register,
    Side,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import Hadamard

from src.circuits.semiclassical_qft import semiclassical_qft_step
from src.circuits.windowed_const_mul import WindowedConstMul

from ..field import DLPInstance, FieldElement, FiniteField, shared_field


def _squared_constants(field: FiniteField, base: FieldElement, m: int) -> tuple[FieldElement, ...]:
    """
    (base^(2^{m-1}), ..., base^2, base) を MSB-first で返す。
    指数ビット列 a_0...a_{m-1} に対し base^a = Π constants[i]^a_i。
    """
    constants = [base] * m
    for i in range(m - 2, -1, -1):
        constants[i] = field.mul(constants[i + 1], constants[i + 1])
    return tuple(constants)


@attrs.frozen(kw_only=True)
class ShorDLP(Bloq):
    """
    Overall Algorithm: https://arxiv.org/abs/1905.09749
    Qubit Recycling: https://arxiv.org/abs/quant-ph/0001066
    Windowing: https://arxiv.org/abs/1905.07682

    |a>|b>|x> -> |a>|b>|x h^a g^b>。
    """

    instance: DLPInstance
    exp_window_size: int
    mul_window_size: int

    def __attrs_post_init__(self) -> None:
        if self.instance.spec.p == 2:
            raise ValueError("GF(2) is not supported in this class currently")

        if not 1 <= self.exp_window_size <= self.exponent_bits:
            raise ValueError("invalid exp_window_size")

        if not 1 <= self.mul_window_size <= self.instance.spec.register_bits:
            raise ValueError("invalid mul_window_size")

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
                Register("x", QUInt(n), shape=(r,), side=Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        m = self.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r

        # target |x> = |1>
        x = np.array(
            [bb.allocate(dtype=QUInt(n)) for _ in range(r)],
            dtype=object,
        )
        x[-1] = bb.add(XorK(dtype=QUInt(n), k=1), x=x[-1])

        outputs: dict[str, SoquetT] = {}

        # working register |0>
        work = np.array(
            [bb.allocate(dtype=QUInt(n)) for _ in range(r)],
            dtype=object,
        )

        # |x=1> -> |h^a> -> |h^a g^b>
        for name, base in (("a", self.instance.h), ("b", self.instance.g)):
            constants = _squared_constants(self.field, base, m)
            history: list[SoquetT] = []

            # exponent windowing
            for start in range(0, m, self.exp_window_size):
                window_constants = constants[start : start + self.exp_window_size]
                width = len(window_constants)

                # Recyclingするcontrol qubits
                bits = np.array(
                    [
                        bb.add(
                            Hadamard(),
                            q=bb.allocate(dtype=QBit()),
                        )
                        for _ in range(width)
                    ],
                    dtype=object,
                )

                # windowing論文のei
                exp_window = bb.join(
                    bits,
                    dtype=BQUInt(
                        width,
                        2**width,
                    ),
                )

                exp_window, x, work = bb.add_t(
                    WindowedConstMul(
                        spec=self.instance.spec,
                        window_constants=window_constants,
                        mul_window_size=self.mul_window_size,
                    ),
                    exp_window=exp_window,
                    x=x,
                    work=work,
                )

                bits = bb.split(exp_window)

                for bit in bits:
                    history = semiclassical_qft_step(
                        bb,
                        bit,
                        history,
                        inverse=True,
                    )

            outputs[name] = np.asarray(
                history[::-1],
                dtype=object,
            )

        for soq in work:
            bb.free(soq)

        outputs["x"] = x
        return outputs

    def adjoint(self):
        raise NotImplementedError("This Bloq contains measurements.")
