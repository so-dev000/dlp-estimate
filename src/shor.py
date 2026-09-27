import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QBit, QUInt, Register, Side, Signature, SoquetT
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.qft import QFTTextBook

from .field import DLPInstance
from .oracle import DLPOracle


@attrs.frozen(kw_only=True)
class ShorDLP(Bloq):
    """
    Nielsen-Chuang5.4.2節のShor-DLP実装

    |a=0>|b=0>|y=0>
          -> H^{⊗m}をa, bに適用
          -> (1/2^m) Σ_{a,b} |a>|b>|y=0>
          -> DLPOracleを適用
          -> (1/2^m) Σ_{a,b} |a>|b>|y=encode(h^a g^b)>
          -> QFT^{-1}をa, bに適用
          -> |a>|b>|y>
    """

    instance: DLPInstance

    @property
    def exponent_bits(self) -> int:
        """2つの指数レジスタに共通のビット幅。2**m > q を満たす最小値。"""
        return self.instance.q.bit_length()

    @property
    def signature(self) -> Signature:
        m = self.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r
        return Signature(
            [
                Register("a", QUInt(m), side=Side.RIGHT),
                Register("b", QUInt(m), side=Side.RIGHT),
                Register("y", QBit(), (r * n,), Side.RIGHT),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        m = self.exponent_bits
        n = self.instance.spec.coefficient_bits
        r = self.instance.spec.r

        # 初期状態 |0>|0>|0>
        a = bb.allocate(dtype=QUInt(m))
        b = bb.allocate(dtype=QUInt(m))
        y = np.array(
            [bb.allocate(dtype=QBit()) for _ in range(r * n)],
            dtype=object,
        )

        # 一様重ね合わせ |0>|0>|0> -> (1/2^m) Σ_{a,b} |a>|b>|0>
        a_bits = bb.split(a)
        b_bits = bb.split(b)
        for i in range(m):
            a_bits[i] = bb.add(Hadamard(), q=a_bits[i])
            b_bits[i] = bb.add(Hadamard(), q=b_bits[i])
        a = bb.join(a_bits, dtype=QUInt(m))
        b = bb.join(b_bits, dtype=QUInt(m))

        # DLPOracleを適用 |a>|b>|y=0> -> |a>|b>|encode(h^a g^b)>
        a, b, y = bb.add_t(DLPOracle(instance=self.instance, exponent_bits=m), a=a, b=b, y=y)

        # 逆QFTを適用
        (a,) = bb.add_t(QFTTextBook(bitsize=m, with_reverse=True).adjoint(), q=a)
        (b,) = bb.add_t(QFTTextBook(bitsize=m, with_reverse=True).adjoint(), q=b)

        return {"a": a, "b": b, "y": y}
