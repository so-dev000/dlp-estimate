from dataclasses import dataclass

from .field import Factorization, FieldElement, PolynomialCoefficients


@dataclass(frozen=True, kw_only=True)
class Params:
    """検証済みの DLP 入力。eval_point では重い体・位数・部分群検証を行わない。"""

    label: str = ""
    p: int
    r: int
    f: PolynomialCoefficients
    q: int
    q_factors: Factorization | None = None
    g: FieldElement
    h: FieldElement
