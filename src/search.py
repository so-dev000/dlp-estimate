from dataclasses import asdict, dataclass
from typing import NotRequired, TypedDict

from .field import DLPInstance, FieldElement, FieldSpec, PolynomialCoefficients
from .logical_resources import estimate_resources
from .shor import ShorDLP, make_shor_config


@dataclass(frozen=True, kw_only=True)
class Params:
    """DLP入力。検証はFieldSpecとDLPInstanceに任せる。"""

    p: int
    r: int
    f: PolynomialCoefficients
    q: int
    g: FieldElement
    h: FieldElement
    exponent_bits: int | None = None


class EvalRow(TypedDict):
    p: int
    r: int
    f: PolynomialCoefficients
    q: int
    g: FieldElement
    h: FieldElement
    exponent_bits: int
    logical_qubits: int
    t: int
    toffoli: int
    cswap: int
    and_bloq: int
    clifford: int
    rotation: int
    measurement: int
    error: NotRequired[str]


def eval_point(params: Params) -> EvalRow:
    spec = FieldSpec(p=params.p, r=params.r, f=params.f)
    instance = DLPInstance(spec=spec, q=params.q, g=params.g, h=params.h)
    config = make_shor_config(instance, exponent_bits=params.exponent_bits)
    resource = estimate_resources(ShorDLP(instance=instance, config=config))
    return EvalRow(
        p=params.p,
        r=params.r,
        f=params.f,
        q=params.q,
        g=params.g,
        h=params.h,
        exponent_bits=config.exponent_bits,
        logical_qubits=resource.logical_qubits,
        t=int(resource.gates.t),
        toffoli=int(resource.gates.toffoli),
        cswap=int(resource.gates.cswap),
        and_bloq=int(resource.gates.and_bloq),
        clifford=int(resource.gates.clifford),
        rotation=int(resource.gates.rotation),
        measurement=int(resource.gates.measurement),
    )


def sweep(points: list[Params]) -> list[dict]:
    rows = []
    for params in points:
        try:
            rows.append(eval_point(params))
        except Exception as e:
            rows.append(asdict(params) | {"error": f"{type(e).__name__}: {e}"})
    return rows
