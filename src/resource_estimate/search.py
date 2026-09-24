import contextlib
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict

from ..field import DLPInstance, FieldElement, FieldSpec, PolynomialCoefficients
from ..shor import ShorDLP, make_shor_config
from .qualtran_resources import (
    QualtranPhysicalConfig,
    estimate_logical_resources,
    estimate_physical_resources,
)


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
    # DLP instance
    p: int
    r: int
    f: PolynomialCoefficients
    q: int
    g: FieldElement
    h: FieldElement
    field_bits: int
    # Shor configuration
    exponent_bits: int
    # Logical resources
    logical_qubits: int
    # Raw gate counts
    t: int
    toffoli: int
    cswap: int
    and_bloq: int
    clifford: int
    rotation: int
    measurement: int
    # Normalized gate counts
    ccz: int
    t_equiv_default: int
    # Physical resources
    physical_qubits: int
    n_cycles: int
    duration_hr: float
    failure_prob: float
    code_distance: int
    # Error message
    error: NotRequired[str]


def eval_point(
    params: Params,
    physical_config: QualtranPhysicalConfig,
) -> EvalRow:
    spec = FieldSpec(p=params.p, r=params.r, f=params.f)
    instance = DLPInstance(spec=spec, q=params.q, g=params.g, h=params.h)
    config = make_shor_config(instance, exponent_bits=params.exponent_bits)
    resource = estimate_logical_resources(ShorDLP(instance=instance, config=config))
    physical = estimate_physical_resources(resource, physical_config)
    return EvalRow(
        p=params.p,
        r=params.r,
        f=params.f,
        q=params.q,
        g=params.g,
        h=params.h,
        field_bits=spec.field_bits,
        exponent_bits=config.exponent_bits,
        logical_qubits=resource.logical_qubits,
        t=int(resource.gates.t),
        toffoli=int(resource.gates.toffoli),
        cswap=int(resource.gates.cswap),
        and_bloq=int(resource.gates.and_bloq),
        clifford=int(resource.gates.clifford),
        rotation=int(resource.gates.rotation),
        ccz=resource.ccz,
        t_equiv_default=resource.t_equiv_default,
        measurement=int(resource.gates.measurement),
        physical_qubits=physical.physical_qubits,
        n_cycles=physical.n_cycles,
        duration_hr=physical.duration_hr,
        failure_prob=physical.failure_prob,
        code_distance=physical.code_distance,
    )


def sweep(
    points: list[Params],
    physical_config: QualtranPhysicalConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for params in points:
        try:
            rows.append(dict(eval_point(params, physical_config)))
            print(f"Evaluated: {params}")
        except Exception as e:
            row: dict[str, Any] = asdict(params) | {"error": f"{type(e).__name__}: {e}"}
            if "field_bits" not in row:
                with contextlib.suppress(Exception):
                    row["field_bits"] = FieldSpec(p=params.p, r=params.r, f=params.f).field_bits
            rows.append(row)
    return rows
