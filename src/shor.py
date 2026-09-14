from functools import cached_property

import attrs
import numpy as np
from qualtran import Bloq, BloqBuilder, QUInt, Register, Signature, SoquetT
from qualtran.simulation.classical_sim import ClassicalValT

from .arithmetic import ControlledConstMul
from .field import DLPInstance, FieldElement, FieldSpec, FiniteField


@attrs.frozen(kw_only=True)
class ShorConfig:
    exponent_bits: int  # 2つの指数レジスタに共通のビット幅

    def __attrs_post_init__(self) -> None:
        if type(self.exponent_bits) is not int or self.exponent_bits < 1:
            raise ValueError("exponent_bits must be a positive integer")


def make_shor_config(
    instance: DLPInstance,
    *,
    exponent_bits: int | None = None,
) -> ShorConfig:
    minimum_bits = (instance.q * instance.q).bit_length()
    if exponent_bits is None:
        exponent_bits = minimum_bits

    config = ShorConfig(exponent_bits=exponent_bits)
    if config.exponent_bits < minimum_bits:
        raise ValueError(f"exponent_bits must be at least {minimum_bits} to satisfy 2**m > q**2")
    return config


def _validate_exponentiation_inputs(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> None:
    if type(exponent_bits) is not int or exponent_bits < 1:
        raise ValueError("exponent_bits must be a positive integer")
    if field.to_galois(base) == 0:
        raise ValueError("base must be nonzero")


def _squared_constants(
    field: FiniteField, base: FieldElement, exponent_bits: int
) -> tuple[FieldElement, ...]:
    constants = []
    constant = base
    for _ in range(exponent_bits):
        constants.append(constant)
        constant = field.mul(constant, constant)
    return tuple(constants)


@attrs.frozen(kw_only=True)
class FieldExponentiation(Bloq):
    """
    (a, x) -> (a, base**a * x)
    a = Σ a_i * 2**i, a_i ∈ {0, 1}, i = 0, ..., exponent_bits - 1を使用して
    base ** a = Π (base ** (2**i)) ** a_i を計算する
    """

    spec: FieldSpec
    base: FieldElement
    exponent_bits: int

    def __attrs_post_init__(self) -> None:
        _ = self.field
        _validate_exponentiation_inputs(self.field, self.base, self.exponent_bits)

    @cached_property
    def field(self) -> FiniteField:
        """素数性・既約性を検証済みの有限体。"""
        return FiniteField(self.spec)

    @property
    def signature(self) -> Signature:
        n = self.spec.coefficient_bits
        return Signature(
            [
                Register("exponent", QUInt(self.exponent_bits)),
                Register("x", QUInt(n), (self.spec.r,)),
            ]
        )

    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]:
        bits = bb.split(soqs["exponent"])
        x = soqs["x"]
        assert isinstance(bits, np.ndarray)

        constants = _squared_constants(self.field, self.base, self.exponent_bits)
        for i, constant in enumerate(constants):
            # 指数はbig-endianなので、第i定数 base**(2**i) は bits[-1-i] で指定する
            bits[-1 - i], x = bb.add_t(
                ControlledConstMul(spec=self.spec, c=constant),
                ctrl=bits[-1 - i],
                x=x,
            )

        return {
            "exponent": bb.join(bits, dtype=QUInt(self.exponent_bits)),
            "x": x,
        }

    def on_classical_vals(self, **vals: ClassicalValT) -> dict[str, ClassicalValT]:
        r, p = self.spec.r, self.spec.p
        exponent = int(vals["exponent"])
        if not 0 <= exponent < 2**self.exponent_bits:
            raise ValueError(f"exponent must be in [0, {2**self.exponent_bits})")

        x = np.asarray(vals["x"])
        if x.shape != (r,):
            raise ValueError(f"x must have shape ({r},), got {x.shape}")
        coefficients = tuple(int(v) for v in x)
        if any(not 0 <= coefficient < p for coefficient in coefficients):
            raise ValueError(f"x coefficients must be in [0, {p})")

        power = self.field.pow(self.base, exponent)
        result = self.field.mul(power, coefficients)
        return {"exponent": exponent, "x": np.array(result, dtype=object)}

    def adjoint(self) -> FieldExponentiation:
        return FieldExponentiation(
            spec=self.spec,
            base=self.field.inv(self.base),
            exponent_bits=self.exponent_bits,
        )


@attrs.frozen(kw_only=True)
class ShorDLP(Bloq):
    instance: DLPInstance
    config: ShorConfig

    @cached_property
    def field(self) -> FiniteField: ...
    @property
    def signature(self) -> Signature: ...
    def build_composite_bloq(self, bb: BloqBuilder, **soqs: SoquetT) -> dict[str, SoquetT]: ...
