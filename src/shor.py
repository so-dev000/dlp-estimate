import attrs

from .field import DLPInstance, FieldElement, FiniteField


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


def exponentiation_constants(
    field: FiniteField,
    base: FieldElement,
    exponent_bits: int,
) -> tuple[FieldElement, ...]:
    constants = []
    constant = base
    for _ in range(exponent_bits):
        constants.append(constant)
        constant = field.mul(constant, constant)
    return tuple(constants)
