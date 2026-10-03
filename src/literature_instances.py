from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from src.field import Factorization
from src.instances import Params

_SCHEMA_VERSION = 1
_LITERATURE_DATA_PACKAGE = "src.literature_data"


def _decode_integer(value: object) -> int:
    """Canonical JSON上の整数表現をPythonのintへ変換する。"""
    if isinstance(value, str):
        return int(value)

    if isinstance(value, dict) and value.get("encoding") == "mersenne":
        exponent = value.get("exponent")
        if type(exponent) is not int or exponent < 1:
            raise ValueError("invalid Mersenne exponent")
        return (1 << exponent) - 1

    raise ValueError(f"unsupported integer encoding: {value!r}")


def _decode_coefficients(
    value: dict[str, Any],
    *,
    length: int,
) -> tuple[int, ...]:
    """
    Canonical JSON上の多項式・有限体要素を、
    高次数順のdense coefficient tupleへ変換する。
    """
    encoding = value.get("encoding")

    if encoding == "dense_coefficients_high_to_low":
        raw_values = value.get("values")
        if not isinstance(raw_values, list):
            raise ValueError("dense coefficient encoding requires a values list")

        coefficients = tuple(int(x) for x in raw_values)
        if len(coefficients) != length:
            raise ValueError(f"expected {length} coefficients, got {len(coefficients)}")

        return coefficients

    if encoding == "gf2_nonzero_exponents":
        raw_exponents = value.get("exponents")
        if not isinstance(raw_exponents, list):
            raise ValueError("gf2_nonzero_exponents encoding requires an exponents list")

        coefficients = [0] * length
        for exponent in raw_exponents:
            if type(exponent) is not int or not 0 <= exponent < length:
                raise ValueError(f"invalid GF(2) exponent {exponent}")

            # GF(2)上の加算なので重複した場合はXORする。
            coefficients[length - 1 - exponent] ^= 1

        return tuple(coefficients)

    if encoding == "gf2_lsb0_hex":
        encoded_length = value.get("length")
        if encoded_length != length:
            raise ValueError(f"expected encoded length {length}, got {encoded_length}")

        encoded_hex = value.get("hex")
        if not isinstance(encoded_hex, str):
            raise ValueError("gf2_lsb0_hex requires a hexadecimal string")

        try:
            packed = int(encoded_hex, 16)
        except ValueError as exc:
            raise ValueError("invalid hexadecimal GF(2) encoding") from exc

        if packed.bit_length() > length:
            raise ValueError("packed GF(2) element exceeds declared length")

        expected_sha256 = value.get("sha256_little_endian")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str):
                raise ValueError("invalid SHA-256 value")

            nbytes = (length + 7) // 8
            actual_sha256 = hashlib.sha256(packed.to_bytes(nbytes, "little")).hexdigest()

            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "packed GF(2) element checksum mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

        return tuple((packed >> exponent) & 1 for exponent in range(length - 1, -1, -1))

    raise ValueError(f"unsupported coefficient encoding: {encoding!r}")


def _decode_factorization(value: object) -> Factorization | None:
    """
    Canonical JSON上のqの完全因数分解を読み込む。

    None は完全因数分解が文献から得られていないことを表す。
    GKLWZ21 では q = 2^30750 - 1 の完全因数分解が既知ではないため None。
    KLEINJUNG14 では q = 2^1279 - 1 が素数であるため完全因数分解が既知。
    """
    if value is None:
        return None

    if not isinstance(value, list):
        raise ValueError("q_factors must be a list or null")

    factors: list[tuple[int, int]] = []

    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"invalid factorization entry: {entry!r}")

        prime = int(entry[0])
        exponent = int(entry[1])

        if prime <= 1:
            raise ValueError(f"factor must be greater than 1, got {prime}")

        if exponent < 1:
            raise ValueError(f"factor exponent must be positive, got {exponent}")

        factors.append((prime, exponent))

    return tuple(factors)


def _load_record(label: str) -> dict[str, Any]:
    """指定した literature instance の canonical JSON を読み込む。"""
    if not label:
        raise ValueError("label must not be empty")

    path = files(_LITERATURE_DATA_PACKAGE).joinpath(f"{label}.json")

    if not path.is_file():
        raise FileNotFoundError(f"literature instance not found: {label!r}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"literature instance must contain a JSON object: {label!r}")

    schema_version = data.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported literature-instance schema for {label!r}: {schema_version!r}"
        )

    stored_label = data.get("label")
    if stored_label != label:
        raise ValueError(f"label mismatch: expected {label!r}, got {stored_label!r}")

    return data


def load_literature_params(label: str) -> Params:
    """
    Canonical literature-instance JSON を Params へ変換する。

    元論文の tower representation の再構築、flatten、
    subgroup projection などはここでは行わない。
    それらの導出と検証は script/generate_literature_instances.py が担当する。
    """
    data = _load_record(label)

    field = data.get("field")
    dlp = data.get("dlp")

    if not isinstance(field, dict):
        raise ValueError(f"invalid field section for literature instance {label!r}")

    if not isinstance(dlp, dict):
        raise ValueError(f"invalid dlp section for literature instance {label!r}")

    p = int(field["p"])
    r = int(field["r"])

    if p < 2:
        raise ValueError(f"p must be at least 2, got {p}")

    if r < 1:
        raise ValueError(f"r must be positive, got {r}")

    modulus = field.get("modulus")
    g_encoded = dlp.get("g")
    h_encoded = dlp.get("h")

    if not isinstance(modulus, dict):
        raise ValueError("field.modulus must be an object")

    if not isinstance(g_encoded, dict):
        raise ValueError("dlp.g must be an object")

    if not isinstance(h_encoded, dict):
        raise ValueError("dlp.h must be an object")

    f = _decode_coefficients(modulus, length=r + 1)
    g = _decode_coefficients(g_encoded, length=r)
    h = _decode_coefficients(h_encoded, length=r)

    q = _decode_integer(dlp["q"])
    q_factors = _decode_factorization(dlp.get("q_factors"))

    return Params(
        label=label,
        p=p,
        r=r,
        f=f,
        q=q,
        q_factors=q_factors,
        g=g,
        h=h,
    )


DGP21 = load_literature_params("dgp21")
GKLWZ21 = load_literature_params("gklwz21")
KLEINJUNG14 = load_literature_params("kleinjung14")

LITERATURE_INSTANCES: list[Params] = [
    DGP21,
    # GKLWZ21,
    KLEINJUNG14,
]

FP6_LITERATURE_INSTANCES: list[Params] = [
    DGP21,
]

GF2_LITERATURE_INSTANCES: list[Params] = [
    # GKLWZ21,
    KLEINJUNG14,
]
