import math


def validate_probability(value: float, name: str, *, endpoints: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    valid = 0 <= value <= 1 if endpoints else 0 < value < 1
    if not math.isfinite(value) or not valid:
        interval = "[0, 1]" if endpoints else "(0, 1)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return float(value)
