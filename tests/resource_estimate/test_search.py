import pytest

from src.resource_estimate.qualtran_resources import QualtranPhysicalConfig
from src.resource_estimate.search import EvalRow, Params, eval_point, sweep


@pytest.fixture
def minimal_params() -> Params:
    """GF(5) の部分群 {1, 4} で 4^d = 1 を解く最小入力。"""
    return Params(p=5, r=1, f=(0, 1), q=2, g=(4,), h=(1,))


@pytest.fixture
def physical_config() -> QualtranPhysicalConfig:
    return QualtranPhysicalConfig(
        physical_error_rate=1e-3,
        cycle_time_us=0.01,
        data_d=7,
        data_block="fast",
        distillation_d_x=7,
        distillation_d_z=7,
        distillation_d_m=7,
        qec_scheme="gidney_fowler",
    )


def test_eval_point_returns_all_columns(
    minimal_params: Params, physical_config: QualtranPhysicalConfig
) -> None:
    row = eval_point(minimal_params, physical_config)

    assert set(EvalRow.__annotations__) - {"error"} <= set(row)
    assert row["p"] == 5
    assert row["r"] == 1
    assert row["f"] == (0, 1)
    assert row["q"] == 2
    assert row["logical_qubits"] > 0
    assert row["ccz"] == row["toffoli"] + row["cswap"] + row["and_bloq"]
    assert "error" not in row


def test_eval_point_defaults_to_minimum_exponent_bits(
    minimal_params: Params, physical_config: QualtranPhysicalConfig
) -> None:
    # 2**m > q**2 を満たす最小ビット幅は (2*2).bit_length() = 3。
    assert eval_point(minimal_params, physical_config)["exponent_bits"] == 3

    explicit = Params(
        p=minimal_params.p,
        r=minimal_params.r,
        f=minimal_params.f,
        q=minimal_params.q,
        g=minimal_params.g,
        h=minimal_params.h,
        exponent_bits=5,
    )
    assert eval_point(explicit, physical_config)["exponent_bits"] == 5


def test_eval_point_rejects_invalid_dlp_input(
    minimal_params: Params, physical_config: QualtranPhysicalConfig
) -> None:
    invalid = Params(
        p=minimal_params.p,
        r=minimal_params.r,
        f=minimal_params.f,
        q=minimal_params.q,
        g=(2,),  # 位数 4 であり q=2 と不一致。
        h=minimal_params.h,
    )
    with pytest.raises(ValueError, match="order of g"):
        eval_point(invalid, physical_config)


def test_sweep_continues_on_error_and_reports_type_name(
    minimal_params: Params, physical_config: QualtranPhysicalConfig
) -> None:
    invalid = Params(
        p=minimal_params.p,
        r=minimal_params.r,
        f=minimal_params.f,
        q=minimal_params.q,
        g=(2,),
        h=minimal_params.h,
    )
    rows = sweep([minimal_params, invalid], physical_config)

    assert len(rows) == 2
    assert "error" not in rows[0]
    assert rows[1]["error"].startswith("ValueError: ")
    # 失敗行も入力パラメータを保持する。
    assert rows[1]["g"] == (2,)
    assert rows[1]["exponent_bits"] is None


def test_sweep_uses_given_physical_config(minimal_params: Params) -> None:
    config = QualtranPhysicalConfig(
        physical_error_rate=1e-3,
        cycle_time_us=0.01,
        data_d=11,
        data_block="compact",
        distillation_d_x=11,
        distillation_d_z=11,
        distillation_d_m=11,
        qec_scheme="gidney_fowler",
    )
    rows = sweep([minimal_params], config)

    assert rows[0]["code_distance"] == 11
