from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

GRAPHS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("scaling_qubits.png", "physical_qubits", "Physical qubits", "physical qubits", True),
    (
        "scaling_runtime_total.png",
        "duration_hr_total",
        "Total runtime (R runs)",
        "total runtime [hours]",
        False,
    ),
)
GRAPH_FILES = tuple(name for name, *_ in GRAPHS)


def _sci_notation(v: float) -> str:
    mantissa, _, exp = f"{v:.1e}".partition("e")
    return f"${mantissa}\\times 10^{{{int(exp)}}}$"


def _plot(
    data: list[dict[str, Any]],
    out_path: Path,
    y_key: str,
    title: str,
    ylabel: str,
    sci: bool,
) -> Path:
    fig, ax = plt.subplots()
    ax.scatter([row["field_bits"] for row in data], [row[y_key] for row in data])
    for i, row in enumerate(data):
        ax.annotate(
            row.get("label", ""),
            (row["field_bits"], row[y_key]),
            xytext=(4, 6 if i % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("field size [bits]")
    ax.set_ylabel(ylabel)
    if sci:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _sci_notation(v)))
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_graphs(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """全グラフを out_dir に生成し、保存先の一覧を返す。成功行がなければ空リスト。"""
    base_need = {"field_bits"}
    data = sorted(
        (row for row in rows if "error" not in row and base_need <= row.keys()),
        key=lambda row: row["field_bits"],
    )
    if not data:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, y_key, title, ylabel, sci in GRAPHS:
        graph_data = [row for row in data if y_key in row]
        if not graph_data:
            continue
        paths.append(_plot(graph_data, out_dir / name, y_key, title, ylabel, sci))
    return paths
