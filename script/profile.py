from __future__ import annotations

import csv
import runpy
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import qualtran.resource_counting._costing as costing
from qualtran import Bloq
from qualtran.resource_counting import QECGatesCost, QubitCount

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "script" / "estimate_qualtran.py"
OUT_DIR = ROOT / "results" / "qualtran"

TIMEOUT_SECONDS = 60.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


T = TypeVar("T")


class ProfileTimeout(BaseException):
    """Global profiling timeout.

    BaseException を継承することで、推定コード側の
    `except Exception` に捕捉されるのを避ける。
    """


@dataclass
class ProfileStat:
    calls: int = 0
    inclusive_s: float = 0.0
    self_s: float = 0.0
    max_s: float = 0.0


@dataclass
class ProfileFrame:
    child_s: float = 0.0


class CostProfiler:
    def __init__(self) -> None:
        self.stats: dict[str, ProfileStat] = {}
        self.stack: list[ProfileFrame] = []

    def measure(
        self,
        bloq: Bloq,
        fn: Callable[[], T],
    ) -> T:
        name = type(bloq).__name__

        frame = ProfileFrame()
        self.stack.append(frame)

        start = time.perf_counter()

        try:
            return fn()
        finally:
            elapsed = time.perf_counter() - start

            finished = self.stack.pop()

            self_time = max(
                0.0,
                elapsed - finished.child_s,
            )

            stat = self.stats.setdefault(
                name,
                ProfileStat(),
            )

            stat.calls += 1
            stat.inclusive_s += elapsed
            stat.self_s += self_time
            stat.max_s = max(
                stat.max_s,
                elapsed,
            )

            if self.stack:
                self.stack[-1].child_s += elapsed

    @property
    def total_self_s(self) -> float:
        return sum(stat.self_s for stat in self.stats.values())

    def rows(self) -> list[dict[str, object]]:
        total_self = self.total_self_s

        rows: list[dict[str, object]] = []

        for name, stat in self.stats.items():
            rows.append(
                {
                    "bloq": name,
                    "calls": stat.calls,
                    "inclusive_s": stat.inclusive_s,
                    "self_s": stat.self_s,
                    "self_pct": (100.0 * stat.self_s / total_self if total_self > 0.0 else 0.0),
                    "avg_ms": (1000.0 * stat.inclusive_s / stat.calls if stat.calls else 0.0),
                    "max_ms": 1000.0 * stat.max_s,
                }
            )

        rows.sort(
            key=lambda row: float(row["self_s"]),
            reverse=True,
        )

        return rows


QUBIT_PROFILER = CostProfiler()
QEC_PROFILER = CostProfiler()

_ORIGINAL_GET_COST_VALUE = costing._get_cost_value


def _profiled_get_cost_value(
    bloq,
    cost_key,
    *,
    costs_cache,
    generalizer,
):
    if isinstance(cost_key, QECGatesCost):
        profiler = QEC_PROFILER

    elif isinstance(cost_key, QubitCount):
        # FastQubitCount は QubitCount の subclass なのでここに入る。
        profiler = QUBIT_PROFILER

    else:
        return _ORIGINAL_GET_COST_VALUE(
            bloq,
            cost_key,
            costs_cache=costs_cache,
            generalizer=generalizer,
        )

    def run():
        return _ORIGINAL_GET_COST_VALUE(
            bloq,
            cost_key,
            costs_cache=costs_cache,
            generalizer=generalizer,
        )

    return profiler.measure(
        bloq,
        run,
    )


def _timeout_handler(
    signum: int,
    frame: object,
) -> None:
    raise ProfileTimeout(f"profile timeout after {TIMEOUT_SECONDS:.0f} s")


def write_profile_csv(
    profiler: CostProfiler,
    path: Path,
) -> None:
    rows = profiler.rows()

    columns = (
        "bloq",
        "calls",
        "inclusive_s",
        "self_s",
        "self_pct",
        "avg_ms",
        "max_ms",
    )

    with path.open(
        "w",
        newline="",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_profile(
    title: str,
    profiler: CostProfiler,
) -> None:
    rows = profiler.rows()

    print()
    print("=" * 116)
    print(f"{title} profile (sorted by self time)")
    print("=" * 116)

    print(
        f"{'Bloq':<42}"
        f"{'calls':>9}"
        f"{'inclusive[s]':>16}"
        f"{'self[s]':>14}"
        f"{'self[%]':>11}"
        f"{'avg[ms]':>12}"
        f"{'max[ms]':>12}"
    )

    print("-" * 116)

    for row in rows:
        print(
            f"{row['bloq']!s:<42}"
            f"{int(row['calls']):>9}"
            f"{float(row['inclusive_s']):>16.6f}"
            f"{float(row['self_s']):>14.6f}"
            f"{float(row['self_pct']):>11.2f}"
            f"{float(row['avg_ms']):>12.3f}"
            f"{float(row['max_ms']):>12.3f}"
        )

    print("-" * 116)

    print(f"Total measured {title} self time: {profiler.total_self_s:.3f} s")


def save_profiles() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    qubit_csv = OUT_DIR / "qubit_count_profile.csv"

    qec_csv = OUT_DIR / "qec_gates_profile.csv"

    write_profile_csv(
        QUBIT_PROFILER,
        qubit_csv,
    )

    write_profile_csv(
        QEC_PROFILER,
        qec_csv,
    )

    print_profile(
        "QubitCount",
        QUBIT_PROFILER,
    )

    print()
    print(f"CSV: {qubit_csv.name}")

    print_profile(
        "QECGatesCost",
        QEC_PROFILER,
    )

    print()
    print(f"CSV: {qec_csv.name}")


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_handler = signal.getsignal(signal.SIGALRM)

    costing._get_cost_value = _profiled_get_cost_value

    timed_out = False

    start = time.perf_counter()

    try:
        signal.signal(
            signal.SIGALRM,
            _timeout_handler,
        )

        signal.setitimer(
            signal.ITIMER_REAL,
            TIMEOUT_SECONDS,
        )

        runpy.run_path(
            str(TARGET),
            run_name="__main__",
        )

    except ProfileTimeout:
        timed_out = True

    finally:
        elapsed = time.perf_counter() - start

        # timer を必ず解除。
        signal.setitimer(
            signal.ITIMER_REAL,
            0.0,
        )

        signal.signal(
            signal.SIGALRM,
            old_handler,
        )

        # Qualtran の monkey patch も必ず戻す。
        costing._get_cost_value = _ORIGINAL_GET_COST_VALUE

        if timed_out:
            print()
            print("=" * 116)
            print(f"TIMEOUT: stopped after {elapsed:.3f} s (limit={TIMEOUT_SECONDS:.0f} s)")
            print("=" * 116)

        save_profiles()


if __name__ == "__main__":
    main()
