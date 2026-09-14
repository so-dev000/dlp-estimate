import argparse

import cirq
from cirq.contrib.svg import SVGCircuit
from qualtran._infra.composite_bloq import DidNotFlattenAnythingError

from src.field import DLPInstance, FieldSpec
from src.shor import DLPFunctionOracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    spec = FieldSpec(p=5, r=1, f=(0, 1))
    instance = DLPInstance(spec=spec, q=4, q_factors=((2, 2),), g=(2,), h=(1,))
    bloq = DLPFunctionOracle(instance=instance, exponent_bits=2)

    cbloq = bloq.decompose_bloq()
    for _ in range(args.depth - 1):
        try:
            cbloq = cbloq.flatten_once()
        except DidNotFlattenAnythingError:
            break

    try:
        circuit = cirq.Circuit(cbloq.to_cirq_circuit())
    except ValueError as e:
        raise SystemExit(e) from e
    with open("cirq.svg", "w") as f:
        f.write(SVGCircuit(circuit)._repr_svg_())
    print(f"wrote cirq.svg: depth={args.depth}")


if __name__ == "__main__":
    main()
