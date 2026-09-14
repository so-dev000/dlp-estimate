import tempfile
import webbrowser
from pathlib import Path

from qualtran.drawing import GraphvizCallGraph
from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

from src.field import DLPInstance, FieldSpec
from src.shor import ShorDLP, make_shor_config

spec = FieldSpec(p=5, r=1, f=(0, 1))
instance = DLPInstance(spec=spec, q=4, q_factors=((2, 2),), g=(2,), h=(3,))
config = make_shor_config(instance)
bloq = ShorDLP(instance=instance, config=config)

print(f"p={spec.p} r={spec.r} f={spec.f} q={instance.q} q_factors={instance.q_factors}")
print(f"g={instance.g} h={instance.h} exponent_bits={config.exponent_bits}")
print(f"coefficient_bits={spec.coefficient_bits}")

counts = get_cost_value(bloq, QECGatesCost(legacy_shims=False))
qubits = get_cost_value(bloq, QubitCount())
print(f"logical_qubits: {qubits}")
for key in ("t", "toffoli", "cswap", "and_bloq", "clifford", "rotation", "measurement"):
    print(f"{key}: {getattr(counts, key)}")

with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", prefix="call-graph-") as f:
    f.write(GraphvizCallGraph.from_bloq(bloq).get_svg_bytes())
    path = Path(f.name)

webbrowser.open(path.as_uri())
print(f"opened: {path}")
