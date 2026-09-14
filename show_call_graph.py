import tempfile
import webbrowser
from pathlib import Path

from qualtran.drawing import GraphvizCallGraph

from src.field import DLPInstance, FieldSpec
from src.shor import DLPOracle, make_shor_config

spec = FieldSpec(p=5, r=1, f=(0, 1))
instance = DLPInstance(spec=spec, q=4, q_factors=((2, 2),), g=(2,), h=(3,))
config = make_shor_config(instance)
bloq = DLPOracle(instance=instance, exponent_bits=config.exponent_bits)
with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", prefix="call-graph-") as f:
    f.write(GraphvizCallGraph.from_bloq(bloq, max_depth=3).get_svg_bytes())
    path = Path(f.name)

webbrowser.open(path.as_uri())
print(f"opened: {path}")
