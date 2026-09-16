import webbrowser
from pathlib import Path

from qualtran.drawing import GraphvizCallGraph

from src.field import DLPInstance, FieldSpec
from src.shor import ShorDLP, make_shor_config

p = 3
spec = FieldSpec(p=p, r=2, f=(1, 0, 1))

instance = DLPInstance(
    spec=spec,
    q=spec.order - 1,
    q_factors=((2, 3),),
    g=(1, 1),
    h=(1, 2),
)

config = make_shor_config(instance)
bloq = ShorDLP(instance=instance, config=config)

# max_depth:
#   1: ShorDLP直下だけ
#   2: 算術Bloqまで
#   3+: より低レベルのBloqまで
drawer = GraphvizCallGraph.from_bloq(
    bloq,
    max_depth=2,
    agg_gate_counts="factored",
)

# SVGとして保存
output = Path("svg/shor_dlp_call_graph.svg").resolve()
output.write_bytes(drawer.get_svg_bytes())

print(f"Saved to: {output}")

# ブラウザで開く
webbrowser.open_new_tab(output.as_uri())
