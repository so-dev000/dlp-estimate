from qualtran import BloqBuilder, CBit, CtrlSpec, SoquetT
from qualtran.bloqs.basic_gates import Hadamard, MeasureZ, ZPowGate


# https://arxiv.org/abs/quant-ph/9511007
def semiclassical_qft_step(
    bb: BloqBuilder,
    bit: SoquetT,
    history: list[SoquetT],
    *,
    inverse: bool = False,
):
    history = list(history)
    sign = -1 if inverse else 1
    # ZPowGateはdiag(1, exp(iπe))なので、P(φ_t)=diag(1, exp(2πiφ_t))を実装するにはe=2φ_t
    for d in range(1, len(history) + 1):
        feed_forward = ZPowGate(exponent=sign * (2.0**-d)).controlled(
            CtrlSpec(qdtypes=CBit(), cvs=1)
        )
        (ctrl_name,) = feed_forward.ctrl_reg_names
        out = bb.add_d(feed_forward, **{ctrl_name: history[-d], "q": bit})
        history[-d], bit = out[ctrl_name], out["q"]

    bit = bb.add(Hadamard(), q=bit)
    c_new = bb.add(MeasureZ(), q=bit)
    history.append(c_new)
    return history
