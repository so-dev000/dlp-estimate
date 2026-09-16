from src.field import DLPInstance, FieldSpec
from src.logical_resources import estimate_resources
from src.shor import ShorDLP, make_shor_config

p = 3
spec = FieldSpec(p=p, r=2, f=(1, 0, 1))

instance = DLPInstance(spec=spec, q=spec.order - 1, g=(1, 1), h=(1, 2))

config = make_shor_config(instance)
bloq = ShorDLP(instance=instance, config=config)

resources = estimate_resources(bloq)


print(f"GF({p}^{spec.r})")
print(f"exponent bits: {config.exponent_bits}")
print(f"logical qubits: {resources.logical_qubits}")
print(f"T: {resources.gates.t}")
print(f"Toffoli: {resources.gates.toffoli}")
print(f"CSwap: {resources.gates.cswap}")
print(f"And: {resources.gates.and_bloq}")
print(f"Clifford: {resources.gates.clifford}")
print(f"Rotation: {resources.gates.rotation}")
print(f"Measurement: {resources.gates.measurement}")
