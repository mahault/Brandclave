"""Debug Genius API permissions."""
import json
from services.active_inference.genius_client import GeniusClient, VFGBuilder

client = GeniusClient()

print("1. Health check:")
print(f"   Healthy: {client.health_check()}")

print("\n2. Version:")
print(json.dumps(client.get_version(), indent=2))

print("\n3. License metadata:")
print(json.dumps(client.get_license_metadata(), indent=2))

print("\n4. Trying to get existing graph:")
graph = client.get_graph()
print(f"   Graph exists: {graph is not None}")
if graph:
    print(json.dumps(graph, indent=2)[:500])

print("\n5. Trying to create a simple graph:")
vfg = (
    VFGBuilder()
    .add_variable("test_var", ["a", "b", "c"])
    .add_categorical_factor("test_prior", "test_var", [0.33, 0.33, 0.34])
    .build()
)
print(f"   VFG: {json.dumps(vfg, indent=2)}")

try:
    result = client.set_graph(vfg)
    print(f"   Set graph result: {result}")
except Exception as e:
    print(f"   Set graph error: {e}")

print("\n6. Trying inference on existing graph (if any):")
try:
    result = client.infer(variables=["test_var"], wait=True)
    print(f"   Infer result: {result}")
except Exception as e:
    print(f"   Infer error: {e}")

client.close()
