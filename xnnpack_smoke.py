from voidx_experiment import load_local_meshes
from voidx_pipeline import VoidXPipeline

mesh = load_local_meshes(r"C:\Users\diwak\Downloads\HumanRig_test")[0]["mesh_path"]
print(f"Smoke mesh: {mesh}")
for i in range(10):
    result = VoidXPipeline().run(mesh)      # create -> run -> destroy each time
    print(f"  pipeline {i+1}/10 OK ({len(result['pose'])} joints)")
print("XNNPACK FIX VERIFIED — safe to run the ablations.")