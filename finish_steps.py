from voidx_experiment import (load_local_meshes, run_ablation_experiment,
                              run_runtime_experiment)

dataset = load_local_meshes(r"C:\Users\diwak\Downloads\HumanRig_test")
run_ablation_experiment(dataset, r".\results", num_meshes=10)
run_runtime_experiment(dataset, r".\results")
print("Done — ablations.json and runtime.json refreshed.")