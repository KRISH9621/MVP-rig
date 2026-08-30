# fbx_to_obj_trimesh.py — run in your NORMAL terminal (alpha env)
import trimesh

FBX = r"C:\Users\diwak\Desktop\HumanRig_test\char93\mesh.fbx"""   # ← your FBX filename
OBJ = r"C:\Users\diwak\Desktop\HumanRig_test\char93\model.obj"

scene = trimesh.load(FBX)
if isinstance(scene, trimesh.Scene):
    geoms = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph.get(node)
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        geoms.append(g)
    mesh = trimesh.util.concatenate(geoms)
else:
    mesh = scene

mesh.export(OBJ)
print(f"Exported {len(mesh.vertices)} vertices, {len(mesh.faces)} faces -> {OBJ}")