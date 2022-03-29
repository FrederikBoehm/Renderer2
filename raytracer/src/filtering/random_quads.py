import bpy
import random

def init():
    cube = bpy.data.objects.get("Cube")
    if cube:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

def generate_planes():
    for i in range(1000):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        
        rotX = random.uniform(-0.17, 0.17)
        rotY = random.uniform(-0.17, 0.17)
        bpy.ops.mesh.primitive_plane_add(location=(x, y, 0), rotation=(rotX, rotY, 0))

def join_planes():
    scene = bpy.context.scene

    obs = []
    for ob in scene.objects:
        if ob.type == 'MESH':
            obs.append(ob)

    ctx = bpy.context.copy()

    ctx['active_object'] = obs[0]

    # ctx['selected_objects'] = obs
    # In Blender 2.8x this needs to be the following instead:
    ctx['selected_editable_objects'] = obs

    # We need the scene bases as well for joining.
    # Remove this line in Blender >= 2.80!
    # ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]

    bpy.ops.object.join(ctx)

def generate_material():
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        if ob.type == 'MESH':
            obs.append(ob)
            
    bpy.context["active_object"] = obs[0]

    # Get material
    mat = bpy.data.materials.get("Material")
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name="Material")

    mat.use_nodes = True

    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.800000, 0.442748, 0.231366, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.001
    bsdf.inputs["Specular"].default_value = 0.95
    bsdf.inputs["IOR"].default_value = 2.0

    

if __name__ == "__main__":
    init()
    generate_planes()
    join_planes()
    # generate_material()