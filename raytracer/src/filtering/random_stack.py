import bpy
import random


for i in range(20):
    z = random.uniform(-5, 5)

    bpy.ops.mesh.primitive_plane_add(location=(0, 0, z), size=10)