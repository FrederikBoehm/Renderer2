import bpy
import random


for i in range(1000):
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    
    rotX = random.uniform(-0.17, 0.17)
    rotY = random.uniform(-0.17, 0.17)
    bpy.ops.mesh.primitive_plane_add(location=(x, y, 0), rotation=(rotX, rotY, 0))