import bpy
from .errors import MeshError
from . import material,utils
import numpy as np

def createMesh(name,*, vertices, faces=[], edges=[],matrix=None, mat = None):

    vertices = utils.ndarray_pydata.parse(vertices)
    if len(faces)>0:
        faces = utils.ndarray_pydata.parse(faces)

    # 创建mesh
    mesh = bpy.data.meshes.new(name)
    if vertices:
        if matrix is not None:
            vertices = np.transpose(np.matmul(matrix,np.transpose(vertices)))
        mesh.from_pydata(vertices, edges, faces)
        mesh.validate()


    # 创建对象
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)


    # 赋材质
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    if not mat:
        mat = material.createDiffuseMaterial(0.8,0,8,0.8)
    obj.active_material = mat
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action='DESELECT')
    return obj