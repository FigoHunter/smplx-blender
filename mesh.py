import bpy
from .errors import MeshError
from . import material,utils
import numpy as np

def createMesh(name,*, vertices, faces, matrix=None,edges=[], mat = None):

    vertices = utils.ndarray_pydata.parse(vertices)
    faces = utils.ndarray_pydata.parse(faces)
    
    print(vertices.__class__.__name__)
    print(matrix.__class__.__name__)

    # 创建mesh
    mesh = bpy.data.meshes.new(name)
    if vertices:
        print("Matrix: ")
        print(matrix)
        print("Verts")
        print(vertices)
        if matrix is not None:
            vertices = np.transpose(np.matmul(matrix,np.transpose(vertices)))
        if not faces:
            raise MeshError("faces is not defined")
        mesh.from_pydata(vertices, edges, faces)
        mesh.validate()


    # 创建对象
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # 赋材质
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    if not mat:
        mat = material.createDiffuseMaterial()
    obj.active_material = mat
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action='DESELECT')
    return obj