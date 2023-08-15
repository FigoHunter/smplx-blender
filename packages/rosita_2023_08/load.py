import os
import pickle
from smplx_blender import mesh

def loadRnetData(path):
    data = pickle.load(open(path, "rb"))
    return data

def loadGroundTruthData(path):
    favor_data = pickle.load(open(path, "rb"))
    return favor_data

def loadGroundTruthIndicator(data, mat=None, matrix=None):
    indicator_meshes = data["indicator_meshes"]
    index=0
    objs=[]
    for i_mesh in indicator_meshes:
        name=f"indicator_{str(index)}"
        verts=i_mesh["verts"]
        faces=i_mesh["faces"]
        obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=mat, matrix=matrix)
        objs.append(obj)
        index=index+1
    return objs
        