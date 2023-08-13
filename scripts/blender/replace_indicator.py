import numpy as np
import os
from smplx_blender import utils
import glob
import bpy
import mathutils
from math import radians

pkl = "A001-2023-0511-1802-23-task-243-seq-12"

# pkl_path = os.path.join(r".\data",r"favor_preview\tmp\favor_pass1",pkl+".pkl")
# replace_dir = os.path.join(r".\data",r"favor_preview\tmp\favor_pass1",pkl)

pkl_path = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1",pkl+".pkl")
replace_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1",pkl)


matrix=np.array([[1,0,0],[0,0,-1],[0,1,0]])
affine_mat = utils.getAffineMat(matrix)

replace_paths = glob.glob(os.path.join(replace_dir,"*.npz"))

print(replace_paths)

models = {}

indicators={}

for collection in bpy.data.collections:
   if collection.name == "Models":
        model_collection = collection
        for obj in collection.objects:
            models[obj.name] = obj
        break

scene_collection = bpy.context.scene.collection
for obj in scene_collection.objects:
    if obj.name.startswith("indicator"):
        indicators[obj.name] = obj



for path in replace_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    rep_from = name.split("-",1)[0]
    rep_to = name.split("-",1)[1]

    replace = np.load(path)
    trs = mathutils.Matrix(replace["arr_0"])
    affine=mathutils.Matrix(affine_mat)

    from_obj = indicators["indicator_"+rep_from]
    
    if rep_to not in models:
        print("skip: "+rep_to)
        continue
    to_obj_src = models[rep_to]
    to_obj = to_obj_src.copy()
    scene_collection.objects.link(to_obj)
    offset = mathutils.Matrix.Rotation(radians(90),4,'X')

    res = affine@trs@affine.inverted()@offset
    to_obj.matrix_world=res


