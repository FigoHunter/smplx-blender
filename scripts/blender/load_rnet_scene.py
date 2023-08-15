import bpy
import os
from smplx_blender import utils,mesh
import rosita_2023_08.load
import numpy as np
from tqdm import tqdm
from importlib import reload
from smplx_blender.ops import rosita_materials 
import re

reload(utils)
reload(mesh)
reload(rosita_2023_08.load)

target="A002-2023-0419-1400-37-task-59-seq-56-dispenser11.pkl"

rnet_path = os.path.join(utils.DATA_PATH, "rnet_extracted", target)
gt_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")
extracted = rosita_2023_08.load.loadRnetData(rnet_path)

gt_target = re.findall(r'\S+-[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{2}-task-[0-9]+-seq-[0-9]+',target)[0]


coord=np.array([[1,0,0],[0,0,-1],[0,1,0]])

start_frame = 0
table_height = 0.84

body_verts_seq=extracted["body"]["verts"]
body_faces = extracted["body"]["faces"]
manip_verts_seq=extracted["manip_obj"]["verts"]
manip_faces=extracted["manip_obj"]["faces"]

offset = extracted["offset"]@coord
offset = (-offset[0], offset[1], table_height)

skip = int(len(body_verts_seq)/3)-1


frames = [*range(start_frame,len(body_verts_seq),skip)]

print(offset[0])
print(offset[1])
print(offset[2])

bpy.context.scene.objects["Environment"].location=offset

bodies=[]
manips=[]

for fid in frames:
    body_verts = body_verts_seq[fid]
    manip_verts = manip_verts_seq[fid]
    name="smplx_" + str(fid)
    obj = mesh.createMesh(name,vertices=body_verts,faces=body_faces, matrix=coord)
    bodies.append(obj)

    name="manip_" + str(fid)
    obj=mesh.createMesh(name,vertices=manip_verts,faces=manip_faces, matrix=coord)
    manips.append(obj)

gt_obj = bpy.data.objects.new("GroundTruthData", None)
gt_obj.location=(0,0,0)
bpy.context.scene.collection.objects.link(gt_obj)


favor_data = rosita_2023_08.load.loadGroundTruthData(os.path.join(gt_dir,gt_target+".pkl"))
indicators = rosita_2023_08.load.loadGroundTruthIndicator(favor_data,matrix=coord)

rosita_materials.assign_body_materials(bodies)
rosita_materials.assign_indicator_materials(indicators)
rosita_materials.assign_manip_materials(manips)
for indicator in indicators:
    indicator.parent=gt_obj

gt_obj.location=offset