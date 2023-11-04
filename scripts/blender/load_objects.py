import bpy
import os
from smplx_blender import utils,mesh,file,render,body,material,bobject
import rosita_2023_08.load
import rosita_2023_08.smplx_model
import numpy as np
from importlib import reload
from rosita_2023_08.ops import rosita_materials 
from glob import glob

# reload(utils)
# reload(mesh)
# reload(file)
# reload(render)
# reload(rosita_2023_08.load)
# reload(rosita_materials)
# reload(body)

render_setup_path = os.path.join(utils.DATA_PATH,"render_setup.blend")

# parser.add_argument('--skip', type='int', help="跳帧")

datadir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")
outputdir = os.path.join(utils.DATA_PATH,r"render\gt_videos")
skip=5
overwrite = False

print("datadir: ")
print(datadir)
print("")
print("outputdir: ")
print(outputdir)
print("")

if not os.path.exists(datadir):
    raise Exception("File Not Found: "+datadir)
if not os.path.isdir(datadir) and os.path.splitext(datadir)[1] != "pkl":
    raise Exception("File Extension Not Supported: "+datadir)

if os.path.exists(outputdir) and not os.path.isdir(outputdir):
    raise Exception("Output Path Not Valid: "+outputdir)
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

if os.path.isdir(datadir):
    data_list=glob(os.path.join(datadir,"*.pkl"))
else:
    data_list=[datadir]

print("data_list: ")
print(data_list)
print("")

data_path = data_list[0]
fid=0
data_name = os.path.splitext(os.path.basename(data_path))[0]

file.openBlendFile(render_setup_path)

data_gt=rosita_2023_08.load.loadGroundTruthData(data_path)
indicator_list = rosita_2023_08.load.loadGroundTruthIndicator(data_gt,matrix=rosita_2023_08.coord_matrix)
rosita_materials.assign_indicator_materials(indicator_list)

grab_frame,end_frame=rosita_2023_08.load.loadGroundTruthMovingFrame(data_gt)
frames = data_gt["length"]

smplx_model=rosita_2023_08.smplx_model.getBodyModel('female')

    
manip_obj=rosita_2023_08.load.loadGroundTruthManip(data_gt,frames=fid,
    mat=material.getMaterialByName("manip"),matrix=rosita_2023_08.coord_matrix)

render.render(render_path)

