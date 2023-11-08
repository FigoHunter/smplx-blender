import bpy
import os
from smplx_blender import utils,mesh,file,render,body,material,bobject
import rosita_2023_08.load
import rosita_2023_08.smplx_model
import rosita_2023_08.replace_model
import numpy as np
from importlib import reload
from rosita_2023_08.ops import rosita_materials
import blender_figo.collection
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
skip=5
overwrite = False

print("datadir: ")
print(datadir)
print("")

if not os.path.exists(datadir):
    raise Exception("File Not Found: "+datadir)
if not os.path.isdir(datadir) and os.path.splitext(datadir)[1] != "pkl":
    raise Exception("File Extension Not Supported: "+datadir)
data_list=[ os.path.basename(os.path.dirname(p)) for p in glob(os.path.join(datadir,'**',"*.npz"))]

data_name = data_list[10]

data_path = os.path.join(datadir,data_name+'.pkl')

fid=0

blender_figo.collection.removeCollection('Replaced', True)
blender_figo.collection.removeCollection('Manip', True)
blender_figo.collection.removeCollection('Indicator', True)
blender_figo.collection.removeCollection('Body', True)

rep_data = rosita_2023_08.replace_model.get_replace_data(data_name)
print(rep_data)
data_gt=rosita_2023_08.load.loadGroundTruthData(data_path)
replace_list = rosita_2023_08.replace_model.load_replaced_objects(rep_data, collection=blender_figo.collection.getOrNewCollection('Replaced'))
manip_obj=rosita_2023_08.load.loadGroundTruthManip(data_gt,frames=fid,
    mat=material.getMaterialByName("manip"),matrix=rosita_2023_08.coord_matrix, collection=blender_figo.collection.getOrNewCollection('Manip'))
