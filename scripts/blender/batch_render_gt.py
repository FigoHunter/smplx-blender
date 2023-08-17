import bpy
import os
from smplx_blender import utils,mesh,file,render,body,material,bobject
import rosita_2023_08.load
import rosita_2023_08.smplx_model
import numpy as np
from importlib import reload
from smplx_blender.ops import rosita_materials 
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

for data_path in data_list:
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    
    file.openBlendFile(render_setup_path)

    data_gt=rosita_2023_08.load.loadGroundTruthData(data_path)
    indicator_list = rosita_2023_08.load.loadGroundTruthIndicator(data_gt,matrix=rosita_2023_08.coord_matrix)
    rosita_materials.assign_indicator_materials(indicator_list)

    grab_frame,end_frame=rosita_2023_08.load.loadGroundTruthMovingFrame(data_gt)
    frames = data_gt["length"]

    smplx_model=rosita_2023_08.smplx_model.getBodyModel('female')

    id = 0

    for fid in [*range(0,min(frames,end_frame+int(120/skip*4)),skip)]:
        render_dir=os.path.join(outputdir,data_name)
        render_path=os.path.join(render_dir,"{:05d}.png".format(id))

        if not overwrite and os.path.exists(render_path):
            id=id+1
            continue

        body_obj = rosita_2023_08.load.loadGroundTruthBody(data_gt,smplx_model,
            frames=fid,mat=material.getMaterialByName("body"),matrix=rosita_2023_08.coord_matrix)
        
        manip_obj=rosita_2023_08.load.loadGroundTruthManip(data_gt,frames=fid,
            mat=material.getMaterialByName("manip"),matrix=rosita_2023_08.coord_matrix)

        render.render(render_path)

        bobject.removeObject(body_obj)
        bobject.removeObject(manip_obj)

        id=id+1

    input=os.path.join(render_dir,"%5d.png")
    output=os.path.join(render_dir,"output.mp4")
    os.system(f"ffmpeg -y -f image2 -i {input} -pix_fmt yuv420p -c:v libx264 {output}")