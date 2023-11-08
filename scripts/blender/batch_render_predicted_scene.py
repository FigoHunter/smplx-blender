import bpy
import os
from smplx_blender import utils,mesh,file,render
from blender_figo import batch_process
import numpy as np
from glob import glob
import traceback

import load_predicted_scene

filename="predicted.png"
res=(1440,1080)

render_setup_path = os.path.join(utils.DATA_PATH,"render_setup.blend")
mnet_dir=rnet_path = os.path.join(utils.DATA_PATH, "mnet_extracted")
outputdir = os.path.join(utils.DATA_PATH,r"render\rebuttle\predicted")

print("datadir: ")
print(mnet_dir)
print("")
print("outputdir: ")
print(outputdir)
print("")

if os.path.exists(outputdir) and not os.path.isdir(outputdir):
    raise Exception("Output Path Not Valid: "+outputdir)
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

mnet_pkls=[]

for f in os.listdir(mnet_dir):
    path = os.path.join(mnet_dir, f)
    if f.endswith(".pkl"):
        mnet_pkls.append(os.path.basename(f))

print("data_list: ")
print(mnet_pkls)
print("")

for target in mnet_pkls:
    try:
        path=os.path.join(outputdir, os.path.splitext(target)[0],filename)
        if os.path.exists(path):
            print(f'skip: {target}')
            continue
        file.openBlendFile(render_setup_path)
        load_predicted_scene.load_predicted_scene(target)
        render.render(path,res_x=res[0],res_y=res[1])
    except Exception as e:
        print(e)
        traceback.print_exc()
        # input('press enter to resume')
