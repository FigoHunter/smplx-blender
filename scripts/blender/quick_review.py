import os
import numpy as np
from smplx_blender import utils,file,mesh,body,material
import bpy
import pickle
from tqdm import tqdm
import torch
import mathutils
import glob
from math import radians
from importlib import reload

print('=====================Start========================')

reload(body)
reload(mesh)
reload(file)
reload(utils)
reload(material)

res_x=800
res_y=600

root = os.path.join(utils.DATA_PATH, r"favor_preview\tmp\favor_pass1")

render_setup_path = os.path.join(utils.DATA_PATH,"render_setup.blend")
model_collection_path = os.path.join(utils.DATA_PATH,"objs.blend")
smplx_path=os.path.join(utils.DATA_PATH,"favor_preview","body_utils","body_models")
render_path = os.path.join(utils.DATA_PATH, "render")

coord=np.array([[1,0,0],[0,0,-1],[0,1,0]])

scenes = []
indicators_models={}
smplx_models={}
replaced_models={}
manip_models={}
replace_models={}

indicator_mats=[]
smplx_mats=[]
manip_mats=[]

for f in os.listdir(root):
    path = os.path.join(root, f)
    if os.path.isdir(path):
        scenes.append(os.path.basename(f))

def reset():
    indicators_models.clear()
    smplx_models.clear()
    replaced_models.clear()
    replace_models.clear()
    manip_models.clear()
    indicator_mats.clear()
    smplx_mats.clear()
    manip_mats.clear()


def loadScene(scene):
    file.openBlendFile(render_setup_path)
    model_collections = file.linkBlendFile(model_collection_path,collections=True)["collections"]
    for c in model_collections:
        bpy.context.scene.collection.objects.link(c)
        c.hide_render = True

def __loadMaterials(): 
    for mat in bpy.data.materials:
        if mat.name.startswith('indicator'):
            indicator_mats.append(mat)
        if mat.name.startswith('body'):
            smplx_mats.append(mat)
        if mat.name.startswith('manip'):
            manip_mats.append(mat)

def __loadSmplx(frame_data, smplx_model):
    smplx_frame_d = frame_data
    smplx_frame_d = {k: torch.tensor(v).to("cpu")[None] for k, v in smplx_frame_d.items()}
    smplx_frame_d["betas"] = torch.tensor(favor_data["betas"][None]).to("cpu")
    smplx_results = smplx_model(return_verts=True, **smplx_frame_d)
    verts = smplx_results.vertices.detach().cpu().numpy()[0]
    faces = smplx_model.faces
    return verts, faces

def __loadReplaceModels():
    for collection in bpy.data.collections:
        if collection.name == "Models":
            for obj in collection.objects:
                replace_models[obj.name] = obj
            break

def loadData(scene):
    pkl_path = os.path.join(root, scene+".pkl")
    print("pkl path: "+pkl_path)

    global favor_data,indicator_mats
    favor_data = pickle.load(open(pkl_path, "rb"))

    __loadMaterials()

    for i,i_mesh in tqdm(enumerate(favor_data["indicator_meshes"])):
        name=f"indicator_{str(i)}"
        verts=i_mesh["verts"]
        faces=i_mesh["faces"]
        mat = indicator_mats[i%4]
        obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=mat, matrix=coord)
        indicators_models[obj.name]=(obj)

    frames = [0,favor_data["moving_frame"][0],favor_data["moving_frame"][1]]

    bm = body.get_body_model(smplx_path,"smplx", "female", 1, device="cpu")
    affine_coord=mathutils.Matrix(utils.getAffineMat(coord))

    for i,fid in tqdm(enumerate(frames)):
        data = favor_data["smplx"][fid]
        verts,faces = __loadSmplx(data, bm)
        name="smplx_" + str(i)
        mat_index = (int(i/(len(frames)-1)*(len(smplx_mats)-1))+1)%len(smplx_mats)
        obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=smplx_mats[mat_index], matrix=coord)
        smplx_models[obj.name]=obj

        verts = favor_data["fixed_obj_mesh"]["verts"]
        faces = favor_data["fixed_obj_mesh"]["faces"]
        name="manip_" + str(i)
        mat_index = (int(i/(len(frames)-1)*(len(manip_mats)-1))+1)%len(manip_mats)
        obj=mesh.createMesh(name, vertices=verts, faces=faces, mat=manip_mats[mat_index], matrix=coord)
        trs = mathutils.Matrix(favor_data["obj_pose"][fid])
        obj.matrix_world = affine_coord@trs@affine_coord.inverted()
        manip_models[obj.name]=obj

    replace_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1",scene)
    replace_paths = glob.glob(os.path.join(replace_dir,"*.npz"))
    __loadReplaceModels()
    for path in replace_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        rep_from = name.split("-",1)[0]
        rep_to = name.split("-",1)[1]

        replace = np.load(path)
        trs = mathutils.Matrix(replace["arr_0"])
        if rep_to not in replace_models:
            print("skip: "+rep_to)
            continue
        to_obj_src = replace_models[rep_to]
        to_obj = to_obj_src.copy()
        bpy.context.scene.collection.objects.link(to_obj)
        offset = mathutils.Matrix.Rotation(radians(90),4,'X')

        res = affine_coord@trs@affine_coord.inverted()@offset
        to_obj.matrix_world=res
        replaced_models[to_obj.name] = to_obj

def __render(filename):
    path = os.path.join(render_path, filename)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.ops.render.render()
    bpy.data.images["Render Result"].save_render(path)


def render_replaced_models(scene):
    for o in indicators_models.values():
        o.hide_render=True
    for o in smplx_models.values():
        o.hide_render=True
    for o in replaced_models.values():
        o.hide_render=False
    for o in manip_models.values():
        o.hide_render=True
    __render(os.path.join(scene,"0.png"))

def render_indicator_models(scene):
    for o in indicators_models.values():
        o.hide_render=False
    for o in smplx_models.values():
        o.hide_render=True
    for o in replaced_models.values():
        o.hide_render=True
    for o in manip_models.values():
        o.hide_render=True
    __render(os.path.join(scene,"1.png"))

def render_grab_models(scene):
    for o in indicators_models.values():
        o.hide_render=False
    for o in smplx_models.values():
        o.hide_render=False
    for o in replaced_models.values():
        o.hide_render=True
    for o in manip_models.values():
        o.hide_render=False
    __render(os.path.join(scene,"2.png"))

for scene in scenes:
    loadScene(scene)
    loadData(scene)
    render_replaced_models(scene)
    render_indicator_models(scene)
    render_grab_models(scene)
    reset()