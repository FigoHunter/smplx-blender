import os
import pickle
import numpy as np
import smplx
import torch
from tqdm import tqdm
from importlib import reload
import mathutils
import bpy


from smplx_blender import body,mesh,material,utils

reload(body)
reload(mesh)
reload(material)
reload(utils)

skip=300
start_frame=300

device = "cpu"
gender = "female"

model_path=os.path.join(utils.DATA_PATH,"favor_preview","body_utils","body_models")
path=os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1\A004-2023-0526-1808-38-task-1159-seq-73.pkl")
table_path=os.path.join(utils.DATA_PATH,"favor_preview","assets","table_only.obj")

bm = body.get_body_model(model_path,"smplx", gender, 1, device=device)

favor_data = pickle.load(open(path, "rb"))

frames = favor_data["length"]
obj_pose = favor_data["obj_pose"]
obj_mesh = favor_data["obj_mesh"]
fixed_obj_mesh = favor_data["fixed_obj_mesh"]
moving_frame = favor_data["moving_frame"]
subject_name = favor_data["subject_name"]
indicator_meshes = favor_data["indicator_meshes"]
smplx_data = favor_data["smplx"]

index=0

matrix=np.array([[-1,0,0],[0,0,1],[0,1,0]])
affine_mat = matrix.copy()
affine_mat = np.insert(affine_mat,3,values=[0,0,0],axis=1)
affine_mat = np.insert(affine_mat,3,values=[0,0,0,1],axis=0)
print(affine_mat)
# indicator
whiteMat = material.createDiffuseMaterial(0.8,0.8,0.8,1)

for i_mesh in indicator_meshes:
    index=index+1
    name=f"indicator_{str(index)}"
    verts=i_mesh["verts"]
    faces=i_mesh["faces"]
    mesh.createMesh(name, vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)

# smplx
for fid, smplx_frame_d in tqdm(enumerate(smplx_data[start_frame:: skip])):
    smplx_frame_d = {k: torch.tensor(v).to(device)[None] for k, v in smplx_frame_d.items()}
    smplx_frame_d["betas"] = torch.tensor(favor_data["betas"][None]).to(device)
    smplx_results = bm(return_verts=True, **smplx_frame_d)

    verts = smplx_results.vertices.detach().cpu().numpy()[0]
    faces = bm.faces
    name=str(fid)
    mesh.createMesh(name, vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)

    # manipulated
    verts = fixed_obj_mesh["verts"]
    
    faces = fixed_obj_mesh["faces"]
    obj=mesh.createMesh("manip", vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)
    trs = obj_pose[fid * skip + start_frame]
    obj.matrix_world=mathutils.Matrix(np.matmul(np.matmul(affine_mat,trs), np.linalg.inv(affine_mat)))
    