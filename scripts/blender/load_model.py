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

print("======================Start=============================")

skip=300
start_frame=0

device = "cpu"
gender = "female"

pkl="A001-2023-0511-1940-05-task-349-seq-36"

model_path=os.path.join(utils.DATA_PATH,"favor_preview","body_utils","body_models")
path=os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1", pkl+".pkl")
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

split_frame = moving_frame[0]
end_frame = moving_frame[1]
fids = [*range(start_frame, split_frame, int((split_frame-start_frame)/4)-1)]
fids.extend([*range(split_frame,end_frame,int((end_frame-split_frame)/4)-1)])
print(fids)


matrix=np.array([[1,0,0],[0,0,-1],[0,1,0]])
affine_mat = matrix.copy()
affine_mat = np.insert(affine_mat,3,values=[0,0,0],axis=1)
affine_mat = np.insert(affine_mat,3,values=[0,0,0,1],axis=0)
print(affine_mat)
# indicator
whiteMat = material.createDiffuseMaterial(0.8,0.8,0.8,1)

for i_mesh in indicator_meshes:
    name=f"indicator_{str(index)}"
    verts=i_mesh["verts"]
    faces=i_mesh["faces"]
    obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)

    index=index+1


# smplx
for fid in fids:
    smplx_frame_d = smplx_data[fid]
    smplx_frame_d = {k: torch.tensor(v).to(device)[None] for k, v in smplx_frame_d.items()}
    smplx_frame_d["betas"] = torch.tensor(favor_data["betas"][None]).to(device)
    smplx_results = bm(return_verts=True, **smplx_frame_d)

    verts = smplx_results.vertices.detach().cpu().numpy()[0]
    faces = bm.faces
    name="smplx_" + str(fid)
    obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)

    # manipulated
    verts = fixed_obj_mesh["verts"]
    
    faces = fixed_obj_mesh["faces"]
    name="manip_" + str(fid)
    obj=mesh.createMesh(name, vertices=verts, faces=faces, mat=whiteMat, matrix=matrix)

    trs = obj_pose[fid]
    obj.matrix_world=mathutils.Matrix(np.matmul(np.matmul(affine_mat,trs), np.linalg.inv(affine_mat)))
    