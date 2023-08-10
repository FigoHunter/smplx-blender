import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
from smplx_blender import body,mesh,material,utils
print(torch.__version__)
print(torch.cuda.is_available())

model_path=os.path.join(utils.DATA_PATH,"favor_preview","body_utils","body_models")
path=os.path.join(utils.DATA_PATH,"zenan","A002-2023-0419-1400-37-task-5-seq-5-cylinder_bottle_s341.pkl")


skip=100
start_frame=0

matrix=np.array([[-1,0,0],[0,0,1],[0,1,0]])


device = "cpu"
gender = "female"

bm = body.get_body_model(model_path,"smplx", gender, 1, device=device)
result = pickle.load(open(path, "rb"))

def parms_6D2full(pose, trans):

    bs = trans.shape[0]
    pose = pose.reshape([bs, -1, 3, 3])

    body_parms = full2bone(pose, trans)
    body_parms['fullpose_rotmat'] = pose

    return body_parms

def full2bone(pose, trans):

    global_orient = pose[:, 0:1]
    body_pose = pose[:, 1:22]
    jaw_pose = pose[:, 22:23]
    leye_pose = pose[:, 23:24]
    reye_pose = pose[:, 24:25]
    left_hand_pose = pose[:, 25:40]
    right_hand_pose = pose[:, 40:]

    body_parms = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'transl': trans
    }
    return body_parms


batch=result['batch_gt']

body_params_gt = parms_6D2full(batch['fullpose_rotmat'][0], batch['transl'][0])
body_params_gt['betas']=torch.repeat_interleave(batch['betas'].clone(),22, dim=0).reshape(-1,10).to(device)
sbj_output_gt = bm(**body_params_gt)
v_gt = sbj_output_gt.vertices.reshape(-1, 10475, 3)##(22, 10475, 3)

for fid,vertices in tqdm(enumerate(v_gt[start_frame::skip])):
    name = str(fid)
    mesh.createMesh(name, vertices=vertices, faces=bm.faces, matrix=matrix)