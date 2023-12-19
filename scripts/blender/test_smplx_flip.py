import os
import rosita_2023_08.load
import rosita_2023_08.smplx_model
import numpy as np
from glob import glob
from smplx_blender import utils
import torch
from scipy.spatial.transform import Rotation as R

target="A002-2023-0419-1400-37-task-0-seq-2.pkl"

datadir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")

# 原始Favor数据加载
data_gt=rosita_2023_08.load.loadGroundTruthData(os.path.join(datadir, target))
smplx_model=rosita_2023_08.smplx_model.getBodyModel('female')

fid = data_gt["moving_frame"][0]
smplx_data = data_gt["smplx"]

smplx_frame_d = smplx_data[fid]
smplx_frame_d = {k: torch.tensor(v).to("cpu")[None] for k, v in smplx_frame_d.items()}
smplx_frame_d["betas"] = torch.tensor(data_gt["betas"][None]).to("cpu")
smplx_results = smplx_model(return_verts=True, **smplx_frame_d)

# 原始Favor数据构建blender mesh渲染
from smplx_blender import mesh
from blender_figo import collection
collection.removeCollection("smplx")
c = collection.getOrNewCollection("smplx")
verts = smplx_results.vertices.detach().cpu().numpy()[0]
faces = smplx_model.faces
obj = mesh.createMesh("1", vertices=verts, faces=faces, matrix=rosita_2023_08.coord_matrix,collection=c)

'''
smplx_frame_d 数据

{'transl': tensor([[0.0124, 0.4093, 0.4591]]), 'global_orient': tensor([[-0.2581, -2.8783,  0.4164]]), 'body_pose': tensor([[-3.9070e-01, -1.1346e-01, -1.0630e-01, -1.8893e-01, -1.2868e-02,
         -2.0085e-01,  1.3649e-02,  5.7270e-02, -1.3235e-01,  3.8643e-01,
         -1.4892e-01, -1.6027e-01,  4.1869e-01,  5.6149e-02,  4.3360e-02,
         -8.2642e-02, -6.8793e-02,  2.7583e-02, -3.2431e-01,  1.9919e-01,
          1.2039e-01, -4.9693e-01, -2.2798e-01,  6.0510e-02,  1.1018e-01,
         -2.0510e-02,  1.5644e-02, -3.0448e-03,  8.9696e-03, -2.6207e-02,
          9.8709e-04,  1.5896e-02,  1.4589e-02,  1.5649e-01, -3.0198e-01,
          1.3055e-01,  1.1411e-02,  1.2905e-01,  1.2203e-01, -1.0220e-01,
          1.0292e-01,  4.4686e-02,  2.3753e-01, -5.0916e-01,  1.3838e-01,
         -2.8471e-02, -2.1987e-01, -3.9914e-01,  3.7057e-02,  3.6818e-01,
          5.9549e-01,  2.7364e-01, -1.0353e+00,  5.9058e-01,  2.5465e-01,
          5.5741e-01, -3.1697e-01,  2.4473e-01, -1.3598e-01,  2.4274e-01,
         -7.0045e-02, -3.3975e-01, -4.1309e-02]]), 'left_hand_pose': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'right_hand_pose': tensor([[[ 4.7179e-03, -8.7367e-02, -1.0163e-03],
         [ 1.1773e-05,  5.5882e-05,  4.9258e-04],
         [-5.5836e-04, -1.1276e-04,  1.1149e-01],
         [-3.3485e-02, -7.3819e-02,  4.7320e-01],
         [-5.6208e-04, -1.2960e-04,  3.0116e-03],
         [-1.2382e-04,  2.3551e-04,  7.0223e-04],
         [-6.0145e-01,  8.0739e-02,  1.0365e+00],
         [-1.9556e-01, -5.6917e-05,  2.6644e-01],
         [-7.7230e-02, -2.3220e-04,  1.1817e-01],
         [-1.3852e-01,  8.0554e-02,  4.7806e-01],
         [-1.2343e-01, -2.1414e-04,  6.8536e-01],
         [-2.2280e-01, -1.4877e-04,  6.8685e-01],
         [ 9.1217e-01,  1.6390e-01,  2.8463e-01],
         [-4.1091e-02, -3.8960e-02, -3.2564e-02],
         [ 4.9512e-02, -9.1085e-02,  4.1448e-02]]]), 'leye_pose': tensor([[0., 0., 0.]]), 'reye_pose': tensor([[0., 0., 0.]]), 'betas': tensor([[ 0.8514, -0.1162, -0.8182, -0.2834,  0.5147, -0.2417, -0.5456, -0.6966,
          0.3408, -0.5830, -1.3159, -0.0545,  0.4412, -1.1093, -0.1157, -0.1568,
         -0.1397,  0.3679, -0.4785, -0.0439,  0.9008, -0.0961, -0.0219, -0.1494,
         -1.4848,  0.2330, -0.0792, -0.3516,  0.5593, -1.1831, -0.2976, -0.4300,
          0.2402, -1.0620, -0.0573, -0.5361,  0.0560, -0.1494, -0.0048, -0.2134,
          1.0656,  0.7949,  1.3386,  0.1027,  0.2455, -0.4114,  0.9072,  0.6007,
          0.0421,  0.6405, -0.1330,  1.4939,  0.0934, -1.6345,  0.0333,  1.2651,
         -0.5780, -1.8516, -1.1917,  0.7125, -1.3109, -0.8392, -0.9361,  0.9458,
          0.8204,  0.2252, -1.2758,  0.7517,  0.2797,  0.3090, -0.0119, -1.8464,
         -1.4342,  0.5670,  1.1845, -1.8772, -0.2411,  0.0127, -1.6683, -0.4382,
         -0.2888,  0.6981, -0.2171,  1.2973, -0.1111,  0.2316, -0.0474,  0.6557,
         -0.2193,  0.6510,  0.0613, -0.0446,  2.4966, -0.5414,  0.1352, -1.1519,
         -0.6063, -1.7786, -0.9366,  0.3243]])}
'''

BODY_JOINT_NAMES = [
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

FINGER_JOINT_NAMES = [
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',
]

import copy

def mirror_rot(rotvec):
    # 镜像轴角 x 轴，实现轴的左右镜像 [-x,y,z]
    # 再xyz全部取反，实现旋转方向的镜像 [x,-y,-z]

    rotvec=np.array([rotvec[0],-rotvec[1],-rotvec[2]])
    return rotvec

def mirror_smplx_data(smplx_frame_d):
    # 镜像根位置
    global_pos = smplx_frame_d['transl'][0]
    global_pos[:] = torch.tensor([-global_pos[0],global_pos[1],global_pos[2]])

    # 镜像根旋转
    global_rot = smplx_frame_d['global_orient'][0]
    global_rot[:] = torch.tensor(mirror_rot(global_rot))

    # 镜像身体姿态
    body_pose=smplx_frame_d['body_pose'][0]
    mir_body_pose = copy.deepcopy(body_pose)

    for i, name in enumerate(BODY_JOINT_NAMES):
        if name.startswith('left'):
            mir_name = name.replace('left', 'right',1)
            mir_idx = BODY_JOINT_NAMES.index(mir_name)
            # 获取对称关节的原始旋转
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            # 镜像对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        elif name.startswith('right'):
            mir_name = name.replace('right', 'left',1)
            mir_idx = BODY_JOINT_NAMES.index(mir_name)
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        else:
            # 镜像非对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(body_pose[i*3:(i+1)*3]))
    body_pose[:]=mir_body_pose[:]

    # 镜像左右手
    left_hand_pose = smplx_frame_d['left_hand_pose'][0]
    right_hand_pose = smplx_frame_d['right_hand_pose'][0].view(-1)
    mir_left_hand_pose = copy.deepcopy(left_hand_pose)
    mir_right_hand_pose = copy.deepcopy(right_hand_pose)

    for i, name in enumerate(FINGER_JOINT_NAMES):
        mir_left_hand_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(right_hand_pose[i*3:(i+1)*3]))
        mir_right_hand_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(left_hand_pose[i*3:(i+1)*3]))

    left_hand_pose[:]=mir_left_hand_pose[:]
    right_hand_pose[:]=mir_right_hand_pose[:]
    smplx_frame_d['right_hand_pose'][0][:,:]=right_hand_pose.reshape([15,3])[:,:]

# 镜像Tensor数据
mirror_smplx_data(smplx_frame_d)

# 镜像结果构建mesh
smplx_results = smplx_model(return_verts=True, **smplx_frame_d)
c = collection.getOrNewCollection("smplx")
verts = smplx_results.vertices.detach().cpu().numpy()[0]
faces = smplx_model.faces
obj = mesh.createMesh("2", vertices=verts, faces=faces, matrix=rosita_2023_08.coord_matrix,collection=c)