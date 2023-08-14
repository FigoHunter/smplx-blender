import torch
import numpy as np
import pickle
import os
import open3d as o3d
import time
from psbody.mesh import MeshViewers, Mesh
import time
from visualization.task import Task
import json
import trimesh
from smplx.lbs import batch_rigid_transform, batch_rodrigues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parms_6D2full(pose, trans, d62rot=True):
    bs = trans.shape[0]

    pose = pose.reshape([bs, -1, 3, 3])

    body_parms = full2bone(pose, trans)
    body_parms["fullpose_rotmat"] = pose

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
        "global_orient": global_orient,
        "body_pose": body_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "transl": trans,
    }
    return body_parms


# def d62rotmat(pose):
#     pose = to_tensor(pose)
#     reshaped_input = pose.reshape(-1, 6)
#     return t3d.rotation_6d_to_matrix(reshaped_input)


model_path = "body_utils/body_models/smplx"
from smplx import SMPLXLayer

female_model = SMPLXLayer(
    model_path=model_path,
    gender="female",
    num_pca_comps=45,
    flat_hand_mean=True,
)
sbj_m = female_model.to(device)
face = sbj_m.faces


path = "tmp/RNet_inference_result/pkl/1/A002-2023-0419-1400-37-task-81-seq-78-cylinder_bottle_s298.pkl"
result = np.load(path, allow_pickle=True)

pp = path.split("/")[-1].split("-")[:9]
task_path = "tmp/vg_data/grasp/" + f"{pp[0]}_{'-'.join(pp[1:5])}/" + "_".join(pp[5:]) + "/task_fixed.json"
task = Task.from_dict(json.load(open(task_path, "r")))

print(task.object_mesh.obj_path)
obj_mesh = trimesh.load(task.object_mesh.obj_path, process=False, force="mesh")

obj_rot = result["obj_params_pre"]["global_orient_rotmat"][0].cpu().numpy()
obj_tsl = result["obj_params_pre"]["transl"][0].cpu().numpy()

batch_size = result["sbj_params_pre"]["left_hand_pose"].shape[0]
device = result["sbj_params_pre"]["global_orient"].device
full_pose = torch.cat(
    [
        result["sbj_params_pre"]["global_orient"].reshape(-1, 1, 3, 3),
        result["sbj_params_pre"]["body_pose"].reshape(-1, 21, 3, 3),
        torch.zeros(batch_size, 1, 3, 3).to(device),
        result["sbj_params_pre"]["leye_pose"].reshape(-1, 1, 3, 3),
        result["sbj_params_pre"]["reye_pose"].reshape(-1, 1, 3, 3),
        result["sbj_params_pre"]["left_hand_pose"].reshape(-1, 15, 3, 3),
        result["sbj_params_pre"]["right_hand_pose"].reshape(-1, 15, 3, 3),
    ],
    dim=1,
)

full_rotmats = full_pose.view([1, -1, 3, 3])
_, A = batch_rigid_transform(
    full_rotmats, torch.zeros(batch_size, 55, 3).to(device), sbj_m.parents, dtype=torch.float32
)
hand_global = A[:, 21].cpu().numpy()  # T_wh, [B, 4, 4]

obj_transf = np.eye(4)  # T_wo [4, 4]
obj_transf[:3, :3] = obj_rot.T
obj_transf[:3, 3] = obj_tsl


from termcolor import cprint


# hand_global_rot = torch.tensor(manipul_pose[int(frame.split(".")[0])]["h_pose"][0].reshape(-1), device=device)
# hand_global_rot = axis_angle_to_matrix(hand_global_rot).view(3, 3)
# hand_global_rot = gt_hand_pose[:, 0]
# hand_global_rot = axis_angle_to_matrix(hand_global_rot).view(self.batch_size, 3, 3)

# # hand_local = A[:, 19, :3, :3].transpose(-1, -2) @ hand_global_rot  # 19: elbow joint
# hand_global_rot = axis_angle_to_matrix(self.right_hand_pose[:, :3]) @ hand_global_rot
# hand_local = A[:, 19, :3, :3].transpose(-1, -2) @ hand_global_rot  # 19: elbow joint
# hand_local = matrix_to_axis_angle(hand_local)  # [B, 3]

# body_param["body_pose"] = torch.cat([body_param["body_pose"][:, :-3], hand_local], dim=1)

batch = result["batch_gt"]

# result["sbj_params_pre"]["transl"] = torch.zeros_like(result["sbj_params_pre"]["transl"])
result["sbj_params_pre"]["right_hand_pose"] = result["sbj_params_pre"]["right_hand_pose"][[-1]].repeat(
    batch_size, 1, 1, 1
)

# body_params_gt = parms_6D2full(batch["fullpose_rotmat"][0], batch["transl"][0], d62rot=False)
# body_params_gt["betas"] = torch.repeat_interleave(batch["betas"].clone(), 22, dim=0).reshape(-1, 10).to(device)
sbj_output_gt = sbj_m(**result["sbj_params_pre"])
v_gt = sbj_output_gt.vertices.reshape(-1, 10475, 3)  ##(22, 10475, 3)

j_gt = sbj_output_gt.joints.reshape(batch_size, -1, 3)[:, 21]  # [B, 3]
hand_global[:, :3, 3] = j_gt.cpu().detach().numpy()  # [B, 4, 4]


delta_pose = np.linalg.inv(hand_global[-1]) @ obj_transf  # T_ho , [4, 4]


tmp_face = sbj_m.faces
tmp_obj_verts = None
body_verts = np.asarray(v_gt[0].cpu().detach())

vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=640)

manipul_obj = o3d.geometry.TriangleMesh()
manipul_obj.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
manipul_obj.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
manipul_obj.compute_vertex_normals()
vis.add_geometry(manipul_obj)

manipul_body = o3d.geometry.TriangleMesh()
manipul_body.vertices = o3d.utility.Vector3dVector(body_verts)
manipul_body.triangles = o3d.utility.Vector3iVector(tmp_face)
manipul_body.compute_vertex_normals()
vis.add_geometry(manipul_body)

from visualization.transform import aa_to_rotmat, rotmat_to_aa

obj_global_rot = []
obj_global_aa = []
obj_tsl = []
for k in range(sbj_output_gt.vertices.shape[0]):
    new_transf = hand_global[k] @ delta_pose
    obj_global_rot.append(new_transf[:3, :3].T)
    obj_global_aa.append(rotmat_to_aa(new_transf[:3, :3].T))
    obj_tsl.append(new_transf[:3, 3])
obj_global_rot = np.stack(obj_global_rot)
obj_global_aa = np.stack(obj_global_aa)
obj_tsl = np.stack(obj_tsl)

result["obj_params_pre"]["global_orient_rotmat"] = torch.tensor(obj_global_rot, device=device)
result["obj_params_pre"]["global_orient"] = torch.tensor(obj_global_aa, device=device)
result["obj_params_pre"]["transl"] = torch.tensor(obj_tsl, device=device)


# relative tsl
relative_tsl = result["batch_gt"]["rel_trans"]
# scene += relative_tsl


# dump
# path = "path to dump"
# with open(path, "wb") as f:
#     pickle.dump(result, f)


for k in range(sbj_output_gt.vertices.shape[0]):
    body_verts = np.asarray(v_gt[k].cpu().detach())
    manipul_body.vertices = o3d.utility.Vector3dVector(body_verts)
    manipul_body.triangles = o3d.utility.Vector3iVector(tmp_face)
    manipul_body.compute_vertex_normals()
    vis.update_geometry(manipul_body)
    new_transf = hand_global[k] @ delta_pose
    manipul_obj.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices @ new_transf[:3, :3].T + new_transf[:3, 3])
    manipul_obj.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
    manipul_obj.compute_vertex_normals()
    vis.update_geometry(manipul_obj)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)

vis.run()


# from scripts.vis_inet import open3d_show

# open3d_show(
#     obj_verts=tmp_obj_verts,
#     obj_normals=None,
#     contact_info=None,
#     hand_verts=tmp_hand_verts,
#     hand_faces=tmp_face,
#     show_hand_normals=False,
# )
