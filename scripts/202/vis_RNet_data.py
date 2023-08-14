import torch
import numpy as np
import  pickle
import os
import open3d as o3d
import time
from psbody.mesh import MeshViewers, Mesh
import time
from rosita.process_data.utils import to_tensor
import copy

model_path = '/mnt/public/users/lixin/zenan/data/assets/smplx'
from smplx import SMPLXLayer
female_model = SMPLXLayer(
    model_path=model_path,
    gender='female',
    num_pca_comps=45,
    flat_hand_mean=True,
)
sbj_m = female_model
face = sbj_m.faces

# path='/mnt/public/users/lixin/zenan/workdata/rosita_workdata/exp/MNet4rosita_infer_A002_s20_02/MNet_inference_result/html/1/A002-2023-0419-1400-37-task-0-seq-2-cylinder_bottle_s390.pkl'
# path='/mnt/public/users/lixin/zenan/workdata/rosita_workdata/tmp/output/smplx_min_s20/RNet_data4rosita/train/A002-2023-0504-1829-32-task-503-seq-36.npy'
path='rosita_workdata/tmp/output/smplx_50seqs_s20/RNet_data4rosita/train/A002-2023-0419-1400-37-task-0-seq-2.npy'
result = np.load(path,allow_pickle=True)

# obj_info_path='/mnt/public/users/lixin/zenan/workdata/rosita_workdata/tmp/output/smplx_min_s20/RNet_data4rosita/obj_info.npy'
obj_info_path='/mnt/public/users/lixin/zenan/workdata/rosita_workdata/tmp/output/smplx_min_s20/RNet_data4rosita/obj_info.npy'
obj_info = np.load(obj_info_path,allow_pickle=True).item()
for k,v in obj_info.items():##只含有一个obj
    # print(k) ##cylinder_bottle_s230
    v_template_obj=obj_info[k]['verts']

scene = o3d.io.read_triangle_mesh("assets/scene/scene_cleaned_new.obj", True)
scene_v_template=copy.deepcopy(np.asarray(scene.vertices))
TABLE_HEIGHT=np.array([0.0,0.84,0.0]).reshape(-1,3)

ch_num=len(result['transl'])
body_mesh=None
body_pc=None
obj_pc=None

for i in range(ch_num):
    print('ch num =', i)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=786)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])##x、y、z 轴将分别为红色、绿色和蓝色箭头。
    vis.add_geometry(coordinate_frame)
    
    ##每个ch中人体中心不太一样，所以桌子有点位置不同
    rel_trans=result['rel_trans'][i].reshape(-1,3)
    scene_vertices=scene_v_template-rel_trans+TABLE_HEIGHT
    scene.vertices = o3d.utility.Vector3dVector(scene_vertices)
    vis.add_geometry(scene)
    
    for j in range(22):
        print(j)
        verts_sbj= result['verts'][i][j]
        
        transl_obj= result['transl_obj'][i][j]
        orient_rotmat_obj= result['global_orient_rotmat_obj'][i][j]
        verts_obj = np.asarray(torch.matmul(to_tensor(v_template_obj),to_tensor(orient_rotmat_obj)) + to_tensor(transl_obj))

        # if body_mesh is None:
            #     body_mesh = o3d.geometry.TriangleMesh()
            #     body_mesh.vertices = o3d.utility.Vector3dVector(verts_sbj)
            #     body_mesh.triangles = o3d.utility.Vector3iVector(face)
            #     body_mesh.paint_uniform_color([0.9, 0.7, 0.7])
            #     vis.add_geometry(body_mesh)
            # else:
            #     body_mesh.vertices = o3d.utility.Vector3dVector(verts_sbj)
            #     vis.update_geometry(body_mesh)
            
        if body_pc is None:
            body_pc = o3d.geometry.PointCloud()
            body_pc.points = o3d.utility.Vector3dVector(verts_sbj)
            body_pc.colors = o3d.utility.Vector3dVector(np.array([[0.9, 0.7, 0.7]] * verts_sbj.shape[0]))
            vis.add_geometry(body_pc)
        else:
            body_pc.points = o3d.utility.Vector3dVector(verts_sbj)
            vis.update_geometry(body_pc)
                       
        if obj_pc is None:
            obj_pc = o3d.geometry.PointCloud()
            obj_pc.points = o3d.utility.Vector3dVector(verts_obj)
            obj_pc.colors = o3d.utility.Vector3dVector(np.array([[0., 0., 0.2]] * verts_obj.shape[0])) ##black gt
            vis.add_geometry(obj_pc)
        else:
            obj_pc.points = o3d.utility.Vector3dVector(verts_obj)
            vis.update_geometry(obj_pc)
              
        vis.poll_events()
        vis.update_renderer()
        if j>20:
            time.sleep(2)
        if i>10:
            time.sleep(0.2)
        else:            
            time.sleep(0.05)

    vis.destroy_window()
    del vis
    body_pc=None
    obj_pc=None
    
print("end")