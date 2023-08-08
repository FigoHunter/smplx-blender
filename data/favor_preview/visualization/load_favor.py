import argparse
import os
import pickle


import numpy as np
import open3d as o3d
import smplx
import torch

from tqdm import tqdm

from visualization.task import Task


def get_body_model(type, gender, batch_size, device="cuda", v_template=None):
    """
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    """
    # ? you can download these from https://smpl-x.is.tue.mpg.de/index.html
    # ? please follow the instruction in SAGA https://github.com/JiahaoPlus/SAGA
    body_model_path = "./body_utils/body_models"
    body_model = smplx.create(
        body_model_path,
        model_type=type,
        gender=gender,
        ext="npz",
        use_pca=False,
        num_betas=100,
        num_pca_comps=45,
        flat_hand_mean=True,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=batch_size,
        v_template=v_template,
    )
    if device == "cuda":
        return body_model.cuda()
    else:
        return body_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="grabpose-Testing")

    parser.add_argument(
        "--dir",
        type=str,
    )
    parser.add_argument(
        "--data",
        default="./tmp/fitting/A001-2023-0511-1940-05-task-343-seq-32",
        type=str,
        help="The path to the folder that contains grabpose data",
    )
    parser.add_argument("--skip", default=4, type=int, help="skip frame")
    parser.add_argument("--dump_favor", action="store_true", help="dump pose to file")
    args = parser.parse_args()

    assert not args.dump_favor or args.skip == 1, "dump_favor only works with skip=1"

    if args.dir is not None:
        pathes = os.listdir(args.dir)
        pathes = [os.path.join(args.dir, p) for p in pathes]
        pathes.sort(key=lambda x: x.split("/")[-1])
    else:
        pathes = [args.data]

    device = "cpu"
    gender = "female"

    bm = get_body_model("smplx", str(gender), 1, device=device)

    import random

    random.shuffle(pathes)

    for path in pathes:
        print(path)

        seq_name = os.path.basename(path)

        favor_data = pickle.load(open(path, "rb"))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=640)

        scene = o3d.io.read_triangle_mesh("./assets/table_only.obj", True)
        scene.compute_vertex_normals()
        vis.add_geometry(scene)

        try:
            cam_extr = np.loadtxt("./assets/cam_extr.txt")
        except:
            cam_extr = None

        body_mesh = None
        hand_mesh = None

        length = favor_data["length"]
        task = Task.from_dict(favor_data["task"])
        obj_pose = favor_data["obj_pose"]
        obj_mesh = favor_data["obj_mesh"]
        fixed_obj_mesh = favor_data["fixed_obj_mesh"]
        moving_frame = favor_data["moving_frame"]
        subject_name = favor_data["subject_name"]
        indicator_meshes = favor_data["indicator_meshes"]
        smplx_data = favor_data["smplx"]

        for i_mesh in indicator_meshes:
            i_mesh_o3d = o3d.geometry.TriangleMesh()
            i_mesh_o3d.vertices = o3d.utility.Vector3dVector(i_mesh["verts"])
            i_mesh_o3d.triangles = o3d.utility.Vector3iVector(i_mesh["faces"])
            i_mesh_o3d.compute_vertex_normals()
            vis.add_geometry(i_mesh_o3d)
        manipul_obj = o3d.geometry.TriangleMesh()
        manipul_obj.vertices = o3d.utility.Vector3dVector(fixed_obj_mesh["verts"])
        manipul_obj.triangles = o3d.utility.Vector3iVector(fixed_obj_mesh["faces"])
        manipul_obj.compute_vertex_normals()
        vis.add_geometry(manipul_obj)

        for fid, smplx_frame_d in tqdm(enumerate(smplx_data[:: args.skip])):
            smplx_frame_d = {k: torch.tensor(v).to(device)[None] for k, v in smplx_frame_d.items()}
            smplx_frame_d["betas"] = torch.tensor(favor_data["betas"][None]).to(device)

            smplx_results = bm(return_verts=True, **smplx_frame_d)

            verts = smplx_results.vertices.detach().cpu().numpy()
            face = bm.faces

            manipul_obj.transform(obj_pose[fid * args.skip])

            if body_mesh is None:
                body_mesh = o3d.geometry.TriangleMesh()
                body_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
                body_mesh.triangles = o3d.utility.Vector3iVector(face)
                body_mesh.compute_vertex_normals()
                body_mesh.paint_uniform_color([0.9, 0.7, 0.7])
                vis.add_geometry(body_mesh)

                view_control = vis.get_view_control()
                camera = view_control.convert_to_pinhole_camera_parameters()
                if cam_extr is not None:
                    camera.extrinsic = cam_extr
                    view_control.convert_from_pinhole_camera_parameters(camera)
            else:
                body_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
                body_mesh.triangles = o3d.utility.Vector3iVector(face)
                body_mesh.compute_vertex_normals()
            vis.update_geometry(body_mesh)
            vis.update_geometry(manipul_obj)

            vis.poll_events()
            vis.update_renderer()
            manipul_obj.transform(np.linalg.inv(obj_pose[fid * args.skip]))

        vis.destroy_window()
        del view_control
        del vis
