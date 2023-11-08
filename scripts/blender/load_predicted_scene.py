import bpy
import mathutils
import os
from smplx_blender import utils,mesh
import rosita_2023_08.load
import numpy as np
from rosita_2023_08.ops import rosita_materials 
import re
from blender_figo import collection

table_height = 0.84

def load_indicators(scene_name):
    gt_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")
    gt_target = re.findall(r'\S+-[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{2}-task-[0-9]+-seq-[0-9]+',scene_name)[0]   
    favor_data = rosita_2023_08.load.loadGroundTruthData(os.path.join(gt_dir,gt_target+".pkl"))
    coll = collection.getOrNewCollection('Indicators')
    indicators = rosita_2023_08.load.loadGroundTruthIndicator(favor_data,matrix=rosita_2023_08.coord_matrix,collection=coll)
    rosita_materials.assign_indicator_materials(indicators)

def load_mnet_data(scene_name):
    mnet_path = os.path.join(utils.DATA_PATH, "mnet_extracted", scene_name)
    extracted = rosita_2023_08.load.loadMnetData(mnet_path)
    start_frame = 0
    body_verts_seq=extracted["body"]["verts"]
    body_faces = extracted["body"]["faces"]
    manip_verts=extracted["manip_obj"]["verts"][0]
    manip_faces=extracted["manip_obj"]["faces"]

    offset = extracted["offset"]@rosita_2023_08.coord_matrix
    offset = (offset[0], -offset[1], -table_height)

    # manips=[]
    bodies=[]

    skip = int(len(body_verts_seq)/2)-1
    frames = [*range(start_frame,len(body_verts_seq),skip)]

    print(f'mnet frames: {frames}')
    # coll_mnet_manip=collection.getOrNewCollection('Mnet_Manip')
    # name="mnet_manip"
    # obj=mesh.createMesh(name,vertices=manip_verts,faces=manip_faces, matrix=coord,collection=collec)
    # obj.location = -offset
    # manips.append(obj)

    coll_mnet_smplx = collection.getOrNewCollection('Mnet_Smplx')

    for fid in frames:
        body_verts = body_verts_seq[fid]
        name="mnet_smplx_" + str(fid)
        obj = mesh.createMesh(name,vertices=body_verts,faces=body_faces, matrix=rosita_2023_08.coord_matrix, collection=coll_mnet_smplx)
        obj.location = offset
        bodies.append(obj)

    rosita_materials.assign_approach_body_materials(bodies)


def load_rnet_data(scene_name):
    rnet_path = os.path.join(utils.DATA_PATH, "rnet_extracted", scene_name)
    extracted = rosita_2023_08.load.loadRnetData(rnet_path)
    start_frame = 0

    body_verts_seq=extracted["body"]["verts"]
    body_faces = extracted["body"]["faces"]
    manip_verts_seq=extracted["manip_obj"]["verts"]
    manip_faces=extracted["manip_obj"]["faces"]

    offset = extracted["offset"]@rosita_2023_08.coord_matrix
    offset = (offset[0], -offset[1], -table_height)

    skip = int(len(body_verts_seq)/2)-1
    frames = [*range(start_frame,len(body_verts_seq),skip)]

    print(f'rnet frames: {frames}')

    bodies=[]
    manips=[]

    coll_rnet_smplx = collection.getOrNewCollection('Rnet_Smplx')
    coll_rnet_manip = collection.getOrNewCollection('Rnet_Manip')


    for fid in frames:
        body_verts = body_verts_seq[fid]
        manip_verts = manip_verts_seq[fid]

        name="rnet_smplx_" + str(fid)
        obj = mesh.createMesh(name,vertices=body_verts,faces=body_faces, matrix=rosita_2023_08.coord_matrix, collection=coll_rnet_smplx)
        obj.location = offset
        bodies.append(obj)
 
        name="rnet_manip_" + str(fid)
        obj=mesh.createMesh(name,vertices=manip_verts,faces=manip_faces, matrix=rosita_2023_08.coord_matrix, collection=coll_rnet_manip)
        obj.location = offset
        manips.append(obj)

    rosita_materials.assign_grab_body_materials(bodies)
    rosita_materials.assign_manip_materials(manips)

def load_predicted_scene(scene_name):
    collection.removeCollection('Mnet_Smplx')
    collection.removeCollection('Rnet_Smplx')
    collection.removeCollection('Rnet_Manip')
    collection.removeCollection('Indicators')

    load_indicators(scene_name)
    load_mnet_data(scene_name)
    load_rnet_data(scene_name)

if __name__=='__main__':
    target="A002-2023-0419-1400-37-task-0-seq-2-cylinder_bottle_s390.pkl"
    load_predicted_scene(target)