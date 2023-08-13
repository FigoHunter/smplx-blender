import numpy as np
import os
import pickle
from tqdm import tqdm
from smplx_blender import mesh,utils
from importlib import reload
import mathutils


reload(mesh)
reload(utils)

path=os.path.join(utils.DATA_PATH,"zenan","A002-2023-0419-1400-37-task-5-seq-5-cylinder_bottle_s341_dumped.pkl")
skip=1
start_frame=0
matrix=np.array([[-1,0,0],[0,0,1],[0,1,0]])
affine_mat = utils.getAffineMat(matrix)

dumped = pickle.load(open(path, "rb"))
verts = dumped["verts"]
faces = dumped["faces"]

manip_verts = dumped["manip_verts"]
manip_trs = dumped["manip_trs"]


for fid,vertices in tqdm(enumerate(verts[start_frame::skip])):
    name = str(fid)
    mesh.createMesh(name, vertices=vertices, faces=faces, matrix=matrix)

    manip = mesh.createMesh("manip",vertices=manip_verts,matrix=matrix)
    trs = manip_trs[fid * skip + start_frame]
    manip.matrix_world=mathutils.Matrix(np.matmul(np.matmul(affine_mat,trs), np.linalg.inv(affine_mat)))
