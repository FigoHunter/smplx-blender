
def loadGroundTruthBody(data, smplx_model, frames=None, mat=None, matrix=None):
    import torch
    from smplx_blender import mesh,utils

    smplx_data = data["smplx"]
    if frames == None:
        frames=[*range(len(smplx_data))]

    if isinstance(frames, int):
        frames=[frames]
        single_frame=True
    else:
        single_frame=False
        objs=[]

    for fid in frames:
        smplx_frame_d = smplx_data[fid]
        smplx_frame_d = {k: torch.tensor(v).to("cpu")[None] for k, v in smplx_frame_d.items()}
        smplx_frame_d["betas"] = torch.tensor(data["betas"][None]).to("cpu")
        smplx_results = smplx_model(return_verts=True, **smplx_frame_d)

        verts = smplx_results.vertices.detach().cpu().numpy()[0]
        faces = smplx_model.faces
        name="smplx_" + str(fid)
        obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=mat, matrix=matrix)
        if single_frame:
            return obj
        else:
            objs.append(obj)
    return objs

def loadGroundTruthManip(data, frames=None, mat=None, matrix=None):
    import mathutils
    from smplx_blender import mesh,utils

    trs_list = data["obj_pose"]
    verts = data["fixed_obj_mesh"]["verts"]
    faces = data["fixed_obj_mesh"]["faces"]

    if frames == None:
        frames=[*range(len(trs_list))]

    if isinstance(frames, int):
        frames=[frames]
        single_frame=True
    else:
        single_frame=False
        objs=[]
    affine_coord=mathutils.Matrix(utils.getAffineMat(matrix))
    for fid in frames:
        name="manip_" + str(fid)
        obj = mesh.createMesh(name, vertices=verts, faces=faces, mat=mat, matrix=matrix)
        trs = trs_list[fid]
        trs = mathutils.Matrix(trs_list[fid])
        obj.matrix_world = affine_coord@trs@affine_coord.inverted()

        if single_frame:
            return obj
        else:
            objs.append(obj)
    return objs