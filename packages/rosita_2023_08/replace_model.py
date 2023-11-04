import os
from smplx_blender import utils


REPLACE_DIR = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1")


def get_replace_data(scene_name, coord):
    import glob
    replace_dir = os.path.join(utils.DATA_PATH,r"favor_preview\tmp\favor_pass1",scene_name)
    replace_paths = glob.glob(os.path.join(replace_dir,"*.npz"))

    result={}

    for path in replace_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        rep_from = name.split("-",1)[0]
        rep_to = name.split("-",1)[1]

        replace = np.load(path)
        trs = replace["arr_0"]

        result[rep_from] = {
            'rep_to'    : rep_to,
            'trs'       : trs
        }
    return result

def load_replaced_object(name, replace_data, *, collection=None):
    import bpy
    import mathutils
    from math import radians

    matrix=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    affine_mat = utils.getAffineMat(matrix)
    trs = mathutils.Matrix(replace_data[name]['trs'])
    affine=mathutils.Matrix(affine_mat)

    from_obj = bpy.data.objects["indicator_"+name]
    to_obj_src = bpy.data.objects[replace_data[name]['rep_to']]
    to_obj = to_obj_src.copy()
    
    bpy.context.scene.collection.objects.link(to_obj)
    offset = mathutils.Matrix.Rotation(radians(90),4,'X')

    to_obj.matrix_world = affine@trs@affine.inverted()@offset
    return to_obj