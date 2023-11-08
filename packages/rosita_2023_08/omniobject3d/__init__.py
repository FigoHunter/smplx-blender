import os
from smplx_blender import utils
from . import props,ops,panels

DATASET_PATH = os.path.join(utils.DATA_PATH,"omniobj3d")
print(f'asdfas: {DATASET_PATH}')

def register():
    import bpy
    from bpy.props import PointerProperty,CollectionProperty
    bpy.utils.register_class(props.OmniObjectProperties)
    bpy.types.Scene.omniobject_props = PointerProperty(type=props.OmniObjectProperties)

    bpy.utils.register_class(panels.OmniObject_PT_LoadObjects)

    bpy.utils.register_class(ops.LoadObjs)
    bpy.utils.register_class(ops.AddObj)
    bpy.utils.register_class(ops.SelectObj)
    bpy.utils.register_class(ops.RandonmizeObj)

    bpy.utils.register_class(ops.LoadRawScanMesh)
    bpy.types.VIEW3D_MT_object.append(ops.LoadRawScanMesh.menu_func)

    def collhack(scene):
        bpy.app.handlers.depsgraph_update_pre.remove(collhack) 
        if not str(scene.omniobject_props.dataPath):
            print("SetPath")
            scene.omniobject_props.dataPath = DATASET_PATH
        print(str(scene.omniobject_props.dataPath))

    bpy.app.handlers.depsgraph_update_pre.append(collhack)
