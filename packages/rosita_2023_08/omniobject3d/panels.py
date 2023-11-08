import bpy
from . import props
from . import utils

class OmniObject_PT_LoadObjects(bpy.types.Panel):
    bl_label = "Load Obj Models"
    bl_category = "OmniObject"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"



    def draw(self, context):
        layout = self.layout
        layout.label(text='Dataset Path:')
        layout.prop(context.scene.omniobject_props,"dataPath")

        layout.operator("omniobject.loadobjs", text="Reload Objects")
        layout.separator()
        layout.label(text="Select Objects:")

        layout.operator("omniobject.selectobj",text=context.scene.omniobject_props.objFile)
        layout.operator("omniobject.randomobj",text="Randomize")
        layout.operator("omniobject.addobj", text="Add")

