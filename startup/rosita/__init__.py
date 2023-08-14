import bpy

from smplx_blender.ops.rosita_materials import IndicatorMaterialsOperator,BodyMaterialsOperator,ManipMaterialsOperator


bpy.utils.register_class(IndicatorMaterialsOperator)
bpy.types.VIEW3D_MT_object.append(IndicatorMaterialsOperator.menu_func)

bpy.utils.register_class(BodyMaterialsOperator)
bpy.types.VIEW3D_MT_object.append(BodyMaterialsOperator.menu_func)

bpy.utils.register_class(ManipMaterialsOperator)
bpy.types.VIEW3D_MT_object.append(ManipMaterialsOperator.menu_func)