import bpy
from tqdm import tqdm

def __loadIndicatorMaterials(): 
    mats=[]
    for mat in bpy.data.materials:
        if mat.name.startswith('indicator'):
            mats.append(mat)
    assert len(mats)>0, "材质不存在"
    return mats

def __loadBodyMaterials(): 
    mats=[]
    for mat in bpy.data.materials:
        if mat.name.startswith('body'):
            mats.append(mat)
    assert len(mats)>0, "材质不存在"
    return mats

def __loadApproachBodyMaterials(): 
    mats=[]
    for mat in bpy.data.materials:
        if mat.name.startswith('approach_body'):
            mats.append(mat)
    assert len(mats)>0, "材质不存在"
    return mats

def __loadGrabBodyMaterials(): 
    mats=[]
    for mat in bpy.data.materials:
        if mat.name.startswith('grab_body'):
            mats.append(mat)
    assert len(mats)>0, "材质不存在"
    return mats

def __loadManipMaterials(): 
    mats=[]
    for mat in bpy.data.materials:
        if mat.name.startswith('manip'):
            mats.append(mat)
    assert len(mats)>0, "材质不存在"
    return mats

def assign_indicator_materials(objects):
    mats = __loadIndicatorMaterials()
    if len(objects)<1:
        return
    elif len(objects)<2:
        objects[0].active_material=mats[0]
        return
    for i, o in tqdm(enumerate(objects)):
        mat = mats[i%len(mats)]
        o.active_material = mat

def assign_body_materials(objects):
    mats = __loadBodyMaterials()
    if len(objects)<1:
        return
    elif len(objects)<2:
        objects[0].active_material=mats[0]
        return
    for i, o in tqdm(enumerate(objects)):
        mat_index = (int(i/(len(objects)-1)*(len(mats)-1))+1)%len(mats)
        mat = mats[mat_index]
        o.active_material = mat

def assign_approach_body_materials(objects):
    mats = __loadApproachBodyMaterials()
    if len(objects)<1:
        return
    elif len(objects)<2:
        objects[0].active_material=mats[0]
        return
    for i, o in tqdm(enumerate(objects)):
        mat_index = (int(i/(len(objects)-1)*(len(mats)-1))+1)%len(mats)
        mat = mats[mat_index]
        o.active_material = mat

def assign_grab_body_materials(objects):
    mats = __loadGrabBodyMaterials()
    if len(objects)<1:
        return
    elif len(objects)<2:
        objects[0].active_material=mats[0]
        return
    for i, o in tqdm(enumerate(objects)):
        mat_index = (int(i/(len(objects)-1)*(len(mats)-1))+1)%len(mats)
        mat = mats[mat_index]
        o.active_material = mat

def assign_manip_materials(objects):
    mats = __loadManipMaterials()
    if len(objects)<1:
        return
    elif len(objects)<2:
        objects[0].active_material=mats[0]
        return
    for i, o in tqdm(enumerate(objects)):
        mat_index = (int(i/(len(objects)-1)*(len(mats)-1))+1)%len(mats)
        mat = mats[mat_index]
        o.active_material = mat


class IndicatorMaterialsOperator(bpy.types.Operator):
    bl_idname = "rosita.indicator_materials"
    bl_label = "Assign Indicator Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        objects = context.selected_objects
        print(objects)
        assign_indicator_materials(objects)
        return {'FINISHED'}
    
    def menu_func(self, context):
        self.layout.operator(IndicatorMaterialsOperator.bl_idname)

class BodyMaterialsOperator(bpy.types.Operator):
    bl_idname = "rosita.body_materials"
    bl_label = "Assign Body Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        objects = context.selected_objects
        print(objects)
        assign_body_materials(objects)
        return {'FINISHED'}
    
    def menu_func(self, context):
        self.layout.operator(BodyMaterialsOperator.bl_idname)

class ManipMaterialsOperator(bpy.types.Operator):
    bl_idname = "rosita.manip_materials"
    bl_label = "Assign Manip Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        objects = context.selected_objects
        print(objects)
        assign_manip_materials(objects)
        return {'FINISHED'}
    
    def menu_func(self, context):
        self.layout.operator(ManipMaterialsOperator.bl_idname)