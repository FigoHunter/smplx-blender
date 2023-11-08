import bpy
import os
from bpy.props import EnumProperty, CollectionProperty
from . import props
from blender_figo import utils
from blender_figo import file as File
from blender_figo import collection as Collection
from blender_figo import objects as Objects
import mathutils

class LoadObjs(bpy.types.Operator):
    bl_idname = "omniobject.loadobjs"
    bl_label = "Load All Objects"
    bl_options = {'REGISTER'}

    # def divide_chunks(self, l, n):
    #     # looping till length l
    #     for i in range(0, len(l), n): 
    #         yield l[i:i + n]

    def __walk(self, path):
        for root, dirs, files in os.walk(path, followlinks=True):
            for file in files:
                file = os.path.join(root, file)
                if file.endswith('.obj'):
                    yield file

    @classmethod
    def poll(cls, context):
        try:
            return os.path.exists(context.scene.omniobject_props.dataPath) and os.path.isdir(context.scene.omniobject_props.dataPath)
        except: return False

    def execute(self, context):
        items=[("None", "None","")]
        path = os.path.join(context.scene.omniobject_props.dataPath,'decimated')
        for f in self.__walk(path):
            data = os.path.abspath(f)
            name=os.path.relpath(data, path)
            items.append((name, name[2:],""))
        props.setObjFileList(items,context)
        context.scene.omniobject_props.objFile="None"
        return {'FINISHED'}


    @classmethod
    def menu_func(cls, menu, context):
        menu.layout.operator(cls.bl_idname)

class AddObj(bpy.types.Operator):
    bl_idname = "omniobject.addobj"
    bl_label = "Add Objects To Scene"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            selected = context.scene.omniobject_props.objFile
            path = os.path.join(context.scene.omniobject_props.dataPath, 'decimated', selected)
            return os.path.exists(path)
        except: return False


    def execute(self, context):
        import numpy as np
        from math import radians
        import random
        from . import align

        selected = context.scene.omniobject_props.objFile
        path = os.path.join(context.scene.omniobject_props.dataPath, 'decimated', selected)
        print(path)
        if os.path.exists(path):
            matrix=np.array([[1,0,0],[0,0,-1],[0,1,0]])
            affine = mathutils.Matrix(utils.getAffineMat(matrix))

            c = Collection.getOrNewCollection('OmniObject3d')
            File.importFile(path)
            objects = bpy.context.selected_objects
            name = path.replace('\\','/').split('/')[-3]
            if len(objects) >1:
                obj = Objects.createEmpty(name,c)
                for o in objects:
                    Collection.moveCollection(o,c)
                    o.parent = obj
                offset = mathutils.Matrix.Identity(4)
            elif len(objects) == 1:
                obj=bpy.context.selected_objects[0]
                obj.name = name
                obj.data.name = name
                Collection.moveCollection(obj, c)
                offset = mathutils.Matrix.Rotation(radians(90),4,'X')
            props.addLoadedList(context, selected, obj)

            align_data = align.load_align_data(os.path.dirname(path))
            if align_data is not None and len(align_data) > 0:
                print(align_data)
                align_piece=align_data[random.randint(0,len(align_data)-1)]
                ground = align_piece['extent']['min'][1]
                trs = mathutils.Matrix.Translation([0,-ground,0])@mathutils.Matrix(align_piece['matrix'])
                scale = mathutils.Matrix.Scale(0.001,4)
                obj.matrix_world= scale@affine@trs@affine.inverted()@offset
            else:
                obj.matrix_world = mathutils.Matrix.Scale(0.001,4)@offset
        else:
            print(f'[ERROR] {path} does not exist')
        return {'FINISHED'}
    
    @classmethod
    def menu_func(cls, menu, context):
        menu.layout.operator(cls.bl_idname)


class SelectObj(bpy.types.Operator):

    obj_list : bpy.props.EnumProperty(items=props.getObjFileList)

    """Tooltip"""
    bl_idname = "omniobject.selectobj"
    bl_label = "Select Obj"
    bl_options = {'REGISTER', 'UNDO'}
    bl_property = "obj_list"


    def execute(self, context):
        context.scene.omniobject_props.objFile = self.obj_list
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.invoke_search_popup(self)
        return {'FINISHED'}

class RandonmizeObj(bpy.types.Operator):
    bl_idname = "omniobject.randomobj"
    bl_label = "Random Object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import random

        obj_list = props.getObjFileList(context.scene,context)
        index = random.randint(0,len(obj_list)-1)
        selected = obj_list[index][0]
        context.scene.omniobject_props.objFile = selected
        return {'FINISHED'}


class LoadRawScanMesh(bpy.types.Operator):
    bl_idname = "omniobject.loadrawscanmesh"
    bl_label = "Load Raw Scan Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props.validateLoadedList(context)
        props.validateRawScanList(context)
        loaded = props.getLoadedList(context)
        raw_scans = props.getRawScanList(context)
        print(loaded)
        print(raw_scans)
        for rel_path in loaded:
            if rel_path in raw_scans:
                continue
            path = os.path.join(context.scene.omniobject_props.dataPath, 'raw_scans', rel_path)
            print(path)
            if os.path.exists(path):
                c = Collection.getOrNewCollection('OmniRawScan')
                File.importFile(path)
                objects = bpy.context.selected_objects
                name = path.replace('\\','/').split('/')[-3]+'_raw_scan'
                if len(objects) > 1:
                    obj = Objects.createEmpty(name,c)
                    for o in objects:
                        Collection.moveCollection(o,c)
                        o.parent = obj
                elif len(objects) == 1:
                    obj=bpy.context.selected_objects[0]
                    obj.name = name
                    obj.data.name = name
                    Collection.moveCollection(obj, c)
                props.addRawScanList(context, rel_path, obj)

                decimated_obj=bpy.data.objects[loaded[rel_path]]
                if decimated_obj is not None:
                    matrix = decimated_obj.matrix_world
                    obj.matrix_world = matrix
            else:
                print(f'[ERROR] {path} does not exist')
        return {'FINISHED'}
    
    def menu_func(self, context):
        self.layout.operator(LoadRawScanMesh.bl_idname)