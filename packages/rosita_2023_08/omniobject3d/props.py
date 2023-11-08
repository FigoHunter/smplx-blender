import bpy
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty )
from bpy.types import ( PropertyGroup )
import pickle
import codecs


objList=[("None", "None","")]

def getObjFileList(scene, context):
    global objList
    if str(context.scene.omniobject_props.objListJson):
        objList=pickle.loads(codecs.decode(context.scene.omniobject_props.objListJson.encode(), "base64"))
    return objList

def setObjFileList(list, context):
    global objList
    objList = list
    context.scene.omniobject_props.objListJson=codecs.encode(pickle.dumps(objList), "base64").decode()

def getLoadedList(context):
    return deserializeProp(context.scene.omniobject_props.loadedList, default={})

def addLoadedList(context, name, obj):
    ls = getLoadedList(context)
    ls[name] = obj.name
    ls = __validateLoadedList(ls)
    serializeProp(context.scene.omniobject_props.loadedList, ls)

def validateLoadedList(context):
    ls = getLoadedList(context)
    ls = __validateLoadedList(ls)
    serializeProp(context.scene.omniobject_props.loadedList, ls)


def getRawScanList(context):
    return deserializeProp(context.scene.omniobject_props.rawScanList, default={})

def addRawScanList(context, name, obj):
    ls = getRawScanList(context)
    ls[name] = obj.name
    ls = __validateLoadedList(ls)
    serializeProp(context.scene.omniobject_props.rawScanList, ls)

def validateRawScanList(context):
    ls = getRawScanList(context)
    ls = __validateLoadedList(ls)
    serializeProp(context.scene.omniobject_props.rawScanList, ls)

def __validateLoadedList(ls):
    for name in ls:
        obj_name = ls[name]
        if obj_name not in bpy.data.objects:
            ls.remove(name)
    return ls

def deserializeProp(prop, default=None):
    print(str(prop))
    if str(prop):
        data=pickle.loads(codecs.decode(prop.encode(), "base64"))
        print(data)
    else:
        data=default
        print('default')
    return data

def serializeProp(prop, value):
    prop = codecs.encode(pickle.dumps(value), "base64").decode()


class OmniObjectProperties(PropertyGroup):

    objFile : EnumProperty(
        name = "",
        description = "Obj File",
        items = getObjFileList
    )

    dataPath : StringProperty(
        name = "",
        description = "Dataset Path",
        default = "",
        subtype = 'DIR_PATH'
        )
    
    objListJson:StringProperty(
        name = "",
        description = "Obj List",
        default = "",
    )

    loadedList:StringProperty(
        name = "",
        description = "Loaded List",
        default = "",
    )


    rawScanList:StringProperty(
        name = "",
        description = "Raw Scan List",
        default = "",
    )