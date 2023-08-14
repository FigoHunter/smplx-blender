import bpy

def openBlendFile(filepath):
    bpy.ops.wm.open_mainfile(filepath=filepath)

def linkBlendFile(filepath,collections = False, objects = False):
    with bpy.data.libraries.load(filepath,link=True) as (data_from, data_to):
        if collections:
            data_to.collections = data_from.collections
        if objects:
            data_to.objects = data_from.objects

    result = {"collections":[],"objects":[]}

    if collections:
        for new_coll in data_to.collections:
            instance = bpy.data.objects.new(new_coll.name, None)
            instance.instance_type = 'COLLECTION'
            instance.instance_collection = new_coll
            result["collections"].append(instance)

    if objects:
        result["objects"] = list(data_to.objects)
    
    return result