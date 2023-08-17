import bpy

def removeObject(object_to_delete):
    bpy.data.objects.remove(object_to_delete, do_unlink=True)