#    Addon info
bl_info = {
    "name": "MC_29",
    "author": "Rich Colburn, email: the3dadvantage@gmail.com",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Extended Tools > Modeling Cloth",
    "description": "It's like cloth but in a computer!",
    "warning": "3D models of face masks will not protect your computer from viruses",
    "wiki_url": "",
    "category": '3D View'}

if "bpy" in locals():
    import imp
    imp.reload(MC_self_collision)
    imp.reload(MC_object_collision)
    imp.reload(MC_pierce)
    imp.reload(MC_29)
    #imp.reload(ModelingCloth)
    #imp.reload(SurfaceFollow)
    #imp.reload(UVShape)
    #imp.reload(DynamicTensionMap)
    print("Reloaded Modeling Cloth")
else:
    from . import MC_self_collision#, SurfaceFollow, UVShape, DynamicTensionMap
    from . import MC_object_collision#, SurfaceFollow, UVShape, DynamicTensionMap
    from . import MC_pierce
    from . import MC_29#, SurfaceFollow, UVShape, DynamicTensionMap
    print("Imported Modeling Cloth")

   
def register():
    MC_self_collision.register()
    MC_object_collision.register()
    MC_pierce.register()
    MC_29.register()
    #ModelingCloth.register()    
    #SurfaceFollow.register()
    #UVShape.register()
    #DynamicTensionMap.register()

    
def unregister():
    MC_self_collision.unregister()
    MC_object_collision.unregister()
    MC_pierce.unregister()
    MC_29.unregister()
    #ModelingCloth.unregister()
    #SurfaceFollow.unregister()
    #UVShape.unregister()
    #DynamicTensionMap.unregister()

    
if __name__ == "__main__":
    register()
