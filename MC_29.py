# when extruding:
#   currently have to switch to the source shape
#   This is bceause the coords will be in the same location
#       in the source shape. Could pause existing selection
#       along with new selection. Maybe play with source being
#       relative only while that's happening then switch it back
#       to the current being relative to source?
#   To confirm maybe wait until deselct all or wait a given
#       number of iterations?

# cache keyframing could be really cool. So do a linear
# interpolation between cached frames.

# !!!!!!!!!!!!!!!
# mesh keyframing might actually be what I need for
# my procedural animation stuff.
# think about the barycentric springs with equalateral tris applied to an armature
# !!!!!!!!!!!!!!!

# would it make sense to run cloth physics on a lattice or curve object?????

# undo !!!!!!
#def b_log(a=None,b=None):
    #return

# !!! OPTIMISE: lots of placed to provide and "out" array. Start with all the einsums.
# !!! might also check the speed of linalg against v / sqrt(einsum(vec vec))
# !!! np.nan_to_num(  copy=False) # or use an output array

# !!! bend and stretch stiff itters should be set using
#   int div so like bend stiff 2 means 2 iters. below one means stifness value is less than
#   one. Might even be able to turn up stiffness with higher iters and stay stable. 

# add a friction vertex group to collide objects and to cloth.

# !!! when popping out of edit mode all of refresh runs.
#   there is a bug if I don't do that but I need to know what
#   parts of the refresh really need to run

# can I pre-allocate memory before fancy indexing like for tri_coors?
#   tri_co[:] = cloth.co[tridex] ?? Would it be faster??

# !!! bug where start spring length changes if cloth is
#   turned on in edit mode and mesh is distorted. Should
#   be getting spring lengths from source shape. Somehow not.
#   changes were made to def measure_edges(

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


try:
    import bpy
    from bpy.ops import op_as_string
    from bpy.app.handlers import persistent
    import os
    import shutil
    import pathlib
    import inspect
    import bmesh
    import functools as funky
    import numpy as np
    from numpy import newaxis as nax
    import time
    import copy # for duplicate cloth objects


except ImportError:
    print("didn't import correctly")
    print("didn't import correctly")
    print("didn't import correctly")
    pass

try:
    from . import MC_self_collision
    from . import MC_object_collision
    from . import MC_pierce
except:
    MC_object_collision = bpy.data.texts['MC_object_collision.py'].as_module()
    MC_self_collision = bpy.data.texts['MC_self_collision.py'].as_module()
    MC_pierce = bpy.data.texts['MC_pierce.py'].as_module()
    print("Tried to import internal texts.")


# global data
MC_data = {}
MC_data['col_data'] = {'col': False, 'col_update':False}
MC_data['cloths'] = {}
MC_data['iterator'] = 0

# recent_object allows cloth object in ui
#   when selecting empties such as for pinning.
MC_data['recent_object'] = None


big_t = 0.0
def rt_(num=None, skip=True, show=False):
    return
    if not show:
        return
    global big_t
    #if skip:
        #return
    t = time.time()
    if num is not None:    
        print(t - big_t, "timer", num)
    big_t = t


# developer functions ------------------------
def reload():
    """!! for development !! Resets everything"""
    # when this is registered as an addon I will want
    #   to recaluclate these objects not set prop to false
    
    # Set all props to False:
    reload_props = ['continuous', 'collider', 'animated', 'cloth', 'cache', 'cache_only', 'play_cache']
    if 'MC_props' in dir(bpy.types.Object):
        for i in reload_props:
            for ob in bpy.data.objects:
                ob.MC_props[i] = False

    for i in bpy.data.objects:
        if "MC_cloth_id" in i:
            del(i["MC_cloth_id"])
        if "MC_collider_id" in i:
            del(i["MC_collider_id"])

    #for detecting deleted colliders or cloths
    MC_data['cloth_count'] = 0
    MC_data['collider_count'] = 0
    
    try:
        soft_unregister()    
    except:
        print("failed to run soft_unregister")    


def np_co_to_text(ob, co, rw='w', cloth=None):
    """Read or write cache file"""
    name = ob.name + 'cache.npy'

    if rw == 'w':
        if name not in bpy.data.texts:
            bpy.data.texts.new(name)

        txt = bpy.data.texts[name]
        np.savetxt(txt, co)

        return

    vc = len(ob.data.vertices)
    txt = bpy.data.texts[name].as_string()
    frame = bpy.context.scene.frame_current
    start = (frame -1) * vc * 3

    co = np.fromstring(txt, sep='\n')[start: start + (vc * 3)]
    co.shape = (co.shape[0]//3, 3)

    # right here could put in a feature to overwrite in edit mode per frame.
    if ob.data.is_editmode:
        for i, v in enumerate(cloth.obm.verts):
            v.co = co[i]
        bmesh.update_edit_mesh(ob.data)
        return
            
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
    ob.data.update()


# developer functions ------------------------
def get_proxy_co(ob, co=None, proxy=None, return_proxy=False):
    """Gets co with modifiers like cloth"""
    if proxy is None:

        #dg = bpy.context.evaluated_depsgraph_get()
        dg = glob_dg
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    if return_proxy:
        return co, proxy, prox
    
    return co


def get_sc_normals(co, ob):
    """Updates the proxy with current cloth.co
    before getting vertex normals"""
    
    dg = glob_dg
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    proxy.vertices.foreach_set('co', co.ravel())
    proxy.update()
            
    normals = np.zeros((len(proxy.vertices), 3), dtype=np.float32)
    proxy.vertices.foreach_get('normal', normals.ravel())

    return normals



def get_proxy_normals(ob=None, proxy=None):
    if proxy is None:

        dg = glob_dg
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()
        
    normals = np.zeros((len(proxy.vertices), 3), dtype=np.float32)
    proxy.vertices.foreach_get('normal', normals.ravel())

    return normals
    

# universal ---------------------
def absolute_co(ob, co=None):
    """Get vert coords in world space"""
    co, proxy, prox = get_proxy_co(ob, co, return_proxy=True)
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc, proxy, prox


def get_proxy(ob):
    """Gets proxy mesh with mods"""
    # use: ob.to_mesh_clear() to remove
    #dg = bpy.context.evaluated_depsgraph_get()
    dg = glob_dg
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    return proxy


# developer functions ------------------------
def read_python_script(name=None):
    """When this runs it makes a copy of this script
    and saves it to the blend file as a text
    Not a virus... well sort of like a virus"""

    p_ = pathlib.Path(inspect.getfile(inspect.currentframe()))
    py = p_.parts[-1]
    p = p_.parent.joinpath(py)
    try:
        o = open(p)
    except:
        p = p_.parent.joinpath(py) # linux or p1 (not sure why this is happening in p1)
        o = open(p)

    if name is None:
        name = 'new_' + py

    new = bpy.data.texts.new(name)

    r = o.read()
    new.write(r)


# developer functions ------------------------
def create_object_cache(ob):
    print('cache only function')


#print("new--------------------------------------")


# Cache functions ---------------
def cache_interpolation(cloth):
    # !!! have to finish this one
    # Might as well throw in the ability to scale the
    #   cache up and down.
    #   Compute on playback so we don't screw with the files

    ob = cloth.ob
    f = ob.MC_props.current_cache_frame
    fp = cloth.cache_dir
    idx = np.array([int(i.parts[-1]) for i in fp.iterdir()])


# Cache functions ---------------
def cache_only(ob, frame=None):

    self = ob.MC_props

    path = os.path.expanduser("~/Desktop")
    self['cache_folder'] = path
    mc_path = pathlib.Path(path).joinpath('MC_cache_files')

    if not mc_path.exists():
        mc_path.mkdir()

    final_path = mc_path.joinpath(ob.name)
    self['cache_name'] = ob.name
    final_path = mc_path.joinpath(ob.name)

    # create dir if it doesn't exist
    if not final_path.exists():
        final_path.mkdir()

    #f = bpy.context.scene.frame_current
    f = ob.MC_props.current_cache_frame
    ob.MC_props['current_cache_frame'] = f + 1
    if frame is not None:
        f = frame

    txt = final_path.joinpath(str(f))

    np.savetxt(txt, get_proxy_co(ob))


# Cache functions ---------------
def cache(cloth, keying=False):
    """Store a text file of Nx3 numpy coords."""
    ob = cloth.ob
    fp = cloth.cache_dir

    maf = ob.MC_props.max_frames
    con = ob.MC_props.continuous

    # update ccf either to current frame or to +1 if continuous
    ccf = ob.MC_props.current_cache_frame
    f = bpy.context.scene.frame_current
    ob.MC_props['current_cache_frame'] = f

    if keying:
        f = ccf
        ob.MC_props['current_cache_frame'] = ccf

    if con:
        f = ccf
        ob.MC_props['current_cache_frame'] = ccf + 1

    txt = fp.joinpath(str(f))

    #sf = cloth.ob.MC_props.start_frame
    #ef = cloth.ob.MC_props.end_frame
    #if (f >= sf) & (f <= ef):

    nonexistent = not txt.exists()

    if (nonexistent | ob.MC_props.overwrite_cache):
        np.savetxt(txt, cloth.co)
        print('saved a cache file: ', txt)


# Cache functions ---------------
def play_cache(cloth, cb=False):
    """Load a text file of Nx3 numpy coords."""
    if cloth.ob.MC_props.internal_cache:
        np_co_to_text(cloth.ob, cloth.co, rw='r', cloth=cloth)
        return
        print('play cache is running')

    ob = cloth.ob

    if not hasattr(cloth, "cache_dir"):
        ob.MC_props['play_cache'] = False
        return

    cache_interpolation(cloth) # Finish this !!!

    f = bpy.context.scene.frame_current

    if ob.MC_props.continuous:
        f = ob.MC_props.current_cache_frame
        ob.MC_props['current_cache_frame'] = f + 1

    if cb: # when running the callback
        f = ob.MC_props.current_cache_frame

    fp = cloth.cache_dir

    txt = fp.joinpath(str(f))

    if txt.exists():
        cloth.co = np.loadtxt(txt, dtype=np.float32)

    key = 'MC_current'
    # cache only playback
    if ob.MC_props.cache_only:
        key = 'cache_key'
        if not ob.data.is_editmode:
            ob.data.shape_keys.key_blocks['cache_key'].data.foreach_set('co', cloth.co.ravel())
            ob.data.update()
            return

    if ob.data.is_editmode:
        index = ob.data.shape_keys.key_blocks.find(key)
        if ob.active_shape_key_index == index:

            try:
                cloth.obm.verts
            except:

                cloth.obm = bmesh.from_edit_mesh(ob.data)

            for i, j in enumerate(cloth.co):
                cloth.obm.verts[i].co = j
        return

    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co.ravel())
    ob.data.update()


# debugging
def T(type=1, message=''):
    if type == 1:
        return time.time()
    print(time.time() - type, message)


# ============================================================ #
#                    universal functions                       #
#                                                              #


# universal ---------------------
def get_bary_weights(tris, points):
    """Find barycentric weights for triangles.
    Tris is a Nx3x3 set of triangle coords.
    points is the same N in Nx3 coords"""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, None]
    v2 = points - origins

    # ---------
    v0 = cross_vecs[:,0]
    v1 = cross_vecs[:,1]

    d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = d00_d11[:,0]
    d11 = d00_d11[:,1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    weights = np.array([1 - (u+v), u, v, ]).T
    return weights


# universal ---------------------
def update_ob_v_norms(cloth, force=False):
    """updates cloth.v_norms"""
    
    if not force:
        pass
        # check if last co is different
        # for skipping the calc if not needed        
        
    tri_co = cloth.total_co[cloth.oc_total_tridex]
    normals = get_normals_from_tris(tri_co)

    # now get vertex normals with add.at
    cloth.ob_v_norms[:] = 0.0
    np.add.at(cloth.ob_v_norms, cloth.ob_v_norm_indexer, normals[cloth.ob_v_norm_indexer1])
    #print(normals)
    dots = np.sqrt(np.einsum('ij,ij->i', cloth.ob_v_norms, cloth.ob_v_norms))[:, None]
    cloth.ob_v_norms /= dots
    
    
# universal ---------------------
def update_v_norms(cloth):
    """updates cloth.v_norms"""
        
    tri_co = cloth.co[cloth.tridex]
    normals = get_normals_from_tris(tri_co)
    cloth.tri_normals = normals
    # now get vertex normals with add.at
    cloth.v_norms[:] = 0.0
    np.add.at(cloth.v_norms, cloth.v_norm_indexer, normals[cloth.v_norm_indexer1])
    dots = np.sqrt(np.einsum('ij,ij->i', cloth.v_norms, cloth.v_norms))[:, None]
    cloth.v_norms /= dots


# universal ---------------------
def get_normals_from_tris(tris):
    """Returns unit normals from tri coords"""
    origins = tris[:, 0]
    vecs = tris[:, 1:] - origins[:, None]
    cross = np.cross(vecs[:, 0], vecs[:, 1])
    mag = np.sqrt(np.einsum("ij ,ij->i", cross, cross))[:, nax]
    return cross/mag


# universal ---------------------
def cross_from_tris(tris):
    origins = tris[:, 0]
    vecs = tris[:, 1:] - origins[:, nax]
    cross = np.cross(vecs[:, 0], vecs[:, 1])
    return cross


# universal ---------------------
def apply_rotation(object, normals):
    """When applying vectors such as normals we only need
    to rotate"""
    m = np.array(object.matrix_world)
    mat = m[:3, :3].T
    #object.v_normals = object.v_normals @ mat
    return normals @ mat


# universal ---------------------
def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate. Forces need to be scaled"""
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] / np.array(ob.scale, dtype=np.float32) # rotates backwards without T
    return (co @ mat) / np.array(ob.scale, dtype=np.float32)


# universal ---------------------
def revert_transforms(ob, co):
    """Set world coords on object.
    Run before setting coords to deal with object transforms
    if using apply_transforms()"""
    m = np.linalg.inv(ob.matrix_world)
    mat = m[:3, :3]# rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


# universal ---------------------
def revert_in_place(ob, co):
    """Revert world coords to object coords in place."""
    m = np.linalg.inv(ob.matrix_world)
    mat = m[:3, :3]# rotates backwards without T
    loc = m[:3, 3]
    co[:] = co @ mat + loc


# universal ---------------------
def apply_in_place(ob, arr):
    """Overwrite vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    # optimize: np.dot(arr, mat, out=dot_arr)
    # optimize: np.add(dot_arr, loc, out=add_arr)
    arr[:] = arr @ mat + loc
    return arr


# universal ---------------------
def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def invert_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world.inverted(), dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


# universal ---------------------
def copy_object_transforms(ob1, ob2):
    """Put the transforms of one object onto another"""
    M = ob1.matrix_world.copy()
    ob2.matrix_world = M


# universal ---------------------
def manage_locations_as_strings(old, new):
    # vertex coords for each point in old to a key as string
    """ !!! not finished !!! """
    idx_co_key = {}
    vs = ob.data.vertices
    flat_faces = np.hstack(sel_faces)
    for i in range(len(flat_faces)):
        idx_co_key[str(vs[flat_faces[i]].co)] = flat_faces[i]


# universal ---------------------
def offset_face_indices(faces=[]):
    """Sorts the original face vert indices
    for a new mesh from subset. Works with N-gons"""
    # Example: face_verts = [[20, 10, 30], [10, 30, 100, 105]]
    # Converts to [[1, 0, 2], [0, 2, 3, 4]]

    def add(c):
        c['a'] += 1
        return c['a']

    flat = np.hstack(faces)
    idx = np.unique(flat, return_inverse=True)[1]
    c = {'a': -1}
    new_idx = [[idx[add(c)] for j in i] for i in faces]
    return new_idx


# universal ---------------
def link_mesh(verts, edges, faces, name='name'):
    """Generate and link a new object from pydata"""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    mesh_ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob


# universal ---------------
def mesh_from_selection(ob, name='name'):
    """Generate a new mesh from selected faces"""
    obm = get_bmesh(ob)
    obm.faces.ensure_lookup_table()
    faces = [[v.index for v in f.verts] for f in obm.faces if f.select]
    if len(faces) == 0:
        print("No selected faces in mesh_from_selection")
        return # no faces

    obm.verts.ensure_lookup_table()
    verts = [i.co for i in obm.verts if i.select]
    idx_faces = offset_face_indices(faces)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], idx_faces)
    mesh.update()
    if False:
        mesh_ob = ob.copy()
    else:
        mesh_ob = bpy.data.objects.new(name, mesh)

    mesh_ob.data = mesh
    mesh_ob.name = name
    bpy.context.collection.objects.link(mesh_ob)
    return mesh_ob, faces, idx_faces


# universal ---------------
def get_bmesh(ob=None, refresh=False):
    """gets bmesh in editmode or object mode
    by checking the mode"""
    if ob.data.is_editmode:
        return bmesh.from_edit_mesh(ob.data)
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    if refresh:
        obm.verts.ensure_lookup_table()
        obm.edges.ensure_lookup_table()
        obm.faces.ensure_lookup_table()
    return obm


# universal ---------------------
def get_quad_obm(ob):
    """Get a triangulated mesh as quads. Used by
    the bend springs for greater stability"""
    ob.update_from_editmode()
    obm = bmesh.new()
    obm.from_mesh(ob.data)
    bmesh.ops.join_triangles(obm, faces=obm.faces, angle_shape_threshold=0.9, angle_face_threshold=0.9)
    return obm


# universal ---------------
def get_co(ob, ar=None):
    """Get vertex coordinates from an object in object mode"""
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    ob.data.vertices.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def co_overwrite(ob, ar):
    """Fast way to overwrite"""
    ob.data.vertices.foreach_get('co', ar.ravel())


# universal ---------------
def get_co_shape(ob, key=None, ar=None):
    """Get vertex coords from a shape key"""
    if ar is not None:
        ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
        return ar
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------------
def get_poly_centers(ob, co, data=None):
    """Get poly centers. Data is meant
    to be built the first time then
    passed in. (dynamic)"""

    if data is not None:
        data[0][:] = 0
        np.add.at(data[0], data[2], co[data[3]])
        data[0] /= data[1]
        return data[0]

    pc = len(ob.data.polygons)
    pidex = np.hstack([[v for v in p.vertices] for p in ob.data.polygons])

    div = [len(p.vertices) for p in ob.data.polygons]

    indexer = []
    for i, j in enumerate(div):
        indexer += [i] * j
    div = np.array(div, dtype=np.float32)[:, None]

    centers = np.zeros((pc, 3), dtype=np.float32)

    np.add.at(centers, indexer, co[pidex])
    centers /= div

    return [centers, div, indexer, pidex]


# universal ---------------------
def get_poly_centers_bmesh(obm, co, data=None):
    """Get poly centers. Data is meant
    to be built the first time then
    passed in. (dynamic)"""

    if data is not None:
        data[0][:] = 0
        np.add.at(data[0], data[2], co[data[3]])
        data[0] /= data[1]
        return data[0]

    pc = len(obm.faces)
    pidex = np.hstack([[v.index for v in f.verts] for f in obm.faces])

    div = [len(f.verts) for f in obm.faces]

    indexer = []
    for i, j in enumerate(div):
        indexer += [i] * j
    div = np.array(div, dtype=np.float32)[:, None]

    centers = np.zeros((pc, 3), dtype=np.float32)

    np.add.at(centers, indexer, co[pidex])
    centers /= div

    return [centers, div, indexer, pidex]


# universal ---------------
def key_overwrite(ob, ar, key):
    """Fast way to overwrite"""
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())


# universal ---------------
def get_co_edit__(ob, obm=None):
    """Get vertex coordinates from an object in edit mode"""
    if obm is None:
        obm = get_bmesh(ob)
        obm.verts.ensure_lookup_table()
    co = np.array([i.co for i in obm.verts])
    return co


# universal ---------------
def Nx3(ob):
    """For generating a 3d vector array"""
    if ob.data.is_editmode:
        ob.update_from_editmode()
    count = len(ob.data.vertices)
    ar = np.zeros((count, 3), dtype=np.float32)
    return ar


# universal ---------------
def get_co_edit(ob, ar=None, key='MC_current'):
    if ar is None:
        c = len(ob.data.vertices)
        ar = np.empty((c, 3), dtype=np.float32)
    ob.update_from_editmode()
    #ob.data.vertices.foreach_get('co', ar.ravel())
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def get_co_mode(ob=None, key='MC_current'):
    """Edit or object mode"""
    if ob is None: # cloth.target object might be None
        return
    if ob.data.is_editmode:
        ob.update_from_editmode()
    c = len(ob.data.vertices)
    ar = np.empty((c, 3), dtype=np.float32)
    #ob.data.vertices.foreach_get('co', ar.ravel())
    ob.data.shape_keys.key_blocks[key].data.foreach_get('co', ar.ravel())
    return ar


# universal ---------------
def compare_geometry(ob1, ob2, obm1=None, obm2=None, all=False):
    """Check for differences in verts, edges, and faces between two objects"""
    # if all is false we're comparing a target objects. verts in faces
    #   and faces must match.
    def get_counts(obm):
        v_count = len([v for v in obm.verts if len(v.link_faces) > 0])
        #v_count = len(obm.verts) # not sure if I need to separate verts...
        e_count = len(obm.edges)
        f_count = len(obm.faces)
        if all:
            return np.array([v_count, e_count, f_count])
        return np.array([v_count, f_count]) # we can still add sew edges in theory...

    if obm1 is None:
        obm1 = get_bmesh(ob1)
    if obm2 is None:
        obm2 = get_bmesh(ob2)

    c1 = get_counts(obm1)
    c2 = get_counts(obm2)

    return np.all(c1 == c2)


# universal ---------------
def detect_changes(counts, obm):
    """Compare mesh data to detect changes in edit mode"""
    # counts in an np array shape (3,)
    v_count = len(obm.verts)
    e_count = len(obm.edges)
    f_count = len(obm.faces)
    new_counts = np.array([v_count, e_count, f_count])
    # Return True if all are the same
    return np.all(counts == new_counts), f_count < 1


# universal ---------------
def get_mesh_counts(ob, obm=None):
    """Returns information about object mesh in edit or object mode"""
    if obm is not None:
        v_count = len(obm.verts)
        e_count = len(obm.edges)
        f_count = len(obm.faces)
        return np.array([v_count, e_count, f_count])
    v_count = len(ob.data.vertices)
    e_count = len(ob.data.edges)
    f_count = len(ob.data.polygons)
    return np.array([v_count, e_count, f_count])


# universal ---------------
def get_weights(ob, name, obm=None, default=0, verts=None, weights=None):
    """Get vertex weights. If no weight is assigned
    set array value to zero. If default is 1 set
    all values to 1"""
    if obm is None:
        obm = get_bmesh(ob, refresh=True)

    # might want to look into using map()
    count = len(obm.verts)
    if name not in ob.vertex_groups:
        ob.vertex_groups.new(name=name)

    g_idx = ob.vertex_groups[name].index
    arr = np.zeros(count, dtype=np.float32)

    obm.verts.layers.deform.verify()

    deform = obm.verts.layers.deform.active

    dvert_lay = obm.verts.layers.deform.active

    if verts is not None:
        for i, v in enumerate(verts):
            dvert = obm.verts[v][dvert_lay]
            dvert[g_idx] = weights[i]
        return # will run again once the weights are set and build the arrays.

    if dvert_lay is None: # if there are no assigned weights
        return arr

    for v in obm.verts:
        idx = v.index
        dvert = v[dvert_lay]

        if g_idx in dvert:
            arr[idx] = dvert[g_idx]
        else:
            if default == 1:
                dvert[g_idx] = default
            arr[idx] = default

    return arr


# universal !!! broken !!! ---------------------
def box_bary_weights(poly, point, vals=[]):
    """Get values to plot points from tris
    or return the new plotted value."""
    # note: could use a lot of add.at/subtract.at and
    #   obscenely complex indexing to do bend springs
    #   between n-gons. It would be a riot!

    if vals: # two scalar values
        ba = poly[1] - poly[0]
        ca = poly[2] - poly[0]
        v1, v2 = vals[0], vals[1]
        plot = poly[0] + (((ba * v1) + (ca * v2)) * .5)
        #plot = poly[0]# + (((ba * v1) + (ca * v2)) * .5)
        return plot

    pa = point - poly[0]
    ba = poly[1] - poly[0]
    ca = poly[2] - poly[0]

    v1 = np.nan_to_num(pa / ba)
    v2 = np.nan_to_num(pa / ca)
    return [v1, v2]


# cloth setup -------------
def pairs_idx(ar):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns ar (array) and the index that removes the duplicates."""
    # no idea how this works (probably sorcery) but it's really fast
    a = np.sort(ar, axis=1) # because it only sorts on the second acess the index still matches other arrays.
    x = np.random.rand(a.shape[1])
    #x = np.linspace(1, 2, num=a.shape[1])
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index



# cloth setup -------------
def reset_shapes(ob):
    """Create shape keys if they are missing"""

    if ob.data.shape_keys == None:
        ob.shape_key_add(name='Basis')

    keys = ob.data.shape_keys.key_blocks
    if 'MC_source' not in keys:
        ob.shape_key_add(name='MC_source')
        keys['MC_source'].value = 1.0

    if 'MC_current' not in keys:
        ob.shape_key_add(name='MC_current')
        keys['MC_current'].value = 1.0
        keys['MC_current'].relative_key = keys['MC_source']


# cloth setup -------------
def extend_bend_springs():
    #is there a way to get bend relationships from
    #just the basis springs...
    #Or could I do something like virtual springs
    #but for bend sets...
    pass


# abstract bend setup ----------------------------
# precalculated ------------------------
def get_j_surface_offset(cloth):
    """Get the vecs to move the plotted
    wieghts off the surface."""

    ax = cloth.j_axis_vecs
    ce = cloth.j_ce_vecs # has the faces swapped so the normal corresponds to the other side of the axis
    cross = np.cross(ax, ce)

    cloth.j_normals = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]
    cloth.plot_normals = cloth.j_normals[cloth.j_tiler]

    cloth.bend_flat = False
    cloth.plot_vecs = cloth.sco[cloth.swap_jpv] - cloth.j_plot
    cloth.plot_dots = np.einsum('ij,ij->i', cloth.plot_normals, cloth.plot_vecs)[:, None]
    if np.all(cloth.plot_dots < 0.000001):
        cloth.bend_flat = True


# abstract bend setup ----------------------------
# dynamic ------------------------------
def measure_linear_bend(cloth):
    """Takes a set of coords and an edge idx and measures segments"""
    l = cloth.sp_ls # left side of the springs (Full moved takes the place of the right side)
    a,b,c = np.unique(l, return_counts=True, return_inverse=True)

    x = c[b]
    cloth.bend_multiplier = ((x - 2) / 2) + 2
    return

    v = (cloth.full_moved - cloth.co[cloth.sp_ls]) / cloth.divy
    d = np.einsum("ij ,ij->i", v, v)
    return v, d, np.sqrt(d)


# abstract bend setup ----------------------------
# dynamic ------------------------------
def get_eq_tri_tips(cloth, co, centers, skip=False):
    """Slide the centers of each face along
    the axis until it's in the middle for
    using as a triangle. (dynamic)"""

    skip = True # set to false to use eq tris.
    if skip: # skip will test if it really makes any difference to move the tris to the center
        cloth.j_axis_vecs = co[cloth.stacked_edv[:,1]] - co[cloth.stacked_edv[:,0]]
        cloth.j_tips = centers[cloth.stacked_faces]
        cloth.j_ce_vecs = centers[cloth.stacked_faces] - co[cloth.stacked_edv[:,0]]
        return cloth.j_tips, cloth.j_axis_vecs, cloth.j_ce_vecs

    # creates tris from center and middle of edge.
    # Not sure if it makes any difference...
    j_axis_vecs = co[cloth.stacked_edv[:,1]] - co[cloth.stacked_edv[:,0]]
    j_axis_dots = np.einsum('ij,ij->i', j_axis_vecs, j_axis_vecs)
    j_ce_vecs = centers[cloth.stacked_faces] - co[cloth.stacked_edv[:,0]]
    cloth.swap_ce_vecs = centers[cloth.swap_faces] - co[cloth.stacked_edv[:,0]]
    j_cea_dots = np.einsum('ij,ij->i', j_axis_vecs, j_ce_vecs)

    j_div = j_cea_dots / j_axis_dots
    j_spit = j_axis_vecs * j_div[:,None]

    j_cpoe = co[cloth.stacked_edv[:,0]] + j_spit
    jt1 = centers[cloth.stacked_faces] - j_cpoe
    j_mid = co[cloth.stacked_edv[:,0]] + (j_axis_vecs * 0.5)

    cloth.j_tips = j_mid + jt1
    cloth.j_axis_vecs = j_axis_vecs
    cloth.j_ce_vecs = j_ce_vecs
    # ---------------------
    return cloth.j_tips, cloth.j_axis_vecs, cloth.j_ce_vecs


# abstract bend setup ----------------------------
# precalculated ------------------------
def eq_bend_data(cloth):
    """Generates face pairs around axis edges.
    Supports edges with 2-N connected faces.
    Can use internal structures this way."""
    ob = cloth.ob
    if ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = get_bmesh(ob)
    sco = cloth.sco
    
    # eliminate: sew edges, outer edges, bend_stiff group weight 0:
    ed = [e for e in obm.edges if
         (len(e.link_faces) > 1) &
         (cloth.bend_cull[e.verts[0].index]) &
         (cloth.bend_cull[e.verts[1].index])]

    first_row = []
    e_tiled = []
    f_ls = []
    f_rs = []
    for e in ed:
        ls = []
        for f in e.link_faces:
            otf = [lf for lf in e.link_faces if lf != f]
            for lf in otf:
                f_ls += [f.index]
                f_rs += [lf.index]
                e_tiled += [e.index]

    shape1 = len(f_ls)
    paired = np.empty((shape1, 2), dtype=np.int32)
    paired[:, 0] = f_ls
    paired[:, 1] = f_rs

    # faces grouped left and right
    cloth.face_pairs, idx = pairs_idx(paired)
    cloth.stacked_faces = cloth.face_pairs.T.ravel()
    jfps = cloth.stacked_faces.shape[0]

    # swap so we get wieghts from tris opposite axis
    cloth.swap_faces = np.empty(jfps, dtype=np.int32)
    cloth.swap_faces[:jfps//2] = cloth.face_pairs[:, 1]
    cloth.swap_faces[jfps//2:] = cloth.face_pairs[:, 0]

    # remove duplicate pairs so edges match face pairs
    tiled_edges = np.array(e_tiled)[idx]

    # v1 and v2 for each face pair (twice as many faces because each pair shares an edge)
    obm.edges.ensure_lookup_table()
    cloth.edv = np.array([[obm.edges[e].verts[0].index,
                     obm.edges[e].verts[1].index]
                     for e in tiled_edges], dtype=np.int32)

    shape = cloth.edv.shape[0]
    cloth.stacked_edv = np.tile(cloth.edv.ravel(), 2)
    cloth.stacked_edv.shape = (shape * 2, 2)


# abstract bend setup ----------------------------
# precalculated ------------------------
def get_poly_vert_tilers(cloth):
    """Get an index to tile the left and right sides.
    ls and rs is based on the left and right sides of
    the face pairs."""

    if cloth.ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = cloth.obm
        obm.faces.ensure_lookup_table()

    cloth.swap_jpv = []
    cloth.jpv_full = []
    ob = cloth.ob

    cloth.ab_faces = []
    cloth.ab_edges = []

    count = 0
    for i, j in zip(cloth.swap_faces, cloth.stacked_edv): # don't need to swap edv because both sides share the same edge

        pvs = [v.index for v in obm.faces[i].verts]
        nar = np.array(pvs)
        b1 = nar != j[0]
        b2 = nar != j[1]

        nums = np.arange(nar.shape[0]) + count
        cloth.ab_faces += nums[b1 & b2].tolist()
        cloth.ab_edges += nums[~(b1)].tolist()
        cloth.ab_edges += nums[~(b2)].tolist()

        count += nar.shape[0]
        r = [v.index for v in obm.faces[i].verts if v.index not in j]
        cloth.swap_jpv += r

    for i in cloth.swap_faces:
        r = [v.index for v in obm.faces[i].verts]
        cloth.jpv_full += r
    

# abstract bend setup ----------------------------
# precalculated ------------------------
def tiled_weights(cloth):
    """Tile the tris with the polys for getting
    barycentric weights"""

    if cloth.ob.MC_props.quad_bend:
        obm = cloth.quad_obm
    else:
        obm = cloth.obm

    ob = cloth.ob
    face_pairs = cloth.face_pairs

    # counts per poly less the two in the edges
    cloth.full_counts = np.array([len(f.verts) for f in obm.faces], dtype=np.int32)
    cloth.full_div = np.array(cloth.full_counts, dtype=np.float32)[cloth.swap_faces][:, None]
    cloth.plot_counts = cloth.full_counts - 2 # used by plotted centers
    
    # joined:
    jfps = cloth.stacked_faces.shape[0]

    jsc = cloth.plot_counts[cloth.swap_faces]
    cloth.j_tiler = np.hstack([[i] * jsc[i] for i in range(jfps)])
    jscf = cloth.full_counts[cloth.swap_faces]

    ab_tiler_1 = np.array([[i] * jscf[i] for i in range(jfps)])
    
    if ab_tiler_1.dtype == 'object':
        abl = []
        for i in ab_tiler_1:
            abl += i
        cloth.ab_tiler = np.array(abl, dtype=np.int32)
    else:
        cloth.ab_tiler = ab_tiler_1.ravel()
    
    face_verts = np.array([[v.index for v in f.verts] for f in obm.faces])
    if face_verts.dtype == 'object':
        this = []
        for i in face_verts[cloth.swap_faces]:
            this += i
        cloth.sp_ls = np.array(this, dtype=np.int32)
    else:
        cloth.sp_ls = face_verts[cloth.swap_faces].ravel()


# abstract bend setup ----------------------------
# precalculated ------------------------
def triangle_data(cloth):
    
    sco = cloth.sco
    edv = cloth.edv
    # joined tris:
    j_tris = np.zeros((cloth.j_tips.shape[0], 3, 3), dtype=np.float32)
    j_tris[:, :2] = sco[cloth.stacked_edv]
    j_tris[:, 2] = cloth.j_tips
    cloth.j_tris = j_tris
    #-----------------

    # get the tilers for creating tiled weights
    tiled_weights(cloth)

    trial = False
    trial = True
    if trial:
        # can probably speed this up by merging the arrays then slicing
        tips, ax, ce = get_eq_tri_tips(cloth, cloth.sco, cloth.source_centers, skip=False)
        c1 = np.cross(ax, ce)
        c2 = np.cross(c1, ax)

        Uax = ax / np.sqrt(np.einsum('ij,ij->i', ax, ax))[:, None]
        Uc1 = c1 / np.sqrt(np.einsum('ij,ij->i', c1, c1))[:, None]
        Uc2 = c2 / np.sqrt(np.einsum('ij,ij->i', c2, c2))[:, None]

        j_mid = sco[cloth.stacked_edv[:,0]] + (ax * 0.5)

        vecs = sco[cloth.swap_jpv] - j_mid[cloth.j_tiler]

        cloth.d1 = np.einsum('ij,ij->i', vecs, Uax[cloth.j_tiler])
        cloth.d2 = np.einsum('ij,ij->i', vecs, Uc1[cloth.j_tiler])
        cloth.d3 = np.einsum('ij,ij->i', vecs, Uc2[cloth.j_tiler])
        cloth.is_flat = np.all(np.abs(cloth.d2) < 0.00001)


# abstract bend setup ----------------------------
# precalculated ------------------------
def ab_setup(cloth):
    cloth.ab_centers = np.empty((cloth.stacked_faces.shape[0], 3), dtype=np.float32)
    cloth.ab_coords = np.empty((len(cloth.jpv_full), 3), dtype=np.float32)

    l = cloth.sp_ls # left side of the springs (Full moved takes the place of the right side)
    a,b,c = np.unique(l, return_counts=True, return_inverse=True)
    x = c[b] # number of times each vert occurs

    x[x < 8] *= 2

    cloth.bend_multiplier = 1 / x[:, None] #(((x - 2.5) / 2.5) + 2.5)[:, None]

    # multiplying the vertex group here (for some reason...)
    cloth.bend_group_mult = cloth.bend_group[l]
    cloth.bend_multiplier *= cloth.bend_group_mult


# abstract bend setup ----------------------------
# precalculated ------------------------
def bend_setup(cloth):
    # if I end up using cross products and unit
    #   vecs instead of wieghts, I can
    #   add the center of the edge
    #   once at the end
    #   instead of doing it for all three
    rt_()
    if cloth.ob.MC_props.quad_bend:
        cloth.quad_obm = get_quad_obm(cloth.ob)
        cloth.quad_obm.faces.ensure_lookup_table()
        cloth.center_data = get_poly_centers_bmesh(cloth.quad_obm, cloth.sco, data=None)
    else:
        cloth.center_data = get_poly_centers(cloth.ob, cloth.sco, data=None)

    cloth.source_centers = np.copy(cloth.center_data[0]) # so we can overwrite the centers array when dynamic
    eq_bend_data(cloth)
    #rt_('1')
    get_poly_vert_tilers(cloth)
    #rt_('2')    
    get_eq_tri_tips(cloth, cloth.sco, cloth.source_centers)
    #rt_('3')    
    triangle_data(cloth)
    #rt_('4')
    ab_setup(cloth)
    rt_('setup bend springs')

# abstract bend setup ----------------------------
# dynamic ------------------------------
def dynamic(cloth):

    # get centers from MC_current
    centers = get_poly_centers(cloth.ob, cloth.co, cloth.center_data)
    co = cloth.co

    tips, ax, ce = get_eq_tri_tips(cloth, co, centers, skip=False)
    c1 = np.cross(ax, ce)
    c2 = np.cross(c1, ax)

    Uax = ax / np.sqrt(np.einsum('ij,ij->i', ax, ax))[:, None]
    Uc1 = c1 / np.sqrt(np.einsum('ij,ij->i', c1, c1))[:, None]
    Uc2 = c2 / np.sqrt(np.einsum('ij,ij->i', c2, c2))[:, None]

    j_mid = co[cloth.stacked_edv[:,0]] + (ax * 0.5)

    p1 = Uax[cloth.j_tiler] * cloth.d1[:, None]
    p3 = Uc2[cloth.j_tiler] * cloth.d3[:, None]

    origin = j_mid[cloth.j_tiler]

    if not cloth.is_flat:
        p2 = Uc1[cloth.j_tiler] * cloth.d2[:, None]
        final_plot = p1 + p2 + p3 + origin

    if cloth.is_flat:
        final_plot = p1 + p3 + origin

    # get centers from plot
    cloth.ab_centers[:] = 0
    cloth.ab_centers += co[cloth.stacked_edv[:, 0]]
    cloth.ab_centers += co[cloth.stacked_edv[:, 1]]
    np.add.at(cloth.ab_centers, cloth.j_tiler, final_plot)

    cloth.ab_centers /= cloth.full_div

    c_vecs = centers[cloth.swap_faces] - cloth.ab_centers

    cloth.ab_coords[cloth.ab_faces] = final_plot
    cloth.ab_coords[cloth.ab_edges] = cloth.co[cloth.stacked_edv.ravel()]

    full_moved = cloth.ab_coords + c_vecs[cloth.ab_tiler]

    cloth.full_moved = full_moved


# abstract bend setup ----------------------------
# dynamic ------------------------------
def abstract_bend(cloth):
    dynamic(cloth)
    stretch = cloth.ob.MC_props.bend
    cv = (cloth.full_moved - cloth.co[cloth.sp_ls])
    mult = cloth.bend_multiplier * stretch
    cv *= mult
    np.add.at(cloth.co, cloth.sp_ls, np.nan_to_num(cv))


def abstract_bend_(cloth):
    # weighted average method
    # !!! this might be better. Need to test
    dynamic(cloth)
    stretch = cloth.ob.MC_props.bend
    cv = (cloth.full_moved - cloth.co[cloth.sp_ls])
    lens = np.sqrt(np.einsum('ij,ij->i', cv, cv))
    stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
    np.add.at(stretch_array, cloth.sp_ls, lens)
    w = (lens / stretch_array[cloth.sp_ls]) * stretch
    cv *= w[:, None]
    
    np.add.at(cloth.co, cloth.sp_ls, np.nan_to_num(cv))

#                                                                #
#                                                                #
# ------------------- end abstract bend data ------------------- #

# universal ---------------
def inside_triangles(tris, points, check=True, surface_offset=False, cloth=None):
    """Can check inside triangle.
    Can find barycentric weights for triangles.
    Can find multiplier for distance off surface
        from non-unit cross product."""
    origins = tris[:, 0]
    cross_vecs = tris[:, 1:] - origins[:, nax]
    v2 = points - origins

    # ---------
    v0 = cross_vecs[:,0]
    v1 = cross_vecs[:,1]

    d00_d11 = np.einsum('ijk,ijk->ij', cross_vecs, cross_vecs)
    d00 = d00_d11[:,0]
    d11 = d00_d11[:,1]
    d01 = np.einsum('ij,ij->i', v0, v1)
    d02 = np.einsum('ij,ij->i', v0, v2)
    d12 = np.einsum('ij,ij->i', v1, v2)

    div = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * div
    v = (d00 * d12 - d01 * d02) * div

    weights = np.array([1 - (u+v), u, v, ]).T
    if not check | surface_offset:
        return weights

    if not check:
        cross = np.cross(cross_vecs[:,0], cross_vecs[:,1])
        d_v2_c = np.einsum('ij,ij->i', v2, cross)
        d_v2_v2 = np.einsum('ij,ij->i', cross, cross)
        div = d_v2_c / d_v2_v2

        U_cross = cross / np.sqrt(d_v2_v2)[:, None]
        U_d = np.einsum('ij,ij->i', v2, U_cross)


        return weights, div, U_d# for normalized

    check = np.all(weights > 0, axis=0)
    # check if bitwise is faster when using lots of tris
    if False:
        check = (u > 0) & (v > 0) & (u + v < 1)

    return weights.T, check


def get_cloth(ob):
    """Return the cloth instance from the object"""
    return MC_data['cloths'][ob['MC_cloth_id']]

# ^                                                          ^ #
# ^                 END universal functions                  ^ #
# ============================================================ #


# ============================================================ #
#                   precalculated data                         #
#                                                              #

# precalculated ---------------
def closest_point_mesh(cloth, target):
    """Using blender built-in method"""
    
    note = 'can use this to do surface follow with some mods'
    note_2 = 'currently only works with one collider'
    
    # get world co for cloth
    lco = apply_transforms(cloth.ob, cloth.co)
    
    # apply cloth world to object local
    ico = invert_transforms(target, lco)
    
    co = []
    for c in ico:
        hit, loc, norm, face_index = target.closest_point_on_mesh(c)
        co += [loc]
    
    ap_co = apply_transforms(target, co)
    
    vecs = ap_co - lco
    
    # apply global force to cloth local
    move = revert_rotation(cloth.ob, vecs)

    return move
    

def get_tridex(ob, tobm=None):
    """Return an index for viewing the verts as triangles"""
    free = False
    if tobm is None:
        tobm = bmesh.new()
        tobm.from_mesh(ob.data)
        free = True
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    tridex = np.array([[v.index for v in f.verts] for f in tobm.faces], dtype=np.int32)
    if free:
        tobm.free()
    return tridex


def get_tridex_2(ob, mesh=None): # faster than get_tridex()
    """Return an index for viewing the
    verts as triangles using a mesh and
    foreach_get. Faster than get_tridex()"""

    if mesh is not None:
        tobm = bmesh.new()
        tobm.from_mesh(mesh)
        bmesh.ops.triangulate(tobm, faces=tobm.faces)
        me = bpy.data.meshes.new('tris')
        tobm.to_mesh(me)
        p_count = len(me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        me.polygons.foreach_get('vertices', tridex.ravel())

        # clear unused tri mesh
        bpy.data.meshes.remove(me)
        if ob == 'p':
            return tridex, tobm
        
        tobm.free()
        return tridex

    if ob.data.is_editmode:
        ob.update_from_editmode()
        
    tobm = bmesh.new()
    tobm.from_mesh(ob.data)
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)
    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=np.int32)
    me.polygons.foreach_get('vertices', tridex.ravel())

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    return tridex, tobm


def get_sc_edges(ob, fake=False):
    """Edge indexing for self collision"""
    if fake:
        c = len(ob.data.vertices)
        ed = np.empty((c, 2), dtype=np.int32)
        idx = np.arange(c * 2, dtype=np.int32)
        ed[:, 0] = idx[:c]
        ed[:, 1] = idx[c:]
        return ed
    
    ed = np.empty((len(ob.data.edges), 2), dtype=np.int32)
    ob.data.edges.foreach_get('vertices', ed.ravel())
    return ed


# precalculated ---------------
def create_surface_follow_data(active, cloths):
    """Need to run this every time
    we detect 'not same' """

    # !!! need to make a global dg shared between cloth objects???
    selection = active.MC_props.surface_follow_selection_only

    for c in cloths:
        cloth = MC_data['cloths'][c['MC_cloth_id']]
        proxy = active.evaluated_get(cloth.dg)
        cloth.surface = True
        cloth.surface_object = proxy

        vw = get_weights(c, 'MC_surface_follow')
        idx = np.where(vw != 0)[0]
        cloth.bind_idx = idx

        print("have to run the surface calculations when updating groups")

        if selection:
            sel_object, sel_faces, idx_faces = mesh_from_selection(proxy, "temp_delete_me")
            if sel_object is None:
                return 'Select part of the mesh or turn off "Selection Only"'

            # triangulate selection object mesh for barycentric weights
            sel_obm = bmesh.new()
            sel_obm.from_mesh(sel_object.data)
            bmesh.ops.triangulate(sel_obm, faces=sel_obm.faces[:])
            sel_obm.to_mesh(sel_object.data)
            sel_object.data.update()
            copy_object_transforms(proxy, sel_object)
            sel_tridex = get_tridex(sel_object, sel_obm)
            sel_obm.free()

            # find barycentric data from selection mesh
            locs, faces, norms, cos = closest_point_mesh(cloth, c, idx, sel_object)

            # converts coords to dict keys so we can find corresponding verts in surface follow mesh
            idx_co_key = {}
            pvs = proxy.data.vertices
            flat_faces = np.hstack(sel_faces)
            for i in range(len(flat_faces)):
                idx_co_key[str(pvs[flat_faces[i]].co)] = flat_faces[i]

            # use keys to get correspoinding vertex indices
            flat_2 = np.hstack(sel_tridex[faces])
            co_key_2 = []
            vs = sel_object.data.vertices
            tri_keys = [str(vs[i].co) for i in flat_2]
            surface_follow_tridex = [idx_co_key[i] for i in tri_keys]

            # convert to numpy and reshape
            cloth.surface_tridex = np.array(surface_follow_tridex)

            cloth.surface_tris_co = np.array([pvs[i].co for i in cloth.surface_tridex], dtype=np.float32)
            shape = cloth.surface_tridex.shape[0]
            cloth.surface_tridex.shape = (shape // 3, 3)
            cloth.surface_tris_co.shape = (shape // 3, 3, 3)
            #v = proxy.data.vertices
            #for i in cloth.surface_tridex.ravel():
                #bpy.ops.object.empty_add(location=proxy.matrix_world @ v[i].co)
            #print(tri_keys)

            # delete temp mesh
            me = sel_object.data
            bpy.data.objects.remove(sel_object)
            bpy.data.meshes.remove(me)

            # generate barycentric weights
            cloth.surface_bary_weights = inside_triangles(cloth.surface_tris_co, locs, check=False)


            dif = cos - apply_transforms(active, locs)
            #cloth.surface_norms = norms
            cloth.surface_norm_vals = np.sqrt(np.einsum("ij ,ij->i", dif, dif))[:, nax]

            # Critical values for barycentric placement:
            #   1: cloth.surface_tridex          (index of tris in surface object)
            #   2: cloth.surface_bary_weights    (barycentric weights)
            #   3: cloth.surface_object          (object we're following)
            #   4: cloth.surface                 (bool value indicating we should run surface code)
            #   5: cloth.bind_idx = idx          (verts that are bound to the surface)
            #   6: cloth.surface_norms = norms   (direction off surface of tris)
            #   7: cloth.surface_normals = norms (mag of surface norms)


def virtual_springs(cloth):
    """Adds spring sets checking if springs
    are already present using strings.
    Also stores the set so we can check it
    if there are changes in geometry."""

    # when we detect changes in geometry we
    #   regenerate the springs, then add
    #   cloth.virtual_springs to the basic
    #   array after we check that all the
    #   verts in the virtual springs are
    #   still in the mesh.
    verts = cloth.virtual_spring_verts
    ed = cloth.basic_set

    string_spring = [str(e[0]) + str(e[1]) for e in ed]

    strings = []
    new_ed = []
    for v in verts:
        for j in verts:
            if j != v:
                stringy = str(v) + str(j)
                strings.append(stringy)
                #if stringy not in string_spring:
                new_ed.append([v,j])

    in1d = np.in1d(strings, string_spring)
    cull_ed = np.array(new_ed)[~in1d]
    cloth.virtual_springs = cull_ed # store it for checking when changing geometry
    cloth.basic_set = np.append(cloth.basic_set, cull_ed, axis=0)
    cloth.basic_v_fancy = cloth.basic_set[:,0]
    # would be nice to have a mesh or ui magic to visualise virtual springs
    # !!! could do a fixed type sew spring the same way !!!
    # !!! maybe use a vertex group for fixed sewing? !!!
    # !!! Could also use a separate object for fixed sewing !!!

    cloth.measure_length = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32)
    cloth.measure_dot = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32)
    
    cloth.vdl = stretch_springs_basic(cloth)


def get_sew_springs(cloth):
    """Creates data for using add.at with average locations
    of sew verts. Groups areas using a sort of tree where
    multiple edges would bring sew verts together."""
    rt_()

    if cloth.ob.MC_props.simple_sew:
        print("used simple sew")
        obm = cloth.obm
        cloth.sew_fancy_indexer = []
        cloth.sew_add_indexer = []
        #cloth.simple_sew_v = []
        #cloth.simple_sew_fancy = []
        for v in obm.verts:
            le = [e.other_vert(v).index for e in v.link_edges if len(e.link_faces) == 0]
            cloth.sew_add_indexer += [v.index] * len(le)
            cloth.sew_fancy_indexer += le
            
        #print(cloth.simple_sew_v, "sew verts")
        #print(cloth.simple_sew_fancy, "sew fancy")
        #print('done with simple sew')
        if len(cloth.sew_fancy_indexer) != 0:
            cloth.sew = True
        rt_('time to simple sew')
        return
        
    obm = cloth.obm    
    cloth.sew = True
    cull = []
    pairs = [] # edge indices of singe sew edges
    groups = []

    for e in obm.edges:
        if len(e.link_faces) == 0:
            if e.index not in cull:
                cull += [e.index]
                
                v1 = e.verts[0]
                le1 = [ed.index for ed in v1.link_edges if (ed.index not in cull) & (len(ed.link_faces) == 0)]
                cull += le1
                
                v2 = e.verts[1]
                le2 = [ed.index for ed in v2.link_edges if (ed.index not in cull) & (len(ed.link_faces) == 0)]
                cull += le2

                if len(le1 + le2) == 0:
                    pairs += [e.index]
                    continue

                eg = []
                eg += le1
                eg += le2
                keep_going = True
                
                itg = eg
                while len(itg) != 0:
                    for ee in itg:

                        cull += [ee]
                        
                        v1 = obm.edges[ee].verts[0]
                        le1 = [ed.index for ed in v1.link_edges if (ed.index not in cull) & (len(ed.link_faces) == 0)]
                        cull += le1
                        
                        v2 = obm.edges[ee].verts[1]
                        le2 = [ed.index for ed in v2.link_edges if (ed.index not in cull) & (len(ed.link_faces) == 0)]
                        cull += le2
                        merge = le1 + le2

                        if len(merge) == 0:
                            keep_going = False

                        itg = merge
                        eg += merge

                eg += [e.index]
                groups.append(eg)
                
    
    if len(pairs) == 0:
        if len(groups) == 0:    
            cloth.sew = False
            return
                    
    e_count = len(cloth.ob.data.edges)
    eidx = np.empty((e_count, 2), dtype=np.int32)
    cloth.ob.data.edges.foreach_get('vertices', eidx.ravel())
    
    sew_pairs_v = eidx[pairs].ravel().tolist()
    sew_groups_v = []
    
    indexer = []
    divs = []
    for i, g in enumerate(groups):
        temp = []
        for v in eidx[g].ravel(): # compare to unique, unique was slower!!
            if v not in temp:
                temp += [v]
        
        sew_groups_v += temp
        lt = len(temp)
        divs += [lt]
        indexer += [i] * lt
    
    pidx1 = np.arange(len(pairs))
    pidx = np.repeat(pidx1, 2) + (len(groups))

    count = len(groups) + len(pairs)
    cloth.sew_mean = np.zeros((count, 3), dtype=np.float32)
    cloth.all_sew = sew_groups_v + sew_pairs_v
    cloth.sew_mean_idx = indexer + pidx.tolist()
    cloth.sew_div = np.array(divs + ([2] * len(pairs)),dtype=np.float32)[:, None]
    rt_('time to sew')
    

def hook_force(cloth):
    id = cloth.ob['MC_cloth_id']
    empties = [o for o in bpy.data.objects if o.type == 'EMPTY']
    # fix this by creating a prop
    empties = [h for h in empties if 'MC_cloth_id' in h]
    hooks = [h for h in empties if h['MC_cloth_id'] == id]
    if len(hooks) < 1:
        return
    idx = [h.MC_props.hook_index for h in hooks]
    hook_co = np.array([cloth.ob.matrix_world.inverted() @ h.matrix_world.to_translation() for h in hooks], dtype=np.float32)
    cloth.co[idx] = hook_co    


def sew_v_fancy(cloth):

    if cloth.ob.MC_props.simple_sew:
        return
    
    if not cloth.sew:
        return
    
    npas = np.array(cloth.all_sew)
    npsm = np.array(cloth.sew_mean_idx)

    sew_add_indexer = []
    sew_fancy_indexer = []
    
    '''
    [0, 0, 0, 0, 1, 1] This represents the 0 group and the 1 group
    These points all sew together to the same spot.
    
    [ 1  9 21 29  5 25]
    corresponds to the above. 1, 9, 21, 29 all come together.
    5 and 25 come together.
    
    need an array like v_fancy. 
    for the 1 need [9, 21, 9
    for the 9 need [1, 21, 29
    can stack them for add.at like [9, 21, 9, 1, 21, 29
    need to stack them like [1, 1, 1, 9, 9, 9
    '''
    
    for i, j in enumerate(cloth.sew_mean_idx):
        meh = npas[npsm == j]        
        sew_add_indexer += [[npas[i]] * (meh.shape[0] - 1)]
        sew_fancy_indexer += [meh[meh != [npas[i]]]]
        
    cloth.sew_add_indexer = np.hstack(sew_add_indexer)
    cloth. sew_fancy_indexer = np.hstack(sew_fancy_indexer)


def sew_force_1(cloth):
    if not cloth.sew:
        return
    
    tl = cloth.ob.MC_props.target_sew_length / 2
    
    cloth.sew_mean[:] = 0.0
    np.add.at(cloth.sew_mean, cloth.sew_mean_idx, cloth.co[cloth.all_sew])
    means = cloth.sew_mean / cloth.sew_div
    vecs = (means[cloth.sew_mean_idx] - cloth.co[cloth.all_sew]) #* cloth.ob.MC_props.sew_force
    
    
    if cloth.ob.MC_props.self_collide:
        if cloth.ob.MC_props.self_collide_margin > tl * 2:
            tl = cloth.ob.MC_props.self_collide_margin / 2
    
    if tl != 0:
        vecs = cloth.co[cloth.sew_fancy_indexer] - cloth.co[cloth.sew_add_indexer]
        l = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
        move_l = (l - tl)# * cloth.ob.MC_props.sew_force
        vecs *= (move_l / l)[:, None]
        
        # makes shorter springs more powerful
        hold = move_l / move_l ** 2
        vecs *= hold[:, None]
        # -----------------------------------
        
        nn = np.nan_to_num(vecs) * cloth.ob.MC_props.sew_force
        np.add.at(cloth.co, cloth.sew_add_indexer, nn)
        return

    nn = np.nan_to_num(vecs)# * cloth.ob.MC_props.sew_force
    #np.add.at(cloth.co, cloth.all_sew, nn)
    cloth.co[cloth.all_sew] += vecs * cloth.ob.MC_props.sew_force
    print(cloth.all_sew)
    print(cloth.sew_mean_idx, "mean idx")
    #print(np.unique(cloth.all_sew).shape, "uin")
    

def sew_force(cloth):
    if not cloth.sew:
        return

    tl = cloth.ob.MC_props.target_sew_length / 2

    if cloth.ob.MC_props.self_collide:
        if cloth.ob.MC_props.self_collide_margin > tl * 2:
            tl = cloth.ob.MC_props.self_collide_margin * 1.001

    if tl != 0:
        vecs = cloth.co[cloth.sew_fancy_indexer] - cloth.co[cloth.sew_add_indexer]
        l = np.sqrt(np.einsum('ij,ij->i', vecs, vecs))
        move_l = (l - tl)# * cloth.ob.MC_props.sew_force
        vecs *= ((l - tl) / l)[:, None]

        nn = np.nan_to_num(vecs)# * (cloth.ob.MC_props.sew_force * .5)
        short = move_l < 0.5
        nn[short] *= .1
        nn[~short] *= (cloth.ob.MC_props.sew_force * .5)
        
        #nn = np.nan_to_num(vecs) * (cloth.ob.MC_props.sew_force * .5)
        np.add.at(cloth.co, cloth.sew_add_indexer, nn)

        return

    cloth.sew_mean[:] = 0.0
    np.add.at(cloth.sew_mean, cloth.sew_mean_idx, cloth.co[cloth.all_sew])
    means = cloth.sew_mean / cloth.sew_div
    vecs = (means[cloth.sew_mean_idx] - cloth.co[cloth.all_sew])
    
    need_to_verify = "check if add.at works better !!!"
    nn = np.nan_to_num(vecs)# * cloth.ob.MC_props.sew_force
    #np.add.at(cloth.co, cloth.all_sew, nn)
    cloth.co[cloth.all_sew] += vecs * cloth.ob.MC_props.sew_force
        

def get_springs_2(cloth):
    """Create index for viewing stretch springs"""

    obm = cloth.obm

    if not cloth.do_stretch:
        return

    ed = []
    for v in obm.verts:
        if cloth.linear_cull[v.index]:
            v_set = []
            for f in v.link_faces:
                for vj in f.verts:
                    if vj != v:
                        stri = str(v.index) + str(vj.index)
                        if stri not in v_set:
                            ed.append([v.index, vj.index])
                            v_set.append(stri)

    cloth.basic_set = np.array(ed)
    cloth.basic_v_fancy = cloth.basic_set[:,0]
    cloth.stretch_group_mult = cloth.stretch_group[cloth.basic_v_fancy]


# ^                                                          ^ #
# ^                 END precalculated data                   ^ #
# ============================================================ #

# ============================================================ #
#                      cloth instance                          #
#                                                              #

def manage_vertex_groups(cloth):
    """Create vertex groups and generate
    numpy arrays for them."""

    ob = cloth.ob
    #obm = cloth.obm
    #if ob.data.is_editmode:
    #ob.data.update()
    obm = get_bmesh(ob)

    vw = 1 # so we avoid calculating except where the spring values are more than zero
    if ob.MC_props.dense:
        vw = 0

    np_groups = []
    groups = [('MC_pin',            0.0),
              ('MC_drag',           0.0),
              ('MC_surface_follow', 0.0),
              ('MC_bend_stiff',      vw),
              ('MC_stretch',         vw),
              ('MC_collide_offset',  vw),
              ]

    for i in groups:
        np_groups.append(get_weights(ob, i[0], obm, i[1]))

    cloth.pin = np_groups[0][:, None]
    cloth.drag = np_groups[1][:, None]
    cloth.surface_follow = np_groups[2]
    cloth.bend_group = np_groups[3][:, None]
    cloth.stretch_group = np_groups[4][:, None]
    cloth.group_surface_offset = np_groups[5]

    cloth.bend_cull = cloth.bend_group > 0

    cloth.do_bend = False
    if np.any(cloth.bend_cull):
        cloth.do_bend = True
        cloth.sco = get_co_shape(cloth.ob, key='MC_source', ar=None)
    
        if cloth.ob.MC_props.simple_sew:
            cloth.sco = get_co_shape(cloth.ob, "pre_wrap")
            print("used the pre_wrap shape")
        
        bend_setup(cloth)
        
        if cloth.ob.MC_props.simple_sew:
            cloth.sco = get_co_shape(cloth.ob, key='MC_source', ar=None)

    cloth.linear_cull = cloth.stretch_group > 0
    cloth.do_stretch = False
    if np.any(cloth.linear_cull):
        cloth.do_stretch = True
        get_springs_2(cloth)
    
    # get_sew_springs(cloth)
    
    cloth.skip_bend_wieght = np.all(cloth.bend_group == 1) # for optimizing by elimating multiplier
    cloth.skip_stretch_wieght = np.all(cloth.stretch_group == 1) # for optimizing by elimating multiplier
    # when running in p1 just write to these two groups
    # before turning on cloth.

    if ob.data.is_editmode:
        bmesh.update_edit_mesh(ob.data)
        return

    cloth.obm.to_mesh(ob.data)
    cloth.ob.data.update()


class Collider():
    # The collider object
    name = 'inital name'

    def __init__(self, cloth=None):
        
        #print('----------__init__-----------')
        
        colliders = [o for o in bpy.data.objects if (o.MC_props.collider) & (o != cloth.ob)]
        if len(colliders) == 0:
            return
        geo_check = []
        cloth.collider_count = len(colliders)
        
        cloth.total_co = np.empty((0, 3), dtype=np.float32)
        oc_total_tridex = np.empty((0,3), dtype=np.int32)
        cloth.oc_tri_counts = []
        cloth.oc_v_counts = []
        
        cloth.ob_v_norm_indexer1 = []
        cloth.ob_v_norm_indexer = []
        
        shift = 0
        
        f_shift = 0
        for i, c in enumerate(colliders):
            abco, proxy, prox = absolute_co(c)
            gt2, triobm = get_tridex_2(ob='p', mesh=proxy)

            cloth.oc_tri_counts.append(gt2.shape[0])
            cloth.oc_v_counts.append(abco.shape[0])
            oc_total_tridex = np.append(oc_total_tridex, gt2 + shift, axis=0)

            ob_settings = not cloth.ob.MC_props.override_settings

            cloth.ob_v_norm_indexer1 += [[f.index + f_shift for f in v.link_faces] for v in triobm.verts]
            cloth.ob_v_norm_indexer += [[v.index + shift] * len(v.link_faces) for v in triobm.verts]

            geo_check.append(abco.shape[0])
                    
            cloth.total_co = np.append(cloth.total_co, abco, axis=0)
            sh = abco.shape[0]
            shift += sh
            f_shift += len(triobm.faces)
        
        cloth.geo_check = np.array(geo_check)
        cloth.oc_tri_counts = np.cumsum(cloth.oc_tri_counts)
        cloth.oc_v_counts = np.cumsum(cloth.oc_v_counts)
        
        cloth.last_co = np.copy(cloth.total_co)# - cloth.inner_norms # for checing if the collider moved
        cloth.collider_sh = cloth.total_co.shape[0]
        cloth.ob_v_norms = np.zeros_like(cloth.total_co)

        cloth.ob_v_norm_indexer1 = np.hstack(cloth.ob_v_norm_indexer1)
        cloth.ob_v_norm_indexer1 = np.array(cloth.ob_v_norm_indexer1, dtype=np.int32)
        cloth.ob_v_norm_indexer = np.hstack(cloth.ob_v_norm_indexer)
        cloth.ob_v_norm_indexer = np.array(cloth.ob_v_norm_indexer, dtype=np.int32)

        cloth.oc_total_tridex = oc_total_tridex
        update_ob_v_norms(cloth)

        cloth.ob_co = np.empty((cloth.co.shape[0] * 2, 3), dtype=np.float32)
        # ==========================================
        # if I put these on vertex groups I can just multiply the
        # values by the groups. Will have to create groups
        # for the collide objects and set them to one by default.
        
        frs = [c.MC_props.outer_margin for c in colliders]
        sfrs = [c.MC_props.static_friction  * .0001 for c in colliders]
        
        fcs = oc_total_tridex.shape[0]
        cloth.total_margins = np.zeros(cloth.total_co.shape[0], dtype=np.float32)[:, None]
        cloth.total_inner_margins = np.zeros(cloth.total_co.shape[0], dtype=np.float32)[:, None]
        cloth.total_friction = np.ones(fcs, dtype=np.float32)[:, None]
        cloth.total_static = np.zeros(fcs, dtype=np.float32)
        
        cloth.oc_total_tridex = oc_total_tridex
        cloth.oc_indexer = np.arange(oc_total_tridex.shape[0], dtype=np.int32)
        cloth.static = False
        cloth.fcs = fcs
        cloth.oc_tris_six = np.empty((oc_total_tridex.shape[0], 6, 3), dtype=np.float32)
        cloth.oc_eidx = np.arange(len(cloth.ob.data.vertices), dtype=np.int32)
        cloth.traveling_edge_co = np.empty((cloth.co.shape[0], 2, 3), dtype=np.float32)
        
        # !!! not currently using this. Need to research total object bounds speed difference
        cloth.tris6_bool = np.ones(cloth.oc_total_tridex.shape[0], dtype=np.bool)


# cloth instance ---------------
class Cloth(object):
    # The cloth object
    def __init__(self):
        pass

    def refresh(self):
        print('plan to move here for sorting')

    def soft_refresh():
        # like vertex weights without remaking all the springs
        print('does not require recalculating springs')


# cloth instance ---------------
def create_instance(ob=None):
    """Run this when turning on modeling cloth."""
    global glob_dg
    cloth = Cloth()
    cloth.dg = bpy.context.evaluated_depsgraph_get()
    glob_dg = cloth.dg
    if ob is None:
        ob = bpy.context.object
    cloth.ob = ob
    cloth.vcs = 0 # for object collisions
    refresh(cloth)
    return cloth

# ^                                                          ^ #
# ^                   END cloth instance                     ^ #
# ============================================================ #


# ============================================================ #
#                     update the cloth                         #
#                                                              #

# update the cloth ---------------
def update_groups(cloth, obm=None, geometry=False):
    """Create update data in the cloth instance
    related to the vertex groups.
    geometry is run when there are changes in the geomtry"""
    #ob = cloth.ob
    #current_index = ob.vertex_groups.active_index

    # vertex groups
    current = np.copy(cloth.pin)

    cloth.obm = get_bmesh(cloth.ob)
    manage_vertex_groups(cloth)

    if geometry:
        old = current.shape[0]
        new = cloth.pin.shape[0]
        dif = new - old
        if dif > 0:
            fix = np.zeros_like(cloth.pin)
            fix[:old] += current
            current = fix
            cloth.pin_arr = np.append(cloth.pin_arr, cloth.co[old:], axis=0)
            zeros = np.zeros_like(cloth.co[old:])
            cloth.velocity = np.append(cloth.velocity, zeros, axis=0)
            cloth.vel_zero = np.copy(cloth.co)
            cloth.feedback = np.copy(cloth.co)
        else:
            cloth.pin_arr = np.copy(cloth.co)
            cloth.vel_zero = np.copy(cloth.co)
            cloth.feedback = np.copy(cloth.co)
            cloth.velocity = np.zeros_like(cloth.co)
            return
    # !!! Need to store cloth.pin_arr in the save file !!!
    changed = (cloth.pin - current) != 0 # Have to manage the location of the pin verts with weights less than one
    if hasattr(cloth, 'co'):
        cloth.pin_arr[changed.ravel()] = cloth.co[changed.ravel()]

    # update surface weight
    #ob.vertex_groups.active_index = current_index
    

# update the cloth ---------------
def measure_edges(co, idx, cloth, source=False):
    """Takes a set of coords and an edge idx and measures segments"""
    l = idx[:,0]
    r = idx[:,1]
    
    if not source:    
        np.subtract(co[r], co[l], out=cloth.measure_cv, dtype=np.float32)
        np.einsum("ij ,ij->i", cloth.measure_cv, cloth.measure_cv, out=cloth.measure_dot)
        np.sqrt(cloth.measure_dot, out=cloth.measure_length)
        return
    
    v = co[r] - co[l]
    d = np.einsum("ij ,ij->i", v, v)
    le = np.sqrt(d)
    return v, d, le


def stretch_springs_basic(cloth, target=None): # !!! need to finish this
    """Measure the springs"""
    if target is not None:
        dg = cloth.dg
        #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
        #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh
        if cloth.proxy is None:
            cloth.proxy = target.evaluated_get(dg)
        co = get_co_mode(cloth.proxy) # needs to be depsgraph eval
        # need to get co with modifiers that don't affect the vertex count
        # so I could create a list of mods to turn off then use that fancy
        # thing I created for turning off modifiers in the list.
        return measure_edges(co, cloth.basic_set, cloth, source=True)

    # can't figure out how to update new verts to source shape key when
    #   in edit mode. Here we pull from source shape and add verts from
    #   current bmesh where there are new verts. Need to think about
    #   how to fix this so that it blendes correctly with the source
    #   shape or target... Confusing....  Will also need to update
    #   the source shape key with this data once we switch out
    #   of edit mode. If someone is working in edit mode and saves
    #   their file without switching out of edit mode I can't fix
    #   that short of writing these coords to a file.
    
    if cloth.ob.data.is_editmode:
        cloth.ob.update_from_editmode()

    co = get_co_shape(cloth.ob, 'MC_source')
    vdl = measure_edges(co, cloth.basic_set, cloth, source=True)
    return vdl
    #cloth.measure_length = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32)
    #return np.copy(v), np.copy(cloth.measure_dot), np.copy(cloth.measure_length)


def surface_forces(cloth):
    if not cloth.surface:
        return
    # surface follow data ------------------------------------------
    # cloth.surface_follow    (weights on the cloth object)
    # cloth.bind_idx = idx            (verts that are bound to the surface)
    tridex = cloth.surface_tridex          # (index of tris in surface object)
    bary = cloth.surface_bary_weights    # (barycentric weights)
    so = cloth.surface_object      # (object we are following)

    shape = cloth.surface_tridex.shape
    tri_co = np.array([so.data.vertices[i].co for i in tridex.ravel()], dtype=np.float32)
    tri_co.shape = (shape[0] * 3, 3)
    apply_in_place(cloth.surface_object, tri_co)

    tri_co.shape = (shape[0], 3, 3)
    plot = np.sum(tri_co * bary[:, :, nax], axis=1)

    # update the normals -----------------
    cloth.surface_norms = get_normals_from_tris(tri_co)
    norms = cloth.surface_norms * cloth.surface_norm_vals
    #print(cloth.surface_norm_vals[0], 'what is this norm val???????')
    plot += norms
    #plot += apply_rotation(cloth.ob, norms)

    world_co = apply_in_place(cloth.ob, cloth.co[cloth.bind_idx])
    dif = (plot - world_co) * cloth.surface_follow[cloth.bind_idx]
    cloth.co[cloth.bind_idx] += revert_rotation(cloth.ob, dif)


def stretch_solve(cloth):
    """Uses wieghted average to determine how much
    a spring should contribute to the movement."""        
    
    stretch = cloth.ob.MC_props.stretch * 0.5
    push = cloth.ob.MC_props.push
    
    # !!! Optimize here ============================================
    # measure source
    v, d, l = cloth.vdl
    if cloth.ob.MC_props.shrink_grow != 1:
        l = l * cloth.ob.MC_props.shrink_grow
    #dynamic = False
    #if dynamic:
        # !!! don't need to run this all the time. Can get a speed improvement here
        #   by caching these values and running them when other updates run
        # v, d, l = stretch_springs_basic(cloth, cloth.target) # from target or source key
    
    measure_edges(cloth.co, cloth.basic_set, cloth) # from current cloth state
    #cv, cd, cl = measure_edges(cloth.co, cloth.basic_set, cloth, source=True) # from current cloth state
    cv = cloth.measure_cv
    cd = cloth.measure_dot
    cl = cloth.measure_length

    move_l = (cl - l) * stretch

    if not cloth.skip_stretch_wieght:
        move_l *= cloth.stretch_group_mult.ravel()

    # separate push springs
    if push != 1:
        push_springs = move_l < 0
        move_l[push_springs] *= push

    # !!! here we could square move_l to accentuate bigger stretch
    # !!! see if it solves better.

    # mean method -------------------
    cloth.stretch_array[:] = 0.0

    rock_hard_abs = np.abs(move_l)
    np.add.at(cloth.stretch_array, cloth.basic_v_fancy, rock_hard_abs)
    weights = rock_hard_abs / cloth.stretch_array[cloth.basic_v_fancy]
    # mean method -------------------

    # apply forces ------------------
    #if False:
    move = cv * (move_l / cl)[:,None]

    move *= weights[:,None]

    np.add.at(cloth.co, cloth.basic_v_fancy, np.nan_to_num(move))


def update_pins_select_sew_surface(cloth):
    """When iterating forces we get sag if we don't update pins
    and selected areas."""
    # sewing ---------------------
    #sew_force(cloth) # no iterate so no: update_pins_and_select(cloth)
    
    # selected -------------------
    pin_vecs = (cloth.pin_arr - cloth.co)
    cloth.co += (pin_vecs * cloth.pin)
    
    if bpy.context.scene.MC_props.pause_selected:
        if cloth.ob.data.is_editmode:
            cloth.co[cloth.selected] = cloth.select_start[cloth.selected]
            cloth.pin_arr[cloth.selected] = cloth.select_start[cloth.selected]
    
    # hooks ----------------------
    hook_force(cloth)


def inflate_and_wind(cloth):
    skip = True
    wind = False
    props = cloth.ob.MC_props
    inflate = props.inflate * .001
    if inflate != 0:
        skip = False
    
    wind_vec = np.array([props.wind_x, props.wind_y, props.wind_z], dtype=np.float32)
    if not np.all(wind_vec == np.array([0.0, 0.0, 0.0], dtype=np.float32)):
        wind = True
        skip = False
    
    if skip:
        return
    
    t = cloth.co[cloth.tridex]
    ori = t[:, 0]
    t1 = t[:, 1] - ori
    t2 = t[:, 2] - ori
    
    # could use norms from wind instead:
    # norms = cloth.u_norms[trs]
    norms = np.cross(t1, t2)
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    cloth.u_norms = un # could feed this to self collisions
    
    # inflate:
    # could put it on a vertex group...
    move = np.nan_to_num(un * inflate)
    np.add.at(cloth.velocity, cloth.tridex, move[:, None])
    
    # wind:
    if wind:
        turb = props.turbulence
        randir = props.random_direction
        if cloth.turb_count % 10 == 0:
            cloth.turb_count = 1
            cloth.turbulence[:] = (1 + turb) - (np.random.rand(cloth.tridex.shape[0]) * turb)
            cloth.random_dir_2[:] = cloth.random_dir
            cloth.random_dir = np.random.rand(3)
        
        dif = cloth.random_dir_2 - cloth.random_dir
        gradual = cloth.random_dir + (dif/cloth.turb_count)
            
        vec_turb = (1 + randir) - (gradual * randir)
        wind_vec = wind_vec * vec_turb
        
        angle = np.abs(un @ wind_vec) * cloth.turbulence
        move = np.nan_to_num(wind_vec * angle[:, None]) * .001
        np.add.at(cloth.velocity, cloth.tridex, move[:, None])
        
        cloth.turb_count += 1


def wrap_force_cpm(cloth):
    
    print('cpm stands for closest point on mesh')
    # the theory is to get closest point
    # on mesh for each vertex and use that
    # as a direction for a force to move it
    # towards the body.
    
def resolve_self_collisions(cloth):
    
    # find where edges pass through tris?
    # do like a flood fill around the point
    # to get connected points.
    # Do like a flood fill around the tris to
    # get connected tris
    pass
    
    
def basic_flatten_bend(cloth):
    # wouldn't work with folds...
    # find surrounding points and make a triangle.
    # move the point along the normal of that
    #   triangle towards it's surface.
    # could exclude the fold edges...
    # could use this for something like
    # a pre-solve fit. Pull the panels
    # towards each other and towards the body...
    pass

def divide_layers_and_object_collide():
    # object collide is more sure because
    # it doesn't have instabilities like sc
    # could divide the garment panels
    # and treat them as separate objects 
    # Would have to be a heirarchy.
    # cant figure out what order to do
    # the heirarcy. Inner layer colliding
    # with avatar would have to be master.
    pass


def edge_edge_spencer_model(coth):
    for e in boundary_edges:
        # if e1v and e2v are on opposite
        # sides of a boundary tri the edge is probably
        # slid past a boundary edge
        pass
    

def wrap_force(cloth, avatar, frame=0):
    
    move = closest_point_mesh(cloth, avatar)
    
    cloth.co += move * cloth.ob.MC_props.wrap_force
    #cloth.wrap_force = co - cloth.co
    
    #np.array(locs), np.array(faces), np.array(norms), np.array(cos)
    return    
    
    ob = cloth.ob
    m = ob.modifiers.new('MCWF', "SHRINKWRAP")
    m.target = avatar
    m.offset = avatar.MC_props.outer_margin
    m.wrap_method = 'TARGET_PROJECT'

    #bpy.context.scene.frame_current = frame

    bpy.ops.object.modifier_apply_as_shapekey({"object" : cloth.ob}, modifier='MCWF')
    co = get_co_shape(cloth.ob, key='MCWF')
    #cloth.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
    cloth.wrap_force = co - cloth.co
    
    cloth.ob.shape_key_remove(cloth.ob.data.shape_keys.key_blocks['MCWF'])
    

def surface_follow(cloth, avatar, value):
    """Used by p1 with surface deform for putting
    the arms down."""

    ob = cloth.ob
    m = ob.modifiers.new('SF', "SURFACE_DEFORM")
    m.target = avatar
    m.falloff = 16.0
    bpy.ops.object.surfacedeform_bind({"object" : ob}, modifier='SF')
        
    avk = avatar.data.shape_keys.key_blocks['Armature']
    avk.value = value

    if 'Armature.001' in avatar.data.shape_keys.key_blocks:
        avk2 = avatar.data.shape_keys.key_blocks['Armature.001']
        avk2.value = 1 - value
    
    bpy.context.scene.MC_props.interference = True
    

def best_sim(cloth, avatar):
    colliders = [o for o in bpy.data.objects if (o.MC_props.collider) & (cloth.ob != o)]
    
    if cloth.iterator == 0:
        cloth.sim_start_time = time.time()
        cloth.ob.MC_props.extra_bend_iters = 1
        cloth.ob.MC_props.velocity = 0.95
        cloth.ob.MC_props.sew_force = 0.001
        cloth.ob.MC_props.shrink_grow = 1.0
        cloth.ob.MC_props.gravity = -1.0
        cloth.ob.MC_props.wrap_force = 0.0
        cloth.ob.MC_props.self_collide_force = 0.5

    if cloth.iterator == 20:
        cloth.ob.MC_props.sew_force = .001
    if cloth.iterator == 40:
        cloth.ob.MC_props.sew_force = .002
    if cloth.iterator == 60:
        cloth.ob.MC_props.sew_force = .005
    if cloth.iterator == 80:
        cloth.ob.MC_props.sew_force = .1
        cloth.ob.MC_props.shrink_grow = .7
        cloth.ob.MC_props.gravity = 1.0
    
    #if cloth.iterator == 40:
        #cloth.velocity[:] = 0.0
        
    if cloth.iterator == 100:
        surface_follow(cloth, colliders[0], 0.8)
        cloth.ob.MC_props.sew_force = .2        
        cloth.ob.MC_props.gravity = 0.0
        
    if cloth.iterator == 160:
        surface_follow(cloth, colliders[0], 0.6)
        cloth.velocity[:] = 0.0
        
        cloth.ob.MC_props.bend_iters = 4
        cloth.ob.MC_props.gravity = -0.4
        cloth.ob.MC_props.sew_force = .5
        
    if cloth.iterator == 170:
        surface_follow(cloth, colliders[0], 0.4)
        
    if cloth.iterator == 180:
        surface_follow(cloth, colliders[0], 0.2)
        cloth.velocity[:] = 0.0

    if cloth.iterator == 190:
        surface_follow(cloth, colliders[0], 0.0)
        cloth.ob.MC_props.velocity = 0.98
    
    if cloth.iterator == 191:
        cloth.velocity[:] = 0.0
    
    if cloth.iterator > 190:
        if cloth.ob.MC_props.shrink_grow < 1:
            cloth.ob.MC_props.shrink_grow += 0.01
            
    if cloth.iterator == 220:
            cloth.ob.MC_props.bend_iters = 1
            cloth.ob.MC_props.bend = 0.5
            
    if cloth.iterator == 330:
        cloth.ob.MC_props.self_collide_margin = 0.03

    if cloth.iterator == 335:    
        cloth.ob.MC_props.continuous = False
        print("stopped sim")
        print("total time =", time.time() - cloth.sim_start_time)
        print("if it's messed up check the inner margin on the avatar. Remember.....")
        
    print("=========================")
    print(cloth.iterator, "iteration")
    print("=========================")


def pierce_collide(cloth):
    """Where edges pierce faces"""
    
    if not cloth.ob.MC_props.self_collide:
        update_v_norms(cloth) # because self collide runs it
    cloth.pierce_co = cloth.co[cloth.eidx]
    MC_pierce.detect_collisions(cloth)


def ob_collide(cloth):

    colliders = [o for o in bpy.data.objects if (o.MC_props.collider) & (cloth.ob != o)]
    if len(colliders) == 0:
        return
    c_check = True
    if len(colliders) != cloth.collider_count:
        Collider(cloth)
        c_check = False

    if cloth.ob.MC_props.wrap_force != 0:
        wrap_force(cloth, colliders[0])
    
    shift = 0
    f_shift = 0
    for i, c in enumerate(colliders):
        abco, proxy, prox = absolute_co(c)
        
        if abco.shape[0] != cloth.geo_check[i]:
            Collider(cloth)
            print('recalc colliders')
            return
        
        sh = abco.shape[0]

        cloth.total_co[shift: shift + sh] = abco# + surface_offset
        shift += sh
    
    sco = apply_transforms(cloth.ob, cloth.select_start)
    fco = apply_transforms(cloth.ob, cloth.co)        
    
    cloth.ob_co[:cloth.v_count] = sco
    cloth.ob_co[cloth.v_count:] = fco        

    cloth.OM = cloth.ob.MC_props.outer_margin
    update_ob_v_norms(cloth)

    ob_settings = not cloth.ob.MC_props.override_settings
    #cloth.OM = cloth.ob.MC_props.outer_margin
    cloth.static_threshold = cloth.ob.MC_props.static_friction * .0001            
    cloth.object_friction = cloth.ob.MC_props.oc_friction

    oms = [c.MC_props.outer_margin for c in colliders]
    ims = [c.MC_props.inner_margin - c.MC_props.outer_margin for c in colliders]
    frs = [c.MC_props.oc_friction for c in colliders]
    sfrs = [c.MC_props.static_friction  * .0001 for c in colliders]

    cloth.outer_margins = cloth.ob.MC_props.outer_margin
    cloth.inner_margins = cloth.ob.MC_props.inner_margin - cloth.ob.MC_props.outer_margin
    cloth.total_static[:] = cloth.static_threshold
    cloth.total_friction[:] = cloth.object_friction    
    
    if ob_settings:    
        #fcs = [len(p[1].polygons) for p in abc_prox]
        fcs = cloth.oc_tri_counts  #[len(p[1].polygons) for p in abc_prox]
        vcs = cloth.oc_v_counts  #[len(p[1].polygons) for p in abc_prox]
        
        f_shift = 0
        v_shift = 0

        for i in range(len(colliders)):
            cloth.total_margins[v_shift: v_shift+vcs[i]] = oms[i]
            cloth.total_inner_margins[v_shift: v_shift+vcs[i]] = ims[i]
            cloth.total_friction[f_shift: f_shift+fcs[i]] = frs[i]
            cloth.total_static[f_shift: f_shift+fcs[i]] = sfrs[i]
            f_shift = fcs[i]
            v_shift = vcs[i]
            
        cloth.outer_margins = cloth.total_margins
        cloth.inner_margins = cloth.total_inner_margins

    MC_object_collision.detect_collisions(cloth)
    cloth.last_co[:] = cloth.total_co


def spring_basic_no_sw(cloth):
        
    if cloth.ob.MC_props.p1_cloth:

        colliders = [o for o in bpy.data.objects if (o.MC_props.collider) & (cloth.ob != o)]
        best_sim(cloth, colliders[0])
        cloth.iterator += 1

        # for updating after moving the arms in p1
        if bpy.context.scene.MC_props.interference:
            bpy.context.scene.MC_props.interference = False

            bpy.ops.object.modifier_apply_as_shapekey({"object" : cloth.ob}, modifier='SF')
            co = get_co_shape(cloth.ob, key='SF')
            cloth.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
            cloth.ob.shape_key_remove(cloth.ob.data.shape_keys.key_blocks['SF'])

            refresh(cloth, skip=True)
            return
                
    cloth.select_start[:] = cloth.co
    feedback_val = cloth.ob.MC_props.feedback
    # start adding forces -------------------------

    inflate_and_wind(cloth)
    
    grav = np.array([0.0, 0.0, cloth.ob.MC_props.gravity * 0.001])
    w_grav = revert_rotation(cloth.ob, [grav])
    
    #cloth.velocity[:, 2] += cloth.ob.MC_props.gravity * 0.001
    cloth.velocity += w_grav
    cloth.co += cloth.velocity
    
    cloth.vel_zero[:] = cloth.co

    #if cloth.ob.MC_props.stretch > 0: # could add a cloth.do_stretch for the stretch vertex group if they are all zero...
    #if not cloth.ob.MC_props.self_collide: # put this after self collision
    rt_(num=None)
    
    if cloth.do_bend:
        if cloth.ob.MC_props.bend > 0:
            for i in range(cloth.ob.MC_props.bend_iters):
                abstract_bend(cloth)
                if i > 0:
                    update_pins_select_sew_surface(cloth)
    
    #rt_('bend spring time', skip=False, show=True)
    rt_('', skip=False, show=True)
    if cloth.ob.MC_props.stretch > 0:
        cloth.feedback[:] = cloth.co
        for i in range(cloth.ob.MC_props.stretch_iters):
            stretch_solve(cloth)
            sew_force(cloth)
            #if i > 0:
            update_pins_select_sew_surface(cloth)
        spring_move = cloth.co - cloth.feedback
        cloth.velocity += spring_move * feedback_val
    
    rt_(num='stretch time', skip=False, show=True)
    # sewing -------------------
    #sew_force(cloth) # no iterate so no: update_pins_and_select(cloth)
    # sewing -------------------
    
    # surface ------------------
    #surface_forces(cloth) # might need to refresh when iterating bend and stretch. Could put it in update_pins_and_select()
    # surface ------------------
    
    #v_move = cloth.co - cloth.vel_zero
    
    if cloth.ob.MC_props.detect_collisions:
        ob_collide(cloth)
    
    rt_(num='ob collide time')
    if cloth.ob.MC_props.self_collide:
        #if cloth.ob.data.is_editmode:
            #cloth.ob.update_from_editmode()
        cloth.OM = cloth.ob.MC_props.outer_margin
        update_v_norms(cloth)
        sc = MC_self_collision.detect_collisions(cloth)
    # -------------------------------------------
        rt_(num='self collisions sw', skip=False)
        extra_bend = True
        #extra_bend = False
        
        if cloth.ob.MC_props.p1_cloth:
            #if cloth.ob.MC_props.bend > 0:
            for i in range(cloth.ob.MC_props.extra_bend_iters):

                abstract_bend(cloth)
                    #sew_force(cloth)
        rt_(num='extra_bend', skip=False)
    #rt_(num='bend springs sw')
    
    if cloth.ob.MC_props.detangle: # cloth.ob.MC_props.pierce_collide:
        pierce_collide(cloth)
    
    if cloth.ob.MC_props.wrap_force != 0:
        if cloth.wrap_force is not None:    
            cloth.co += cloth.wrap_force * cloth.ob.MC_props.wrap_force

    update_pins_select_sew_surface(cloth) # also hooks

    v_move = cloth.co - cloth.vel_zero
    cloth.velocity += v_move
    cloth.velocity *= cloth.ob.MC_props.velocity

    cloth.velocity *= 1 - cloth.drag
    #cloth.velocity[:,2] += cloth.ob.MC_props.gravity * 0.001 # so after *= vel so it can still fall at zero vel
    #inflate_and_wind(cloth)
    
    # keep this !!! for static friction in MC_object_collision.py
    np.einsum('ij,ij->i', cloth.velocity, cloth.velocity, out=cloth.move_dist)
    # keep this !!!
    
    """
    The mass vertex group for velocity would be simple.
    Velocity gets multiplied by the vertex weight.
    Default would be one. That way we could have drag
    controlled by velocity.
    It should also make it eaiser for points to be pulled
    if they have low mass. Might want to seperate these
    forces...
    """


# update the cloth ---------------
def cloth_physics(ob, cloth):#, colliders):

    if ob.MC_props.cache_only:
        if ob.MC_props.cache:
            cloth.co = get_proxy_co(ob)
            cache(cloth)
            return

    if ob.MC_props.animated:
        ob.MC_props['current_cache_frame'] = bpy.context.scene.frame_current

    # If there is a proxy object will need to check that they match and issue
    #   warnings if there is a mismatch. Might want the option to regen the proxy object
    #   or adapt the cloth object to match so I can sync a patter change
    #   given you can change both now I have to also include logic to
    #   decide who is boss if both are changed in different ways.
    if cloth.target is not None:

        # if target is deleted while still referenced by pointer property
        if len(cloth.target.users_scene) == 0:
            ob.MC_props.target = None
            cloth.target = None # (gets overwritten by cb_target)
            cloth.target_undo = False
            cloth.target_geometry = None # (gets overwritten by cb_target)
            cloth.target_mode = 1
            return

        if cloth.target.data.is_editmode:
            if cloth.target_mode == 1 or cloth.target_undo:
                cloth.target_obm = get_bmesh(cloth.target)
                cloth.target_obm.verts.ensure_lookup_table()
                cloth.target_undo = False
            cloth.target_mode = None

        # target in object mode:
        else: # using else so it won't also run in edit mode
            pass

        dynamic_source = True # can map this to a prop or turn it on and off automatically if use is in a mode that makes it relevant.
        # If there is a target dynamic should prolly be on or if switching from active shape MC_source when in edit mode
        if dynamic_source:
            if not cloth.data.is_editmode: # can use bmesh prolly if not OBJECT mode.
                dg = cloth.dg
                if cloth.proxy is None:
                    cloth.proxy = cloth.target.evaluated_get(dg)
                #proxy = col.ob.to_mesh(bpy.context.evaluated_depsgraph_get(), True, calc_undeformed=False)
                #proxy = col.ob.to_mesh() # shouldn't need to use mesh proxy because I'm using bmesh

                co_overwrite(cloth.proxy, cloth.target_co)

    if cloth.shape_update:

        index = ob.data.shape_keys.key_blocks.find('MC_current')
        if cloth.ob.active_shape_key_index == index:
            refresh(cloth)
            cloth.shape_update = False

        if not cloth.ob.data.is_editmode:
            if cloth.shape_update:
                refresh(cloth)
                cloth.shape_update = False

    if ob.data.is_editmode:
        # prop to go into user preferences. (make it so it won't run in edit mode)
        if not bpy.context.scene.MC_props.run_editmode:
            return
        
        #index = ob.data.shape_keys.key_blocks.find('MC_current')
        #ob.active_shape_key_index = index

        
        #bpy.ops.transform.translate(value=(0.0, 0.0, 0.0))
        cloth.ob.update_from_editmode()
        #cloth.ob.data.update_gpu_tag()

        #for area in bpy.context.window.screen.areas:
            #if area.type == 'VIEW_3D':
                #area.tag_redraw()
                # bmesh gets removed when someone clicks on MC_current shape"
        try:
            cloth.obm.verts
        except:
            cloth.obm = get_bmesh(ob, refresh=True)
            print('ran bmesh update')
        #if cloth.update_lookup:
            #cloth.obm.verts.ensure_lookup_table()
            #cloth.update_lookup = False

        # If we switched to edit mode or started in edit mode:
        if cloth.mode == 1 or cloth.undo:
            cloth.obm = get_bmesh(ob, refresh=True)
            cloth.undo = False
        cloth.mode = None
        # -----------------------------------

        # detect changes in geometry and update
        if cloth.obm is None:
            cloth.obm = get_bmesh(ob)

        if not cloth.ob.MC_props.cache_only:
            same, faces = detect_changes(cloth.geometry, cloth.obm)
            if faces: # zero faces in mesh do nothing
                return
            if not same:
                # for pinning
                print("DANGER!!!!!!!!!!!!!!!!!!")
                
                index = ob.active_shape_key_index
                basis = ob.data.shape_keys.key_blocks.find('Basis')
                
                ob.active_shape_key_index = basis
                refresh(cloth)
                ob.active_shape_key_index = index
                return

        # if we switch to a different shape key in edit mode:
        if not cloth.ob.MC_props.cache_only:
            index = ob.data.shape_keys.key_blocks.find('MC_current')
            if ob.active_shape_key_index != index:
                cloth.shape_update = True
                return

        
        #cloth.co = np.array([v.co for v in cloth.obm.verts])

        if not cloth.ob.MC_props.cache_only:
            
            # This: -------------------
            #cloth.ob.update_from_editmode()
            cloth.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_get('co', cloth.co.ravel())
            # or:
            #cloth.co = np.array([v.co for v in cloth.obm.verts], dtype=np.float32)
            # -------------------------

            cloth.selected[:] = False            
            if bpy.context.scene.MC_props.pause_selected:
                cloth.ob.data.vertices.foreach_get('select', cloth.selected)


            #area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
            #space = next(space for space in area.spaces if space.type == 'VIEW_3D')
            #space.viewport_shade = 'RENDERED'  # set the viewport shading


        """ =============== FORCES EDIT MODE ================ """
        # FORCES FORCES FORCES FORCES FORCES
        if cloth.ob.MC_props.play_cache:
            play_cache(cloth)
            return

        for i in range(cloth.ob.MC_props.sub_frames):
            spring_basic_no_sw(cloth)

        # FORCES FORCES FORCES FORCES FORCES
        """ =============== FORCES EDIT MODE ================ """

        # set coords to current edit mode bmesh
        cloth.obm.verts.ensure_lookup_table()
        for i, j in enumerate(cloth.co):
            cloth.obm.verts[i].co = j

        if cloth.ob.MC_props.cache:
            if cloth.ob.MC_props.internal_cache:    
                np_co_to_text(cloth.ob, cloth.co, rw='w')
            else:
                cache(cloth)

        #update_shading = True
        #update_shading = False
        if bpy.context.scene.MC_props.update_shading: # for live shading update        
            obm = cloth.obm
            #obm.faces.ensure_lookup_table()
            #obm.edges.ensure_lookup_table()
            vsel = obm.verts[0].select
            #fsel = obm.faces[0].select
            #esel = obm.edges[0].select
            obm.verts[0].select = True
            #obm.faces[0].select = True
            #obm.edges[0].select = True
            bpy.ops.mesh.hide()
            bpy.ops.mesh.reveal()
            #obm.faces[0].select = fsel
            #obm.edges[0].select = esel
            obm.verts[0].select = vsel

        return

    # switched out of edit mode
    if cloth.mode is None:
        cloth.mode = 1

        #print("running here")
        #refresh(cloth, skip=True)
        #index = ob.data.shape_keys.key_blocks.find('MC_current')
        #ob.active_shape_key_index = index
        #cloth.obm = get_bmesh(ob) # if I don't do this I can get a bug when I change geometry then pop in and out of edit mode
        
        if not cloth.ob.MC_props.cache_only:
            update_groups(cloth, cloth.obm)

    # OBJECT MODE ====== :
    """ =============== FORCES OBJECT MODE ================ """
    # FORCES FORCES FORCES FORCES
    if cloth.ob.MC_props.play_cache:
        play_cache(cloth)
        return

    for i in range(cloth.ob.MC_props.sub_frames):
        spring_basic_no_sw(cloth)

    # FORCES FORCES FORCES FORCES
    """ =============== FORCES OBJECT MODE ================ """

    # updating the mesh coords -----------------@@
    ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set("co", cloth.co.ravel())
    cloth.ob.data.update()

    if cloth.ob.MC_props.cache:
        if cloth.ob.MC_props.internal_cache:    
            np_co_to_text(cloth.ob, cloth.co, rw='w')
        else:
            cache(cloth)

# update the cloth ---------------
def update_cloth(type=0):

    # run from either the frame handler or the timer
    if type == 0:
        cloths = [i[1] for i in MC_data['cloths'].items() if i[1].ob.MC_props.continuous]
        if len(cloths) == 0:
            bpy.app.timers.unregister(cloth_main)

    if type == 1:
        cloths = [i[1] for i in MC_data['cloths'].items() if i[1].ob.MC_props.animated]
        if len(cloths) == 0:
            install_handler(continuous=True, clear=False, clear_anim=True)

    for cloth in cloths:
        cloth_physics(cloth.ob, cloth)#, colliders)

# ^                                                          ^ #
# ^                 END update the cloth                     ^ #
# ============================================================ #


def refresh(cloth, skip=False):
    
    ob = cloth.ob
    
    cloth.wrap_force = None
    
    if ob.data.is_editmode:
        ob.update_from_editmode()
    
    cloth.p1 = False
    if not skip:    
        cloth.iterator = 0
    # target ----------
    cloth.target = None # (gets overwritten by def cb_target)
    cloth.current_cache_frame = 1 # for the cache continuous playback
    cloth.shape_update = False

    # for detecting mode changes
    cloth.mode = 1
    if ob.data.is_editmode:
        cloth.mode = None
    cloth.undo = False

    cloth.v_count = len(ob.data.vertices)
    v_count = cloth.v_count
    cloth.obm = get_bmesh(ob, refresh=True)
    
    #noise = np.array(np.random.random((cloth.v_count, 3)) * 0.00001, dtype=np.float32)

    cloth.co = get_co_edit(ob)# + noise
    
    if not skip:
        # slowdowns ------------------
        manage_vertex_groups(cloth)
        # slowdowns ------------------

    cloth.move_dist = np.zeros(v_count, dtype=np.float32) # used by static friction
    cloth.pin_arr = np.copy(cloth.co)
    cloth.geometry = get_mesh_counts(ob, cloth.obm)
    
    if not skip:
        # slowdowns ------------------
        cloth.sew = False
        cloth.sew_springs = get_sew_springs(cloth)
        sew_v_fancy(cloth)
        # slowdowns ------------------
    
    if False: # need to check if this works. Used by surface forces. search for " rev " in the collision module    
        cloth.wco = np.copy(cloth.co)
        apply_in_place(cloth.ob, cloth.wco)

    cloth.select_start = np.copy(cloth.co)
    cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32)
    
    if cloth.ob.data.is_editmode:    
        cloth.selected = np.array([v.select for v in cloth.obm.verts])
    else:
        cloth.selected = np.zeros(cloth.co.shape[0], dtype=np.bool) # keep False if in object mode
    
    cloth.velocity = np.zeros_like(cloth.co)
    cloth.vel_zero = np.zeros_like(cloth.co)
    cloth.feedback = np.zeros_like(cloth.co)
    cloth.stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.bend_stretch_array = np.zeros(cloth.co.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.total_tridex = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.measure_dot = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.measure_length = np.zeros(cloth.basic_v_fancy.shape[0], dtype=np.float32) # for calculating the weights of the mean
    cloth.measure_cv = np.zeros((cloth.basic_v_fancy.shape[0], 3), dtype=np.float32) # for calculating the weights of the mean
    
    if not skip:    
        cloth.vdl = stretch_springs_basic(cloth, cloth.target)

    if not skip:
        #if doing_self_collisions:
        cloth.tridex, triobm = get_tridex_2(ob)
        cloth.triobm = triobm

        # for offset_cloth_tris:
        cloth.v_norms = np.empty((cloth.co.shape[0], 3), dtype=np.float32)
        cloth.v_norm_indexer1 = np.array(np.hstack([[f.index for f in v.link_faces] for v in triobm.verts]), dtype=np.int32)
        cloth.v_norm_indexer = np.array(np.hstack([[v.index] * len(v.link_faces) for v in triobm.verts]), dtype=np.int32)
        # ---------------------        
    
    cloth.turbulence = np.random.rand(cloth.tridex.shape[0])
    cloth.turbulence_2 = np.random.rand(cloth.tridex.shape[0])
    cloth.random_dir = np.random.rand(3)
    cloth.random_dir_2 = np.random.rand(3)
    cloth.turb_count = 1

    # old self collisions
    cloth.sc_edges = get_sc_edges(ob, fake=True)
    cloth.sc_eidx = np.arange(len(ob.data.vertices), dtype=np.int32)
    cloth.sc_indexer = np.arange(cloth.tridex.shape[0], dtype=np.int32)
    cloth.tris_six = np.empty((cloth.tridex.shape[0], 6, 3), dtype=np.float32)
    cloth.sc_co = np.empty((cloth.co.shape[0] * 2, 3), dtype=np.float32)

    # pierce data
    if not skip:
        cloth.eidx = get_sc_edges(ob)
        
        cloth.pierce_co = np.empty((cloth.eidx.shape[0], 2, 3), dtype=np.float32)
        #cloth.pierce_co2 = np.empty((cloth.eidx.shape[0] * 2, 3), dtype=np.float32)
        
        c = cloth.eidx.shape[0]
        cloth.pc_edges = np.empty((c, 2), dtype=np.int32)
        idx = np.arange(c * 2, dtype=np.int32)
        cloth.pc_edges[:, 0] = idx[:c]
        cloth.pc_edges[:, 1] = idx[c:]
        cloth.peidx = np.arange(cloth.eidx.shape[0])
        
        # boundary edgs:
        cloth.boundary_bool = np.array([[e.is_boundary for e in t.edges] for t in cloth.triobm.faces], dtype=np.bool)
        cloth.boundary_tris = np.array([np.any(b) for b in cloth.boundary_bool], dtype=np.bool)
        cloth.bt_edges = np.array([[[e.verts[0].index, e.verts[1].index] for e in t.edges] for t in cloth.triobm.faces], dtype=np.int32)
        #cloth.bt_edges
        #print(cloth.bt_edges)
        
        # I hate it when this happens:
        # cloth.bt_edges[cloth.boundary_tris][ וַעֲוֺנֹתָם הוּא יִסְבֹּֽל]
        # Hebrew does not index numpy arrays (copy and paste while mixing Hebrew study and python...)
        # print(cloth.bt_edges[cloth.boundary_tris][cloth.boundary_bool[cloth.boundary_tris]])
        

        
        
        
    # for that p1 experiment thingy with boundary edge to object collisions
    for i, j in enumerate(cloth.obm.verts):
        if j.is_boundary:
            cloth.group_surface_offset[i] = -0.1

    Collider(cloth)



# ============================================================ #
#                      Manage handlers                         #
#                                                              #
# handler ------------------
@persistent
def undo_frustration(scene):
    # someone might edit a mesh then undo it.
    # in this case I need to recalc the springs and such.
    # using id props because object memory adress changes with undo

    # find all the cloth objects in the scene and put them into a list
    cloths = [i for i in bpy.data.objects if i.MC_props.cloth]
    if len(cloths) < 1:
        return
    # throw an id prop on there.
    for i in cloths:
        cloth = MC_data['cloths'][i['MC_cloth_id']]
        cloth.ob = i
        # update for meshes in edit mode
        cloth.undo = True
        cloth.target_undo = True
        try:
            cloth.obm.verts
        except:
            cloth.obm = get_bmesh(cloth.ob)

        if not detect_changes(cloth.geometry, cloth.obm)[0]:
            cloth.springs, cloth.v_fancy, cloth.e_fancy, cloth.flip = get_springs(cloth)

    fun = ["your shoe laces", "something you will wonder about but never notice", "something bad because it loves you", "two of the three things you accomplished at work today", "knots in the threads that hold the fabric of the universe", "a poor financial decision you made", "changes to your online dating profile", "your math homework", "everything you've ever accomplished in life", "something you'll discover one year from today", "the surgery on your cat", "your taxes", "all the mistakes you made as a child", "the mess you made in the bathroom", "the updates to your playstation 3", "nothing! Modeling Cloth makes no mistakes!", "your last three thoughts and you'll never know what they were", "the damage done to the economy by overreaction to covid"]
    msg = "Modeling Cloth undid " + fun[MC_data["iterator"]]
    print(msg)
    #bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
    MC_data['iterator'] += 1
    if MC_data['iterator'] == len(fun):
        MC_data['iterator'] = 0


# handler ------------------
@persistent
def cloth_main(scene=None):
    """Runs the realtime updates"""

    total_time = T()

    kill_me = []
    # check for deleted cloth objects
    for id, val in MC_data['cloths'].items():
        try:
            val.ob.data
        except:
            # remove from dict
            kill_me.append(id)

    for i in kill_me:
        del(MC_data['cloths'][i])
        print('killed wandering cloths')

    kill_me = []
    # check for deleted collider objects
    if False:    
        for id, val in MC_data['colliders'].items():
            try:
                val.ob.data
            except:
                # remove from dict
                kill_me.append(id)

        for i in kill_me:
            del(MC_data['colliders'][i])
            print('killed wandering colliders')

    # run the update -------------
    type = 1 # frame handler or timer continuous
    if scene is None:
        type=0

    delay = bpy.context.scene.MC_props.delay

    update_cloth(type) # type 0 continuous, type 1 animated

    # auto-kill
    auto_kill = True
    auto_kill = False
    if auto_kill:
        if MC_data['count'] == 20:
            print()
            print('--------------')
            print('died')
            return

    return delay


# handler ------------------
def install_handler(continuous=True, clear=False, clear_anim=False):
    """Run this when hitting continuous update or animated"""
    # clean dead versions of the animated handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.frame_change_post])
    booly = [i == 'cloth_main' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.frame_change_post[i])
        print("deleted handler ", i)
    if clear_anim:
        print('ran clear anim handler')
        return

    # clean dead versions of the timer
    if bpy.app.timers.is_registered(cloth_main):
        bpy.app.timers.unregister(cloth_main)

    # for removing all handlers and timers
    if clear:
        print('ran clear handler')
        return

    MC_data['count'] = 0

    if np.any([i.MC_props.continuous for i in bpy.data.objects]):
        # continuous handler
        bpy.app.timers.register(cloth_main, persistent=True)
        #bpy.app.timers.register(funky.partial(cloth_main, delay, kill, T), first_interval=delay_start, persistent=True)
        return

    # animated handler
    if np.any([i.MC_props.animated for i in bpy.data.objects]):
        bpy.app.handlers.frame_change_post.append(cloth_main)


# ^                                                          ^ #
# ^                      END handler                         ^ #
# ============================================================ #


# ============================================================ #
#                    callback functions                        #
#                                                              #


# calback functions ---------------
def oops(self, context):
    # placeholder for reporting errors or other messages
    return


def open_folder(path):
    import subprocess
    import sys

    if sys.platform == 'darwin':
        subprocess.check_call(['open', '--', path])
    elif sys.platform == 'linux2':
        subprocess.check_call(['gnome-open', '--', path])
    elif sys.platform == 'win32':
        subprocess.check_call(['explorer', path])


# calback functions ---------------
# object:
def cb_cache_only(self, context):

    ob = self.id_data

    if self.cache_only:
        self.cloth = True
        #self.cache = True
        self.animated = True
        self.cache_name = ob.name
        return

    self.cache = False
    self.cloth = False
    self.animated = False


def check_file_path(ob):
    """Check if the current filepath is legit"""
    self = ob.MC_props
    custom = pathlib.Path(self.cache_folder)
    mc_path = custom.joinpath('MC_cache_files')
    name = self.cache_name
    final_path = mc_path.joinpath(name)
    valid = final_path.exists()
    return valid, final_path


def cb_cache(self, context):
    """Manage files and paths for saving cache."""
    if self.cache:
        self['play_cache'] = False
        # Might want to overwrite while playing
        #   back with partial influence and running cloth sim

    ob = self.id_data

    cloth = get_cloth(ob)

    # set path to blender path by default
    path = pathlib.Path(bpy.data.filepath).parent #.parent removes .blend file
    if path == '':
        path = os.path.expanduser("~/Desktop")
        self['cache_folder'] = path

    if ob.MC_props.cache_desktop:
        path = os.path.expanduser("~/Desktop")
        self['cache_folder'] = path
    else:
        self['cache_folder'] = str(path) # so it switches back when turning of desktop

    mc_path = pathlib.Path(path).joinpath('MC_cache_files')

    # overwrite if user path is valid:
    custom = pathlib.Path(self.cache_folder)
    if not custom.exists():
        # Report Error
        msg = '"' + self.cache_folder + '"' + " is not a valid filepath. Switching to .blend location Desktop if blend file is not saved"
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        self['cache_folder'] = str(path)
    else:
        mc_path = custom.joinpath('MC_cache_files')

    # create dir if it doesn't exist
    if not mc_path.exists():
        mc_path.mkdir()

    name = str(ob['MC_cloth_id'])

    custom_name = self.cache_name
    if custom_name != "Mr. Purrkins The Deadly Cyborg Kitten":
        if custom_name != '':
            name = custom_name

    self['cache_name'] = name

    final_path = mc_path.joinpath(name)

    # create dir if it doesn't exist
    if not final_path.exists():
        final_path.mkdir()

    cloth.cache_dir = final_path
    return


def cb_cache_playback(self, context):
    ob = self.id_data
    cloth = get_cloth(ob)
    
    if self.play_cache:
        self.animated = True
    
    self['cache'] = False

    if self.cache_only:
        if self.play_cache:
            self['cache'] = False

            if ob.data.shape_keys == None:
                ob.shape_key_add(name='Basis')

            keys = ob.data.shape_keys.key_blocks
            index = ob.data.shape_keys.key_blocks.find('MC_current')
            active = ob.active_shape_key_index
            cloth.key_values = {'MC_active_idx_pre_cache': active}
            for k in keys:
                cloth.key_values[k.name] = k.value

            for k in keys:
                k.value = 0

            keys = ob.data.shape_keys.key_blocks
            if 'cache_key' not in keys:
                ob.shape_key_add(name='cache_key')

            keys['cache_key'].value = 1.0
            index = ob.data.shape_keys.key_blocks.find('cache_key')
            ob.active_shape_key_index = index
            return

        if ob.data.shape_keys is not None:
            keys = ob.data.shape_keys.key_blocks
            if len(keys) >= cloth.key_values['MC_active_idx_pre_cache']:
                ob.active_shape_key_index = cloth.key_values['MC_active_idx_pre_cache']

            if 'cache_key' in keys:
                keys['cache_key'].value = 0

            for k, v in cloth.key_values.items():
                if k in keys:
                    keys[k].value = v


def cb_current_cache_frame(self, context):
    """Keep track of the current saved cache frame."""
    ob = self.id_data

    cloth = get_cloth(ob)
    cloth.cache_dir = check_file_path(ob)[1]
    cloth.current_cache_frame = self.current_cache_frame
    play_cache(cloth, cb=True)


def cb_detect_collisions(self, context):
    ob = self.id_data
    print(ob.name, ' cloth object set to check for collisions')


def cb_collider(self, context):
    """Set up object as collider"""

    ob = self.id_data
    if ob.type != "MESH":
        self['collider'] = False

        # Report Error
        msg = "Must be a mesh. Collisions with non-mesh objects can create black holes potentially destroying the universe."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return

    cloths = [o for o in bpy.data.objects if o.MC_props.cloth]
    cloth_classes = [MC_data['cloths'][o['MC_cloth_id']] for o in cloths]

    for cc in cloth_classes:
        Collider(cc)    
    

# calback functions ---------------
# object:
def cb_cloth(self, context):
    """Set up object as cloth"""

    ob = self.id_data
    self = ob.MC_props
    
    if ob.data.is_editmode:
        ob.update_from_editmode()    
    
    # do I really need this???
    #ob.data.update() # otherwise changes to geometry then trying popping out of edit mode messes up
    
    if self.cache_only:
        cloth = create_instance(ob=ob)
        id_number = ob.name
        MC_data['cloths'][id_number] = cloth
        ob['MC_cloth_id'] = id_number
        return

    # set the recent object for keeping settings active when selecting empties
    recent = MC_data['recent_object']

    if ob.type != "MESH":
        if recent is not None:
            ob = recent
        else:
            self['cloth'] = False

            # Report Error
            msg = "Must be a mesh. Non-mesh objects make terrible shirts."
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return

    if len(ob.data.polygons) < 1:
        self['cloth'] = False

        # Report Error
        msg = "Must have at least one face. Faceless meshes are creepy."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        return

    # The custom object id is the key to the cloth instance in MC_data
    if self.cloth:
        # creat shape keys and set current to active

        reset_shapes(ob)
        index = ob.data.shape_keys.key_blocks.find('MC_current')
        ob.active_shape_key_index = index

        cloth = create_instance(ob=ob)

        # recent_object allows cloth object in ui
        #   when selecting empties such as for pinning.
        MC_data['recent_object'] = bpy.context.object

        # use an id prop so we can find the object after undo
        if False: # using time.time() for id
            d_keys = [i for i in MC_data['cloths'].keys()]
            id_number = 0
            if len(d_keys) > 0:
                id_number = max(d_keys) + 1
                print("created new cloth id", id_number)
            id_number = ob.name # Could create problems

        if 'MC_cloth_id' in ob:
            id_number = ob['MC_cloth_id']
        else:
            id_number = time.time()
        
        MC_data['cloths'][id_number] = cloth
        ob['MC_cloth_id'] = id_number
        #cb_cache(self, context) # if a cache exists at the specified file location. So the usere doesn't have to toggle the property if the file is a valid cache.
        #print('created instance')
        return

    # when setting cloth to False
    if ob['MC_cloth_id'] in MC_data['cloths']:
        del(MC_data['cloths'][ob['MC_cloth_id']])
        del(ob['MC_cloth_id'])
        # recent_object allows cloth object in ui
        #   when selecting empties such as for pinning
        MC_data['recent_object'] = None
        ob.MC_props['continuous'] = False
        ob.MC_props['animated'] = False


@persistent
def reload_from_save(scene=None):
    print("Ran MC load handler")

    cobs = [ob for ob in bpy.data.objects if ob.MC_props.cloth]
    for c in cobs:
        cb_cloth(c, bpy.context)
        cb_continuous(c, bpy.context)
        cb_animated(c, bpy.context)
        

# calback functions ----------------
# object:
def cb_continuous(self, context):
    """Turn continuous update on or off"""
    install_handler(continuous=True)


# calback functions ----------------
# object:
def cb_dense(self, context):
    ob = self.id_data
    print('Ran cb_dense. Did nothing.')


# calback functions ----------------
# object:
def cb_quad_bend(self, context):
    ob = self.id_data
    print('Ran cb_quad_bend. Did nothing.')


# calback functions ----------------
# object:
def cb_animated(self, context):
    """Turn animated update on or off"""
    install_handler(continuous=False) # deletes handler when false.

    if not self.animated:
        return

    # updates groups when we toggle "Animated"
    ob = self.id_data
    cloth = get_cloth(ob)
    if self.cache_only:
        cloth.co = get_proxy_co(ob)
        return

    if ob.data.is_editmode:
        index = ob.data.shape_keys.key_blocks.find('MC_current')
        if ob.active_shape_key_index == index:
            cloth.co = np.array([v.co for v in cloth.obm.verts], dtype=np.float32)
            update_groups(cloth, cloth.obm)
            return
        ob.update_from_editmode()

    cloth.co = get_co_shape(ob, key='MC_current')


# calback functions ----------------
# object:
def cb_target(self, context):
    """Use this object as the source target"""

    # if the target object is deleted while an object is using it:
    if bpy.context.object is None:
        return

    # setting the property normally
    ob = bpy.context.object

    cloth = MC_data["cloths"][ob['MC_cloth_id']]

    # kill target data
    if self.target is None:
        cloth.target = None
        return

    # kill target data
    cloth.proxy = bpy.context.evaluated_get(self.target)
    same = compare_geometry(ob, proxy, obm1=None, obm2=None, all=False)
    if not same:
        msg = "Vertex and Face counts must match. Sew edges don't have to match."
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
        self.target = None
        cloth.target = None
        return

    # Ahh don't kill me I'm a valid target!
    cloth.target = self.target
    cloth.target_geometry = get_mesh_counts(cloth.target)
    cloth.target_co = get_co_mode(cloth.target)


# calback functions ----------------
# object:
def cb_reset(self, context):
    """RESET button"""
    ob = MC_data['recent_object']
    if ob is None:
        ob = bpy.context.object
    cloth = MC_data["cloths"][ob['MC_cloth_id']]
    #RESET(cloth)
    self['reset'] = False


# calback functions ----------------
# scene:
def cb_pause_all(self, context):
    print('paused all')


# calback functions ----------------
# scene:
def cb_play_all(self, context):
    print('play all')


# calback functions ----------------
# scene:
def cb_duplicator(self, context):
    # DEBUG !!!
    # kills or restarts the duplicator/loader for debugging purposes
    if not bpy.app.timers.is_registered(duplication_and_load):
        bpy.app.timers.register(duplication_and_load)
        return

    bpy.app.timers.unregister(duplication_and_load)
    print("unloaded dup/load timer")

# ^                                                          ^ #
# ^                 END callback functions                   ^ #
# ============================================================ #


# ============================================================ #
#                     create properties                        #
#                                                              #

# create properties ----------------
# object:
class McPropsObject(bpy.types.PropertyGroup):

    p1_cloth:\
    bpy.props.BoolProperty(name="p1 cloth", description="we are running in p1", default=False)

    simple_sew:\
    bpy.props.BoolProperty(name="Simple Sew", description="Basic sew springs for p1", default=False)

    is_hook:\
    bpy.props.BoolProperty(name="Hook Object", description="This object is used as a hook", default=False)

    hook_index:\
    bpy.props.IntProperty(name="Hook Index", description="Vert index for hook", default= -1)

    self_collide:\
    bpy.props.BoolProperty(name="Self Collision", description="Self collisions (Hopefully preventing self intersections. Fingers crossed.)", default=False)

    self_collide_margin:\
    bpy.props.FloatProperty(name="Self Collision Margin", description="Self collision margin", default=0.02, min=0, precision=3)

    detangle:\
    bpy.props.BoolProperty(name="Detangle Self Collision", description="Work Out Failed Self collisions (Hopefully fixing self collide failures. Fingers crossed.)", default=False)

    new_self_margin:\
    bpy.props.FloatProperty(name="New Self Margin", description="New Self collision margin", default=0.02, min=0, precision=3)
    
    self_collide_force:\
    bpy.props.FloatProperty(name="Self Collision Force", description="Self collision force", default=0.5, precision=3)

    #sc_vel_damping:\
    #bpy.props.FloatProperty(name="Self Collision Velocity Damping", description="Self self collisions reduces velocity", default=1.0, precision=3)

    sc_friction:\
    bpy.props.FloatProperty(name="Self Collision Friction", description="Self self collisions friction", default=0.5, soft_min=0, soft_max=1, precision=3)

    shrink_grow:\
    bpy.props.FloatProperty(name="Shrink/Grow", description="Change the target size", default=1.0, soft_min=0, soft_max=1000, precision=3)

    wrap_force:\
    bpy.props.FloatProperty(name="Wrap Force", description="Cloth moves towards colliders like shrinkwrap", default=0.0, soft_min=0, soft_max=1, precision=3)

    sc_box_max:\
    bpy.props.IntProperty(name="SC Box Max", description="Max number of sets in an octree box", default=150, min=10)

    collider:\
    bpy.props.BoolProperty(name="Collider", description="Cloth objects collide with this object", default=False, update=cb_collider)

    outer_margin:\
    bpy.props.FloatProperty(name="Outer Margin", description="Distance from surface of collisions on positive normal side", default=0.01, precision=3)

    inner_margin:\
    bpy.props.FloatProperty(name="Inner Margin", description="Points within this distance on the negative side of the normal will be pushed out", default=0.01, precision=3)

    detect_collisions:\
    bpy.props.BoolProperty(name="Detect Collisions", description="This cloth object checks for collisions", default=True, update=cb_detect_collisions)

    override_settings:\
    bpy.props.BoolProperty(name="Override Settings", description="Override object settings", default=False)

    oc_friction:\
    bpy.props.FloatProperty(name="Object Collision Friction", description="Object collision friction", default=0.5, soft_min=0, soft_max=1, precision=3)

    static_friction:\
    bpy.props.FloatProperty(name="Object Collision Static Friction", description="Static friction threshold", default=0.1, soft_min=0, precision=3)

    cloth:\
    bpy.props.BoolProperty(name="Cloth", description="Set this as a cloth object", default=False, update=cb_cloth)

    # handler props for each object
    continuous:\
    bpy.props.BoolProperty(name="Continuous", description="Update cloth continuously", default=False, update=cb_continuous)

    animated:\
    bpy.props.BoolProperty(name="Animated", description="Update cloth only when animation is running", default=False, update=cb_animated)

    target:\
    bpy.props.PointerProperty(type=bpy.types.Object, description="Use this object as the target for stretch and bend springs", update=cb_target)

    # Forces
    gravity:\
    bpy.props.FloatProperty(name="Gravity", description="Strength of the gravity", default=0.0, min=-1000, max=1000, soft_min= -10, soft_max=10, precision=3)

    velocity:\
    bpy.props.FloatProperty(name="Velocity", description="Maintains Velocity", default=0.98, min= -1000, max=1000, soft_min= 0.0, soft_max=1, precision=3)

    wind:\
    bpy.props.FloatProperty(name="Wind", description="This Really Blows", default=0.0, min= -1000, max=1000, soft_min= -10.0, soft_max=10.0, precision=3)

    # stiffness
    feedback:\
    bpy.props.FloatProperty(name="Feedback", description="Extrapolate for faster solve", default=.5, min= -1000, max=1000, soft_min= 0.0, soft_max=1, precision=3)

    stretch_iters:\
    bpy.props.IntProperty(name="Iters", description="Number of iterations of cloth solver", default=2, min=0, max=1000)#, precision=1)

    sub_frames:\
    bpy.props.IntProperty(name="Sub Frames", description="Number of sub frames between display", default=1, min=0, max=1000)#, precision=1)

    stretch:\
    bpy.props.FloatProperty(name="Stretch", description="Strength of the stretch springs", default=1, min=0, max=10, soft_min= 0, soft_max=1, precision=3)

    push:\
    bpy.props.FloatProperty(name="Push", description="Strength of the push springs", default=1, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    bend_iters:\
    bpy.props.IntProperty(name="Bend Iters", description="Number of iterations of bend springs", default=2, min=0, max=1000)#, precision=1)

    extra_bend_iters:\
    bpy.props.IntProperty(name="Extra Bend Iters", description="Extra bend after self collide for p1", default=0, min=0, max=1000)#, precision=1)

    bend:\
    bpy.props.FloatProperty(name="Bend", description="Strength of the bend springs", default=1, min=0, max=10, soft_min= 0, soft_max=1, precision=3)

    dense:\
    bpy.props.BoolProperty(name="Dense", description="Default vertex weights are set to 0. For dense meshes so we can reduce setup calculations", default=False, update=cb_dense)

    quad_bend:\
    bpy.props.BoolProperty(name="Quad Bend Springs", description="Calculate bend springs on a bmesh with joined triangles", default=False, update=cb_quad_bend)

    # Sewing
    sew_force:\
    bpy.props.FloatProperty(name="Sew Force", description="Shrink Sew Edges", default=0.1, min=0, max=1, soft_min= -100, soft_max=100, precision=3)

    # Sewing
    target_sew_length:\
    bpy.props.FloatProperty(name="Target Sew Length", description="Shrink Sew Edges to this Length", default=0.0, min=0, precision=3)

    surface_follow_selection_only:\
    bpy.props.BoolProperty(name="Use Selected Faces", description="Bind only to selected faces", default=False)

    # Vertex Groups
    vg_pin:\
    bpy.props.FloatProperty(name="Pin", description="Pin Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    vg_drag:\
    bpy.props.FloatProperty(name="Drag", description="Drag Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    vg_surface_follow:\
    bpy.props.FloatProperty(name="Surface Follow", description="Surface Follow Vertex Group", default=0, min=0, max=1, soft_min= -2, soft_max=2, precision=3)

    # Edit Mode
    cloth_grab:\
    bpy.props.BoolProperty(name="Cloth Grab", description="Only move cloth during modal grab", default=False)

    # Cache
    cache:\
    bpy.props.BoolProperty(name="Cloth Cache", description='Cache animation when running "Animated" or "Continuous"', default=False, update=cb_cache)

    cache_only:\
    bpy.props.BoolProperty(name="Cache Only", description='Cache vertex coordinates for an evaluated mesh without generating cloth data', default=False, update=cb_cache_only)

    overwrite_cache:\
    bpy.props.BoolProperty(name="Overwrite Cache", description="Save over existing frames", default=False)

    internal_cache:\
    bpy.props.BoolProperty(name="Internal Cache", description="Save cache in blend file", default=False)

    cache_desktop:\
    bpy.props.BoolProperty(name="Cache To Desktop", description='Save the cache files to desktop to they are easy to find', update=cb_cache)

    cache_interpolation:\
    bpy.props.BoolProperty(name="Cache Interpolate", description='Interpolate mesh shape between cached frames.', default=True, update=cb_cache)


    # set the default path
    if False:
        path = bpy.data.filepath
        if path == '':
            path = os.path.expanduser("~/Desktop")

    path = os.path.expanduser("~/Desktop")
    mc_path = str(pathlib.Path(path).parent)#.joinpath('MC_cache_files')

    cache_folder:\
    bpy.props.StringProperty(name="Cache Folder", description="Directory for saving cache files", default=mc_path, update=cb_cache)

    cache_name:\
    bpy.props.StringProperty(name="Custom Name", description="Custom name for saving multiple cache files", default="Mr. Purrkins The Deadly Cyborg Kitten", update=cb_cache)

    play_cache:\
    bpy.props.BoolProperty(name="Play Cache", description="Play the cached animation", default=False, update=cb_cache_playback)

    start_frame:\
    bpy.props.IntProperty(name="Start Frame", description="Start cache on this frame", default=1)#, update=cb_cache)

    end_frame:\
    bpy.props.IntProperty(name="End Frame", description="End cache on this frame", default=250)#, update=cb_cache)

    current_cache_frame:\
    bpy.props.IntProperty(name="End Frame", description="End cache on this frame", default=1, update=cb_current_cache_frame)

    cloth_off:\
    bpy.props.BoolProperty(name="Cloth Off", description="Mesh Keyframe Only: No cloth behavior", default=False)#, update=cb_cache)

    cache_force:\
    bpy.props.FloatProperty(name="Cache Force", description="Target the cached value as a force", default=1.0, min=0, max=1, soft_min= -100, soft_max=100, precision=3)

    # record
    record:\
    bpy.props.BoolProperty(name="Cloth Record", description="Record changes when 'Continuous'", default=False)

    max_frames:\
    bpy.props.IntProperty(name="Max Frames", description="Record this many", default=1000)#, update=cb_cache)


    # extras ------->>>
    # Wind. Note, wind should be measured against normal and be at zero when normals are at zero. Squared should work
    wind_x:\
    bpy.props.FloatProperty(name="Wind X", 
        description="Not the window cleaner", 
        default=0, precision=4)#, update=refresh_noise_decay)

    wind_y:\
    bpy.props.FloatProperty(name="Wind Y", 
        description="Y? Because wind is cool", 
        default=0, precision=4)#, update=refresh_noise_decay)

    wind_z:\
    bpy.props.FloatProperty(name="Wind Z", 
        description="It's windzee outzide", 
        default=0, precision=4)#, update=refresh_noise_decay)

    turbulence:\
    bpy.props.FloatProperty(name="Wind Turbulence", 
        description="Add Randomness to wind strength", 
        default=.5, precision=4)#, update=refresh_noise_decay)

    random_direction:\
    bpy.props.FloatProperty(name="Wind Turbulence", 
        description="Add randomness to wind direction", 
        default=.5, precision=4)#, update=refresh_noise_decay)

    inflate:\
    bpy.props.FloatProperty(name="inflate", 
        description="add force to vertex normals", 
        default=0, precision=4)


# create properties ----------------
# scene:
class McPropsScene(bpy.types.PropertyGroup):

    interference:\
    bpy.props.BoolProperty(name="interference", description="Alien forces from outside this universe hijacked the cloth upsetting coordinates and velocity", default=False)

    kill_duplicator:\
    bpy.props.BoolProperty(name="kill duplicator/loader", description="", default=False, update=cb_duplicator)

    pause_all:\
    bpy.props.BoolProperty(name="Pause All", description="", default=False, update=cb_pause_all)

    play_all:\
    bpy.props.BoolProperty(name="Play all", description="", default=False, update=cb_play_all)

    delay:\
    bpy.props.FloatProperty(name="Delay", description="Slow down the continuous update", default=0, min=0, max=100)

    pause_selected:\
    bpy.props.BoolProperty(name="Cloth Grab", description="Only move cloth during modal grab", default=True)

    run_editmode:\
    bpy.props.BoolProperty(name="Run Editmode", description="Run cloth sim when in edit mode", default=True)

    view_virtual:\
    bpy.props.BoolProperty(name="View Virtual Springs", description="create a mesh to show virtual springs", default=False)
    # make this one a child object that is not selectable.

    update_shading:\
    bpy.props.BoolProperty(name="Update Shading in Edit Mode", description="Bad for performance but keeps eevee shading updated", default=False)
    # make this one a child object that is not selectable.


# ^                                                          ^ #
# ^                     END properties                       ^ #
# ============================================================ #


# ============================================================ #
#                    registered operators                      #
#                                                              #

class MCResetToBasisShape(bpy.types.Operator):
    """Reset the cloth to basis shape"""
    bl_idname = "object.mc_reset_to_basis_shape"
    bl_label = "MC Reset To Basis Shape"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        if ob.data.is_editmode:
            cloth.obm = bmesh.from_edit_mesh(ob.data)
            ob.update_from_editmode()
            Basis = cloth.obm.verts.layers.shape["Basis"]
            for v in cloth.obm.verts:
                v.co = v[Basis]

        reset_shapes(ob)
        cloth.co = get_co_shape(ob, "Basis")
        cloth.velocity[:] = 0
        cloth.pin_arr[:] = cloth.co
        cloth.feedback[:] = 0
        cloth.select_start[:] = cloth.co

        current_key = ob.data.shape_keys.key_blocks['MC_current']
        current_key.data.foreach_set('co', cloth.co.ravel())
        ob.data.update()
        Collider(cloth)
        refresh(cloth, skip=True)
        return {'FINISHED'}


class MCResetSelectedToBasisShape(bpy.types.Operator):
    """Reset the selected verts to basis shape"""
    bl_idname = "object.mc_reset_selected_to_basis_shape"
    bl_label = "MC Reset Selected To Basis Shape"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        if ob.data.is_editmode:
            Basis = cloth.obm.verts.layers.shape["Basis"]
            for v in cloth.obm.verts:
                if v.select:
                    v.co = v[Basis]

        reset_shapes(ob)
        bco = get_co_shape(ob, "Basis")
        cloth.co[cloth.selected] = bco[cloth.selected]
        cloth.pin_arr[cloth.selected] = bco[cloth.selected]
        cloth.select_start[cloth.selected] = bco[cloth.selected]
        cloth.feedback[cloth.selected] = 0

        bco[~cloth.selected] = cloth.co[~cloth.selected]
        ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', bco.ravel())
        ob.data.update()

        return {'FINISHED'}
    

class MCRefreshVertexGroups(bpy.types.Operator):
    """Refresh Vertex Group Weights To Cloth Settings"""
    bl_idname = "object.mc_refresh_vertex_groups"
    bl_label = "MC Refresh Vertex Groups"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = get_cloth(ob)
        refresh(cloth)
        return {'FINISHED'}


class MCSurfaceFollow(bpy.types.Operator):
    """Connect points to nearest surface"""
    bl_idname = "object.mc_surface_follow"
    bl_label = "MC Surface Follow"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):

        active = bpy.context.object
        if active is None:
            msg = "Must have an active mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        if active.type != 'MESH':
            msg = "Active object must be a mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        if not active.data.polygons:
            msg = "Must have at least one face in active mesh"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        cloths = [i for i in bpy.data.objects if ((i.MC_props.cloth) & (i is not active) & (i.select_get()))]
        if not cloths:
            msg = "Must select at least one cloth object and an active object to bind to"
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')
            return {'FINISHED'}

        # writes to cloth instance
        create_surface_follow_data(active, cloths)

        return {'FINISHED'}


class MCCreateSewLines(bpy.types.Operator):
    """Create sew lines between sets of points"""
    bl_idname = "object.mc_create_sew_lines"
    bl_label = "MC Create Sew Lines"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        cloth = MC_data['cloths'][ob['MC_cloth_id']]
    
        if ob.data.is_editmode:        
            bpy.ops.mesh.bridge_edge_loops()
            bpy.ops.mesh.delete(type='ONLY_FACE')

        return {'FINISHED'}


class MCSewToSurface(bpy.types.Operator):
    """Draw a line on the surface and sew to it"""
    bl_idname = "object.mc_sew_to_surface"
    bl_label = "MC Create Surface Sew Line"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        #mode = ob.mode
        #if ob.data.is_editmode:
            #bpy.ops.object.mode_set(mode='OBJECT')

        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        #bco = get_co_shape(ob, "Basis")

        #bpy.ops.object.mode_set(mode=mode)
        #ob.data.update()
        return {'FINISHED'}


# !!! can't undo delete cache or keyframe.
class MCDeleteCache(bpy.types.Operator):
    """Delete cache folder for current object"""
    bl_idname = "object.mc_delete_cache"
    bl_label = "Really delete this folder and its contents?"
    bl_options = {'REGISTER', 'UNDO'}


    def __init__(self):
        print("ran this class method thingy")

        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        if ob.MC_props.cloth:
            cloth = MC_data['cloths'][ob['MC_cloth_id']]

            path = pathlib.Path(ob.MC_props.cache_folder).joinpath('MC_cache_files')
            current = path.joinpath(ob.MC_props.cache_name)

        msg = 'Really delete ' + str(current) + ' and its contents?'

    @classmethod
    def poll(cls, context):
        return True

    def invoke(self, context, event):
        self.bl_label = "Object: {0}".format(context.active_object.name)

        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):

        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        path = pathlib.Path(ob.MC_props.cache_folder).joinpath('MC_cache_files')
        current = path.joinpath(ob.MC_props.cache_name)

        if os.path.exists(current):
            shutil.rmtree(current, ignore_errors=True)
            ob.MC_props['cache'] = False
            ob.MC_props['play_cache'] = False

            msg = 'Deleted ' + str(current) + ' and its contents.'
            bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')

            return {'FINISHED'}

        msg = 'Could not delete ' + str(current) + '. No such file'
        bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')

        return {'FINISHED'}


class MCCreateMeshKeyframe(bpy.types.Operator):
    """Create a linear path between cache files"""
    bl_idname = "object.mc_mesh_keyframe"
    bl_label = "MC Create Mesh Keyframe"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]
        print("Will create mesh cache keyframe... Some day. Sigh...")
        cloth.co = get_co_mode(ob)
        overwrite = ob.MC_props.overwrite_cache
        ob.MC_props.overwrite_cache = True
        cache(cloth, keying=True)
        ob.MC_props.overwrite_cache = overwrite

        return {'FINISHED'}


class MCRemoveMeshKeyframe(bpy.types.Operator):
    """Remove mesh keyframe at current frame"""
    bl_idname = "object.mc_remove_keyframe"
    bl_label = "MC Remove Mesh Keyframe"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object
        cloth = MC_data['cloths'][ob['MC_cloth_id']]
        print("Will remove mesh cache keyframe... Some day. Sigh...")

        return {'FINISHED'}


class MCCreateVirtualSprings(bpy.types.Operator):
    """Create Virtual Springs Between Selected Verts"""
    bl_idname = "object.mc_create_virtual_springs"
    bl_label = "MC Create Virtual Springs"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        cloth = get_cloth(ob)
        obm = cloth.obm
        verts = np.array([v.index for v in obm.verts if v.select])
        cloth.virtual_spring_verts = verts
        virtual_springs(cloth)

        return {'FINISHED'}


class MCApplyForExport(bpy.types.Operator):
    # !!! Not Finished !!!!!!
    """Apply cloth effects to mesh for export."""
    bl_idname = "object.MC_apply_for_export"
    bl_label = "MC Apply For Export"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = get_last_object()[1]
        v_count = len(ob.data.vertices)
        co = np.zeros(v_count * 3, dtype=np.float32)
        ob.data.shape_keys.key_blocks['modeling cloth key'].data.foreach_get('co', co)
        ob.data.shape_keys.key_blocks['Basis'].data.foreach_set('co', co)
        ob.data.shape_keys.key_blocks['Basis'].mute = True
        ob.data.shape_keys.key_blocks['Basis'].mute = False
        ob.data.vertices.foreach_set('co', co)
        ob.data.update()

        return {'FINISHED'}


class MCVertexGroupPin(bpy.types.Operator):
    """Add Selected To Pin Vertex Group"""
    bl_idname = "object.mc_vertex_group_pin"
    bl_label = "MC Vertex Group Pin"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        ob = MC_data['recent_object']
        if ob is None:
            ob = bpy.context.object

        cloth = MC_data['cloths'][ob['MC_cloth_id']]

        return {'FINISHED'}
    
    
class PinSelected(bpy.types.Operator):
    """Add pins to verts selected in edit mode"""
    bl_idname = "object.modeling_cloth_pin_selected"
    bl_label = "Modeling Cloth Pin Selected"
    bl_options = {'REGISTER', 'UNDO'}    
    def execute(self, context):
        ob = bpy.context.object
        id = ob['MC_cloth_id']
        cloth = get_cloth(ob)
        if ob.data.is_editmode:    
            ob.update_from_editmode()
            ob.data.update()

        idx = [i.index for i in ob.data.vertices if i.select]
        aco = apply_transforms(cloth.ob, get_co_shape(cloth.ob, "MC_current"))
        
        for v in idx:    
            e = bpy.data.objects.new('MC_pin', None)
            bpy.context.collection.objects.link(e)
            e.location = aco[v]
            e.select_set(True)
            e.MC_props.is_hook = True
            e['MC_cloth_id'] = id
            e.MC_props.hook_index = v
     
        bpy.context.view_layer.update()
        return {'FINISHED'}


# ^                                                          ^ #
#                  END registered operators                    #
# ============================================================ #


# ============================================================ #
#                         draw code                            #
#                                                              #
class PANEL_PT_MC_Master(bpy.types.Panel):
    """MC Panel"""
    bl_label = "MC Master Panel"
    bl_idname = "PANEL_PT_mc_master_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    @classmethod
    def poll(cls, context):
        ob = bpy.context.object
        if ob is None:
            return False

        if ob.type != 'MESH':
            if MC_data['recent_object'] is None:
                return False
            ob = MC_data['recent_object']

        try: # if we delete the object then grab an empty
            ob.name
        except:
            print("Cloth object was deleted")
            MC_data['recent_object'] = None
            return False

        return True

    def __init__(self):
        ob = bpy.context.object

        if ob.type != 'MESH':
            ob = MC_data['recent_object']

        self.ob = ob
        self.cloth = ob.MC_props.cloth


# MAIN PANEL
class PANEL_PT_modelingClothMain(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Main"""
    bl_label = "MC Main"
    bl_idname = "PANEL_PT_modeling_cloth_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        ob = self.ob
        layout = self.layout
        col = layout.column(align=True)
        col.prop(sc.MC_props, "kill_duplicator", text="kill_duplicator", icon='DUPLICATE')
        col.prop(ob.MC_props, "cache_only", text="Cache Only", icon='RENDER_ANIMATION')
        col.prop(ob.MC_props, "dense", text="Dense Mesh", icon='VIEW_ORTHO')
        col.prop(ob.MC_props, "quad_bend", text="Quad Bend Springs", icon='VIEW_PERSPECTIVE')

        col = layout.column(align=True)
        col.scale_y = 1.5
        col.prop(ob.MC_props, "collider", text="Collide", icon='MOD_PHYSICS')
        if True:    
            if ob.MC_props.collider:
                row = col.row()
                row.scale_y = 0.75
                #row.label(icon='PROP_OFF')
                row.prop(ob.MC_props, "outer_margin", text="Outer Margin")#, icon='PROP_OFF')
                row = col.row()
                row.scale_y = 0.75
                row.prop(ob.MC_props, "inner_margin", text="Inner Margin")#, icon='PROP_OFF')
                row = col.row()
                row.scale_y = 0.75
                #row.label(icon='PROP_OFF')
                row.prop(ob.MC_props, "oc_friction", text="Friction")#, icon='PROP_OFF')
                row = col.row()
                #col = layout.column(align=True)
                row.scale_y = 0.75
                #row.label(icon='PROP_CON')
                row.prop(ob.MC_props, "static_friction", text="Static Friction")#, icon='PROP_CON')
        if ob.MC_props.cloth:
            col.prop(ob.MC_props, "self_collide", text="Self Collision", icon='FULLSCREEN_EXIT')
            if ob.MC_props.self_collide:
                row = col.row()
                #col = layout.column(align=True)
                row.scale_y = 0.75
                row.label(icon='PROP_CON')
                row.prop(ob.MC_props, "self_collide_margin", text="SC Margin", icon='PROP_CON')
                row = col.row()
                row.label(icon='CON_PIVOT')
                row.scale_y = 0.75            
                row.prop(ob.MC_props, "sc_friction", text="Friction", icon='CON_PIVOT')            
                #row = col.row()
                #row.label(icon='CON_PIVOT')            
                #row.scale_y = 0.75            
                #row.prop(ob.MC_props, "sc_vel_damping", text="Damping", icon='CON_PIVOT')
                row = col.row()
                row.label(icon='CON_PIVOT')
                row.scale_y = 0.75            
                row.prop(ob.MC_props, "self_collide_force", text="SC Force", icon='CON_PIVOT')
                row = col.row()            
                #row = col.row()
            #col.prop(ob.MC_props, "new_sc", text="New Self Collision", icon='FULLSCREEN_EXIT')
            #if ob.MC_props.new_sc:
                #row = col.row()
                #col = layout.column(align=True)
                #row.scale_y = 0.75
                #row.label(icon='PROP_CON')
                #row.prop(ob.MC_props, "new_self_margin", text="SC Margin", icon='PROP_CON')
                #row = col.row()
                #row.label(icon='CON_PIVOT')
                #row.scale_y = 0.75            
                #row.prop(ob.MC_props, "sc_friction", text="Friction", icon='CON_PIVOT')            
                row = col.row()            
                # show the box max for collision stuff
                row.label(icon='CON_PIVOT')
                row.scale_y = 0.75            
                row.prop(ob.MC_props, "sc_box_max", text="Box Size", icon='CON_PIVOT')


            # use current mesh or most recent cloth object if current ob isn't mesh

            # if we select a new mesh object we want it to display

            col.prop(ob.MC_props, "detangle", text="Detangle", icon='GRAPH')
            if False: # detanlge options    
                if ob.MC_props.detangle:
                    row = col.row()
                    #col = layout.column(align=True)
                    row.scale_y = 0.75
                    row.label(icon='PROP_CON')
                    row.prop(ob.MC_props, "self_collide_margin", text="SC Margin", icon='PROP_CON')

        col = layout.column(align=True)
        col.scale_y = 1.5
        # display the name of the object if "cloth" is True
        #   so we know what object is the recent object
        recent_name = ''
        if ob.MC_props.cloth:
            recent_name = ob.name

        col.prop(ob.MC_props, "cloth", text="Cloth " + recent_name, icon='MOD_CLOTH')
        if ob.MC_props.cloth:
            col.prop(ob.MC_props, "detect_collisions", text="Detect Collisions", icon='LIGHT_AREA')

            os_text = "Override Settings"
            col.prop(ob.MC_props, "override_settings", text=os_text, icon='SORT_ASC')
            col.prop(ob.MC_props, "shrink_grow", text="Shrink/Grow", icon='FULLSCREEN_EXIT')
            col.prop(ob.MC_props, "wrap_force", text="Wrap Force", icon='MOD_SHRINKWRAP')

            if ob.MC_props.override_settings:
                row = col.row()
                #col = layout.column(align=True)
                row.scale_y = 0.75
                #row.label(icon='PROP_CON')
                row.prop(ob.MC_props, "outer_margin", text="Outer Margin", icon='PROP_OFF')
                row = col.row()
                row.scale_y = 0.75
                row.prop(ob.MC_props, "inner_margin", text="Inner Margin", icon='PROP_OFF')
                row = col.row()
                row.scale_y = 0.75
                #row.label(icon='PROP_OFF')
                row.prop(ob.MC_props, "oc_friction", text="Friction", icon='PROP_OFF')
                row = col.row()
                row.scale_y = 0.75
                #row.label(icon='PROP_OFF')
                row.prop(ob.MC_props, "static_friction", text="Static Friction", icon='PROP_CON')


            col.label(text='Update Mode')
            col = layout.column(align=True)
            row = col.row()
            row.scale_y = 2
            row.prop(ob.MC_props, "continuous", text="Continuous", icon='FILE_REFRESH')
            row = col.row()
            row.scale_y = 2
            row.prop(ob.MC_props, "animated", text="Animated", icon='PLAY')
            row = col.row()
            row.scale_y = 1
            row.prop(sc.MC_props, "delay", text="Delay", icon='SORTTIME')
            box = col.box()
            box.scale_y = 2
            box.operator('object.mc_reset_to_basis_shape', text="RESET", icon='RECOVER_LAST')
            box.operator('object.mc_reset_selected_to_basis_shape', text="RESET SELECTED", icon='RECOVER_LAST')
            if ob.data.is_editmode:
                box.operator('object.mc_refresh_vertex_groups', text="V-group Refresh", icon='GROUP_VERTEX')
            box.operator('object.modeling_cloth_pin_selected', text="Hook Selected", icon='HOOK')
            col = layout.column(align=True)
            col.use_property_decorate = True
            col.label(text='Target Object')
            col.prop(ob.MC_props, "target", text="", icon='DUPLICATE')


# CACHE PANEL
class PANEL_PT_modelingClothCache(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Cache"""
    bl_label = "MC Cache"
    bl_idname = "PANEL_PT_modeling_cloth_cache"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        ob = self.ob

        if ob.MC_props.cloth:
            try:                
                cloth = get_cloth(ob)
            except:
                return
                #reload_from_save()
                #cloth = get_cloth(ob)
            
            col = layout.column(align=True)
            box = col.box()
            bcol = box.column()

            bcol.prop(ob.MC_props, "cache", text="Record", icon='RENDER_ANIMATION')
            bcol.prop(ob.MC_props, "max_frames")
            bcol.prop(ob.MC_props, "overwrite_cache", text="Overwrite", icon='FILE_REFRESH')
            bcol.prop(ob.MC_props, "internal_cache", text="Pack", icon='NORMALIZE_FCURVES')
            bcol = box.column()
            bcol.scale_y = 1.5
            bcol.operator('object.mc_delete_cache', text="Delete Cache", icon='KEY_HLT')
            bcol = box.column()
            bcol.label(text="Cache Foder")
            bcol.prop(ob.MC_props, "cache_folder", text="")
            bcol.label(text="Custom Name")
            bcol.prop(ob.MC_props, "cache_name", text="")
            bcol.prop(ob.MC_props, "cache_desktop", text="Use Desktop")

            #col.separator()
            bcol = col.box().column()

            bcol.label(text='Animate Mode')

            #bcol.enabled = hasattr(cloth, 'cache_dir')
            bcol.enabled = check_file_path(ob)[0]

            bcol.prop(ob.MC_props, 'play_cache', text='Playback', icon='PLAY')
            bcol.prop(ob.MC_props, 'cache_force', text='Influence', icon='SNAP_ON')
            bcol.prop(ob.MC_props, 'current_cache_frame', text='Frame')#, icon='SNAP_ON')



            #row = bcol.row()
            #row.prop(ob.MC_props, "start_frame", text="")
            #row.prop(ob.MC_props, "end_frame", text="")
            #bcol.label(text='Continuous Mode')
            #bcol.prop(ob.MC_props, "record", text="Record", icon='REC')




            folder = 'Cach=None'
            if hasattr(cloth, 'cache_dir'):
                fo = cloth.cache_dir
                if os.path.exists(fo):
                    folder = fo

            col.separator()
            box = col.box().column()
            box.label(text='Mesh Keyframing')
            box.prop(ob.MC_props, 'cloth_off', text='Cloth Off', icon='CANCEL')

            box.separator()

            frame = 'Key=None'

            if hasattr(cloth, 'cache_dir'):
                folder = cloth.cache_dir
                if folder.exists():
                    f = str(folder) + str(sc.frame_current)
                    if os.path.exists(f):
                        frame = 'Key=' + str(sc.frame_current)


            box.operator('object.mc_mesh_keyframe', text="Keyframe Mesh", icon='KEY_HLT')
            box.label(text=frame)
            box.operator('object.mc_remove_keyframe', text="Del Keyframe", icon='KEY_DEHLT')



            #col.separator()
            #box = col.box().column()
            #box.label(text='Record Continuous')
            #box.prop(ob.MC_props, "record", text="Record", icon='REC')
            #box.prop(ob.MC_props, "max_frames")
            #col.separator()
            #box = col.box().column()
            #box.label(text='Playback Mode')
            #box.prop(ob.MC_props, 'play_cache', text='Playback', icon='PLAY')
            #box.prop(ob.MC_props, 'cache_force', text='Influence', icon='SNAP_ON')


# SEWING PANEL
class PANEL_PT_modelingClothSewing(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Settings"""
    bl_label = "MC Sewing"
    bl_idname = "PANEL_PT_modeling_cloth_sewing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1
            col.label(text='Sewing')
            col.prop(ob.MC_props, "sew_force", text="Sew Force")
            col.prop(ob.MC_props, "target_sew_length", text="Target Length")

            box = col.box()
            box.scale_y = 2
            box.operator('object.mc_create_virtual_springs', text="Virtual Springs", icon='AUTOMERGE_ON')
            box.operator('object.mc_create_sew_lines', text="Sew Lines", icon='AUTOMERGE_OFF')
            box.operator('object.mc_sew_to_surface', text="Surface Sewing", icon='MOD_SMOOTH')

            return
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        box = col.box()
        box.scale_y = 2
        box.operator('object.mc_surface_follow', text="Follow Surface", icon='OUTLINER_DATA_SURFACE')
        box = col.box()
        box.scale_y = 1
        box.prop(ob.MC_props, "surface_follow_selection_only", text="Selected Polys Only")


# SETTINGS PANEL
class PANEL_PT_modelingClothSettings(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Settings"""
    bl_label = "MC Settings"
    bl_idname = "PANEL_PT_modeling_cloth_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1.5
            col.label(text='Forces')
            col.prop(ob.MC_props, "velocity", text="velocity")
            col.prop(ob.MC_props, "gravity", text="gravity")
            col.prop(ob.MC_props, "inflate", text="inflate")
            col.prop(ob.MC_props, "wind_x", text="wind x")
            col.prop(ob.MC_props, "wind_y", text="wind y")
            col.prop(ob.MC_props, "wind_z", text="wind z")
            col.prop(ob.MC_props, "turbulence", text="turbulence")
            col.prop(ob.MC_props, "random_direction", text="random direction")

            col.label(text='Springs')
            #col.scale_y = 1
            col = layout.column(align=True)
            col.prop(ob.MC_props, "sub_frames", text="Sub Frames")
            col.prop(ob.MC_props, "stretch_iters", text="stretch iters")
            col.prop(ob.MC_props, "stretch", text="stretch")
            col.prop(ob.MC_props, "push", text="push")
            col.prop(ob.MC_props, "feedback", text="feedback")
            col.prop(ob.MC_props, "bend_iters", text="bend iters")
            col.prop(ob.MC_props, "bend", text="bend")


# VERTEX GROUPS PANEL
class PANEL_PT_modelingClothVertexGroups(PANEL_PT_MC_Master, bpy.types.Panel):
    """Modeling Cloth Vertex Groups"""
    bl_label = "MC Vertex Groups"
    bl_idname = "PANEL_PT_modeling_cloth_vertex_groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        ob = self.ob
        cloth = ob.MC_props.cloth
        if cloth:
            sc = bpy.context.scene
            layout = self.layout

            # use current mesh or most recent cloth object if current ob isn't mesh
            MC_data['recent_object'] = ob
            col = layout.column(align=True)
            col.scale_y = 1
            col.label(text='Vertex Groups')
            col.operator('object.mc_vertex_group_pin', text="Pin Selected", icon='PINNED')
            col.prop(ob.MC_props, "vg_pin", text="pin")
            col.prop(ob.MC_props, "vg_drag", text="drag")
            col.prop(ob.MC_props, "vg_surface_follow", text="Surface Follow")


# EDIT MODE PANEL
class PANEL_PT_modelingClothPreferences(bpy.types.Panel):
    """Modeling Cloth Preferences"""
    bl_label = "MC Preferences"
    bl_idname = "PANEL_PT_modeling_cloth_preferences"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Extended Tools"

    def draw(self, context):
        sc = bpy.context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1
        col.label(text='Preferences')
        col.prop(sc.MC_props, "run_editmode", text="Editmode Run")
        col.prop(sc.MC_props, "update_shading", text="Update Shading")
        col.prop(sc.MC_props, "pause_selected", text="Pause Selected")
        col.prop(sc.MC_props, "view_virtual", text="View Virtual Springs")

# ^                                                          ^ #
# ^                     END draw code                        ^ #
# ============================================================ #


# testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#install_handler(False)


# testing end !!!!!!!!!!!!!!!!!!!!!!!!!



# ============================================================ #
#                         Register                             #
#                                                              #
def duplication_and_load():
    """Runs in it's own handler for updating objects
    that are duplicated while coth properties are true.
    Also checks for cloth, collider, and target objects
    in the file when blender loads."""
    # for loading no need to check for duplicates because we are regenerating data for everyone
    #if load:
    #    return 0


    # for detecting duplicates
    obm = False # because deepcopy doesn't work on bmesh
    print("running duplicator")
    cloths = [i for i in bpy.data.objects if i.MC_props.cloth]
    if len(cloths) > 0:
        id = [i['MC_cloth_id'] for i in cloths]
        idx = max(id) + 1
        u, inv, counts = np.unique(id, return_inverse=True, return_counts=True)
        repeated = counts[inv] > 1
        if np.any(repeated):
            dups = np.array(cloths)[repeated]
            for i in dups:
                cloth_instance = MC_data['cloths'][i['MC_cloth_id']]
                cloth_instance.ob = None
                cloth_instance.target = None # objs don't copy with deepcopy
                if 'obm' in dir(cloth_instance):
                    obm = True # objs don't copy with deepcopy
                    cloth_instance.obm = None # objs don't copy with deepcopy
                MC_data['cloths'][idx] = copy.deepcopy(cloth_instance)
                MC_data['cloths'][idx].ob = i # cloth.ob doesn't copy
                MC_data['cloths'][idx].target = i.MC_props.target # cloth.ob doesn't copy

                # not sure if I need to remake the bmesh since it will be remade anyway... Can't duplicate an object in edit mode. If we switch to edit mode it will remake the bmesh.
                #if obm:
                    #MC_data['cloths'][idx].obm = get_bmesh(i) # bmesh doesn't copy

                i['MC_cloth_id'] = idx
                idx += 1

            print("duplicated an object cloth instance here=============")
            # remove the cloth instances that have been copied
            for i in np.unique(np.array(id)[repeated]):
                MC_data['cloths'].pop(i)

    print("finished duplication =====+++++++++++++++++++++++++")
    colliders = [i for i in bpy.data.objects if i.MC_props.cloth]
    return 1


classes = (
    McPropsObject,
    McPropsScene,
    MCResetToBasisShape,
    MCResetSelectedToBasisShape,
    MCRefreshVertexGroups,
    MCSurfaceFollow,
    MCCreateSewLines,
    MCSewToSurface,
    MCCreateVirtualSprings,
    MCDeleteCache,
    MCCreateMeshKeyframe,
    MCRemoveMeshKeyframe,
    PANEL_PT_modelingClothMain,
    PANEL_PT_modelingClothCache,
    PANEL_PT_modelingClothSettings,
    PANEL_PT_modelingClothSewing,
    PANEL_PT_modelingClothVertexGroups,
    PANEL_PT_modelingClothPreferences,
    MCVertexGroupPin,
    PinSelected,
)


def register():
    # classes
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # props
    bpy.types.Object.MC_props = bpy.props.PointerProperty(type=McPropsObject)
    bpy.types.Scene.MC_props = bpy.props.PointerProperty(type=McPropsScene)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])
        print("deleted handler ", i)

    # drop in the undo handler
    bpy.app.handlers.undo_post.append(undo_frustration)

    # register the data management timer. Updates duplicated objects and objects with modeling cloth properties
    if False:
        bpy.app.timers.register(duplication_and_load)

    # special forces -------------
    bpy.types.Scene.MC_seam_wrangler = False

    # load -----------------------
    #bpy.app.handlers.load_pre.append(reload_from_save)
    bpy.app.handlers.load_post.append(reload_from_save)


def unregister():
    # classes
    
    msg = 'Goodbye cruel world. I may be unregistered but I will live on in your hearts. MC_29 FOREVER!'
    bpy.context.window_manager.popup_menu(oops, title=msg, icon='ERROR')

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

    # props
    del(bpy.types.Scene.MC_props)
    del(bpy.types.Object.MC_props)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])

    handler_names = np.array([i.__name__ for i in bpy.app.handlers.load_post])
    booly = [i == 'reload_from_save' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]

    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.load_post[i])

        
def soft_unregister():
    # props
    del(bpy.types.Scene.MC_props)
    del(bpy.types.Object.MC_props)

    # clean dead versions of the undo handler
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'undo_frustration' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])

    handler_names = np.array([i.__name__ for i in bpy.app.handlers.load_post])
    booly = [i == 'reload_from_save' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]

    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.load_post[i])


if __name__ == "__main__":
    #unregister()
    reload()
    register()
    
    
    
print('--------------- new ---------------')
