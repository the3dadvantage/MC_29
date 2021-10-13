
try:
    import numpy as np
    import bpy
    import bmesh

except ImportError:
    pass
    #print('import fail in grid. Who cares right?')


# universal ----------------------------------
def deselect(ob, sel=None, type='vert'):
    """Deselect all then select something"""
    x = np.zeros(len(ob.data.vertices), dtype=np.bool)
    y = np.zeros(len(ob.data.edges), dtype=np.bool)
    z = np.zeros(len(ob.data.polygons), dtype=np.bool)

    ob.data.vertices.foreach_set('select', x)
    ob.data.edges.foreach_set('select', y)
    ob.data.polygons.foreach_set('select', z)
    
    if sel is not None:    
        if type == 'vert':    
            x[sel] = True
            ob.data.vertices.foreach_set('select', x)
        if type == 'edge':
            y[sel] = True
            ob.data.edges.foreach_set('select', y)
        if type == 'face':
            z[sel] = True
            ob.data.polygons.foreach_set('select', z)
    ob.data.update()


# redistribute ==========================
def measure_angle_at_each_vert(grid):
    """Provide mesh and angle limit in degrees.
    Returns the indices of verts that are sharper than limit"""

    co = grid.co
    
    v1 = np.roll(co, 1, axis=0) - co
    v2 = np.roll(co, -1, axis=0) - co
    
    crosses = np.cross(v1, v2)    
    
    # use the vecs pointing away from each vertex later
    grid.vls = v1
    grid.vrs = v2
    
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, None]
    uv2 = v2 / np.sqrt(rs_dots)[:, None]
    
    con = np.pi /180
    limit = np.cos(con * grid.angle_limit)
    
    angle = np.einsum('ij, ij->i', uv1, uv2)
    sharps = angle > limit   
    
    grid.sharp_normals = np.sign((crosses[sharps][:, 2]))

    if np.sum(sharps) == 0:    
        sharps[0] = True

    return np.arange(co.shape[0])[sharps]


def get_segments(grid):
    """Generate a list of segments
    between sharp edges"""
    sharps = grid.sharps
    segs2 = []
    idx = np.arange(grid.co.shape[0])
    sharp = sharps[0]

    for i in range(len(sharps)):
        if sharps[i] == sharps[-1]:
            final = idx[sharps[i]:].tolist() + idx[:sharps[0] + 1].tolist()
            segs2 += [final]
            return segs2
        
        segs2 += [idx[sharps[i]: sharps[i + 1] + 1].tolist()]
    

def get_seg_length(grid, seg):
    """returns the total length of a set
    of points that are in linear order"""
    co = grid.co
    vecs = co[seg[1:]] - co[seg[:-1]]
    grid.seg_vecs.append(vecs) # might as well save this for later
    seg_length = np.sqrt(np.einsum('ij, ij->i', vecs, vecs))
    grid.seg_lengths.append(seg_length) # saving this also
    total_length = np.sum(seg_length)
    return total_length


def generate_perimeter(grid):
    """Place points around perimeter"""

    # get the length of each segments
    seg_lengths = np.array([get_seg_length(grid, s) for s in grid.segments])
    grid.point_counts = seg_lengths // grid.size
    grid.point_counts[grid.size / seg_lengths > 0.5] = 1
    grid.spacing = seg_lengths / grid.point_counts
    
    # add the first point in the segment (second one gets added next time)   
    seg_sets = np.empty((0,3), dtype=np.float32)
    iters = len(grid.segments)
    
    if grid.v_border:
        grid.sew_edgesss = []
    
    for i in range(iters):    
        seg_sets = move_point_on_path(grid, i, seg_sets)

    if grid.v_border:
        deselect(grid.new_ob, sel=grid.sew_edgesss, type='edge')
        
    return seg_sets
    

def move_point_on_path(grid, idx, seg_sets):
    """Walk the points until we are at the distance
    we space them"""
    
    if grid.v_border:
        obm = grid.new_obm
        
        #I think I''m finally ready to start work for today
        
        # the dictionary that contains all the sew data stuffingtons
        
        '''
        grid.sew_relationships
        so we have to check for grid.v_border and if thats true
        we have to track what bmesh edge we''re on and find
        the sew verts and what other edge they connect to.
        '''

        
    co = grid.co
    seg = grid.segments[idx]
    lengths = grid.seg_lengths[idx]
    spacing = grid.spacing[idx]
    vecs = grid.seg_vecs[idx]
    count = grid.point_counts[idx]

    seg_co_set = [co[seg[0]]] # the last one will be filled in by the first one next time.
    
    if grid.v_border:    
        grid.accumulated_border_count += 1

        sharp_vert = grid.v_loop[seg[0]]
        grid.sew_relationships['no_fold_verts'] += [sharp_vert]
        
        s_vert = obm.verts[sharp_vert]
        three_verts = [e.other_vert(s_vert).index for e in s_vert.link_edges if len(e.link_faces) == 1]

        if True:
            five_verts = [[e.other_vert(obm.verts[v]).index for e in obm.verts[v].link_edges if len(e.link_faces) == 1] for v in three_verts]
            
            for e in five_verts:
                three_verts += e
            
        three_verts += [sharp_vert]        
        grid.sew_relationships['three_verts'] += [three_verts]        

    if count == 0:
        return seg_co_set

    growing_length = 0
    len_idx = 0
    build = spacing
    
    counter = 0
    for x in range(int(count) - 1):
        growing_length = 0
        len_idx = 0
        counter += 1
        while growing_length < spacing:
            growing_length += lengths[len_idx]            
            len_idx += 1
        
        if grid.v_border:
            
            a_vert = grid.v_loop[seg[len_idx]]
            vert = obm.verts[a_vert]

            le = [e.index for e in vert.link_edges if len(e.link_faces) == 0]
            lv = [e.other_vert(vert).index for e in vert.link_edges if len(e.link_faces) == 0]

            grid.accumulated_border_count += 1
            
            current_border_vert = grid.accumulated_border_count
            for v in lv:
                if v in grid.sew_relationships:
                    grid.new_sew_edges += [[grid.sew_relationships[v], current_border_vert]]                
                
            three_verts = [e.other_vert(vert).index for e in vert.link_edges if len(e.link_faces) == 1]
                
            if True:    
                five_verts = [[e.other_vert(obm.verts[v]).index for e in obm.verts[v].link_edges if len(e.link_faces) == 1] for v in three_verts]
                for e in five_verts:
                    three_verts += e            
            
            three_verts += [a_vert]
            
            grid.sew_relationships['no_fold_verts'] += [a_vert]
            grid.sew_relationships['three_verts'] += [three_verts]
            grid.sew_relationships[a_vert] = current_border_vert    
            
            # --------------------------------------------------------------------
            #so there is this vert. it''s either before or after the border vert
            #on the new border. so where are the new border verts here?
            
            # --------------------------------------------------------------------
            

        # back up to the last point now 
        len_idx -= 1
        growing_length -= lengths[len_idx]

        
        # move from the past point along the last vector until we
        # hit the proper spacing
        end_offset = spacing - growing_length
        last_dif = lengths[len_idx] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        along_last = end_offset / last_dif
        
        move = vecs[len_idx]
        
        loc = co[seg[len_idx]] + move * along_last
        
        seg_co_set.append(loc)

        # start from the beginning because it's easier
        spacing += build
    
    # join to master array:
    seg_sets = np.append(seg_sets, seg_co_set, axis=0) 
    return seg_sets


def mag_set(mag, v2):    
    '''Applys sqared mag to v2'''
    d1 = mag ** 2
    d2 = v2 @ v2
    div = d1/d2
    return v2 * np.sqrt(div) # is this really dumb? Why square it then ge the square root?


class Distribute:
    pass
    

def redistribute(cut_polyline, grid_size=4.0, angle=20, v_border=None):
    # walk around the edges and and plot evenly spaced points
    # respecting the sharpness of the angle
    
    grid = Distribute()
    grid.v_border = False
    # p1 garment ---------------------
    if v_border is not None:
        grid.v_border = True
        grid.v_loop = v_border.ordered
        grid.new_ob = v_border.new_ob
        grid.new_obm = v_border.new_obm
        grid.sew_relationships = v_border.sew_relationships
        grid.accumulated_border_count = v_border.accumulated_border_count
        grid.new_sew_edges = v_border.new_sew_edges
        grid.test_attritbute = v_border.test_attritbute
    # p1 garment ---------------------
    
    grid.co = np.zeros((len(cut_polyline), 3), dtype=np.float32)
    grid.co[:, :2] = cut_polyline
    
    grid.angle_limit = 180 - angle
    grid.sharps = measure_angle_at_each_vert(grid)
    grid.segments = get_segments(grid)
    grid.size = grid_size
    grid.seg_vecs = [] # gets filled by the function below
    grid.seg_lengths = [] # gets filled by the function below
    new_co = np.empty((0,3), dtype=np.float32)
    
    # create points for every segment between sharps --------------------
    #for i in range(iters):
    x = generate_perimeter(grid)
    new_co = np.append(new_co, x, axis=0)
    
    return new_co


def make_v_objects(border):
    name = "grid_mesh"


def make_objects(border, ob=None):
    """Creates the grid and border 
    objects in the blender scene"""
    if ob is None:
        ob = bpy.context.object
        if ob == None:
            return
        if not ob.type == "MESH":
            return
        
    name = ob.name + '_new_border'
    grid_name = ob.name + '_new_mc_grid'
    new = border.new_border
    new_ed = border.new_border_edges.tolist()
    mob = None
    gob = None
    if name in bpy.data.objects:
        mob = bpy.data.objects[name]
    
    if grid_name in bpy.data.objects:
        gob = bpy.data.objects[grid_name]
    
    faces = []
    if border.triangles:
        faces = border.br.tolist() + border.tr.tolist()
    else:
        faces = border.q_face.tolist()

    grid = link_mesh(verts=new.tolist(), edges=new_ed, faces=[], name=name, ob=mob)
    grid2 = link_mesh(verts=border.grid_co.tolist(), edges=border.grid_edges, faces=faces, name=grid_name, ob=gob)
    
    grid.matrix_world = ob.matrix_world
    grid2.matrix_world = ob.matrix_world

    border.border_ob = grid
    border.grid_ob = grid2
    border.grid_obm = get_bmesh(border.grid_ob, refresh=True)

    veidx = [[e.verts[0].index, e.verts[1].index] for e in border.grid_obm.edges]
    border.grid_edges = veidx
    border.g_edge_co = border.grid_co[border.grid_edges]
    

def link_mesh(verts, edges, faces, name='name', ob=None):
    """Generate and link a new object from pydata.
    If object already exists replace its data
    with a new mesh and delete the old mesh."""
    if ob is None:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, edges, faces)
        mesh.update()
        mesh_ob = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(mesh_ob)
        return mesh_ob
    
    mesh_ob = ob
    old = ob.data
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    mesh_ob.data = mesh
    bpy.data.meshes.remove(old)
    return mesh_ob


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


def merge_verts(ob, margin=0.001, obm=None):

    """
    Could give springs shorter than a given length a length of zero
    and move them towards the outside before removing doubles to avoid
    moving the boundary when removing doubles.
    """
    if obm is None:
        obm = get_bmesh(poly_line, refresh=True)
    
    bmesh.ops.remove_doubles(obm, verts=obm.verts, dist=margin)
    obm.to_mesh(ob.data)

    ob.data.update()
    obm.clear()
    obm.free()


def ccw(A,B,C):
    return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])


# Return true if line segments AB and CD intersect
def new_intersect(A,B,C,D):
    return (ccw(A,C,D) != ccw(B,C,D)) & (ccw(A,B,C) != ccw(A,B,D))


def edges_edges_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''2d line intersect for two groups of edges.''' 
    # this fails in certain cases.
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = da[:, ::-1] * np.array([1, -1])
    denom = np.einsum('ij,ij->i', dap, db)    
    num = np.einsum('ij,ij->i', dap, dp)
    scale = (num / denom)

    if intersect:
        return b1 + db * scale[:, None], (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale[:, None]


def edge_edges_intersect_2d(a1,a2, b1,b2, intersect=False):
    '''2d line intersect for an edge and a group of edges.'''    
    da = a2-a1 # this is a single vec
    db = b2-b1
    dp = a1-b1
    dap = da[::-1] * np.array([1, -1])
    
    denom = db @ dap    
    num = dp @ dap
    scale = (num / denom)

    if intersect:
        return b1 + db * scale[:, None], (scale > 0) & (scale < 1)
    else:
        return b1 + db * scale[:, None]


def get_ordered_loop(poly_line, edges=None):
    """Takes a bunch of verts and gives the order
    based on connected edges.
    Or, takes an edge array of vertex indices
    and gives the vertex order."""

    if edges is not None:        
        v = edges[0][0]
        le = edges[np.any(v == edges, axis=1)]
        if len(le) != 2:
            print("requires a continuous loop of edges")
            return
            
        ordered = [v]
        for i in range(len(poly_line.data.vertices)):
            #le = v.link_edges
            le = edges[np.any(v == edges, axis=1)]
            if len(le) != 2:
                print("requires a continuous loop of edges")
                break

            ot1 = le[0][le[0] != v]
            ot2 = le[1][le[1] != v]
            v = ot1
            if ot1 in ordered[-2:]:    
                v = ot2
            if v == ordered[0]:
                break

            ordered += [v[0]]
            
        return ordered
        
    obm = get_bmesh(poly_line, refresh=True)
    v = obm.edges[0].verts[0]
    le = v.link_edges

    if len(le) != 2:
        print("requires a continuous loop of edges")
        return
        
    ordered = [v.index]
    for i in range(len(poly_line.data.vertices)):
        le = v.link_edges

        if len(le) != 2:
            print("requires a continuous loop of edges")
            break

        ot1 = le[0].other_vert(v)
        ot2 = le[1].other_vert(v)
        v = ot1
        if ot1.index in ordered[-2:]:    
            v = ot2
        if v.index == ordered[0]:
            break

        ordered += [v.index]
    
    return ordered

'''
not sure how to deal with the folded faces and
creating grids.
The pre-wrap shape should have correct folds.
Could maybe get the fold verts from the json file,
where the verts are in the pre-wrap shape I could
create relationships

it would probably be better to use my own folds.
I could vary the depth of the folds so there wouldn't
be faces already intersected where two flap folds
overlap at the corners. 

Since the pre-wrap already has the folds more or less
correct, maybe I can create a grid around the panels
less the fold verts (as defined by the jason)

maybe I can remove the fold verts and get the sew lines from
the remaining bounds?
'''

def generate_grid(border):
    """Generates a grid slightly larger than the polyline
    bounding box. Can be quads or tris."""
    
    M = border.size * 0.1
    if border.p1:
        M = border.inner_size * 0.1
        border.size = border.inner_size
    
    min = np.min(border.ordered_co, axis=0)
    max = np.max(border.ordered_co, axis=0)
    
    tri_edges = np.empty(0, dtype=np.int32)
    if border.triangles:
        vec = (max + M) - (min - M)
        segs = np.array(vec // border.size, dtype=np.int32)
        offset = (((max[0] + M) - (min[0] - M)) / (segs[0] - 1)) * 0.5 # segs - 1 because linspace includes start
        
        tri_edges = np.empty(((segs[0] - 1) * (segs[1] - 1), 2), dtype=np.int32)
        
        cols_rows = np.arange(segs[0] * segs[1])
        cols_rows.shape = (segs[1], segs[0])
        
        edxing = []
        count = 0
        for i in range(segs[1]):
            ar = np.arange(segs[0] - 1)
            if i % 2 == 0:
                ar += count
                edxing += ar.tolist()
            else:
                ar += (count + 1)
                edxing += ar.tolist()
                
            count += segs[0]
        
        nped = np.array(edxing)
        ls = nped[:-(segs[0] - 1)]
        rs = nped[(segs[0] - 1):]
        tri_edges[:, 0] = ls
        tri_edges[:, 1] = rs            
    else:
        vec = (max + M) - (min - M)
        segs = vec // border.size
    
    border.box_min = np.copy(min)
    border.box_max = np.copy(max)

    max += M
    min -= M
    if border.triangles:
        min[0] -= offset
    
    border.cols_rows = (int(segs[1]), int(segs[0]), 3)

    x_vals = np.linspace(min[0], max[0], int(segs[0]))
    y_vals = np.linspace(min[1], max[1], int(segs[1]))
    grid = np.meshgrid(x_vals, y_vals)

    lr_shape = grid[0].shape[0]
    ud_shape = grid[1].shape[1]
    total_shape = lr_shape * ud_shape
    
    grid_co = np.zeros((total_shape, 3), dtype=np.float32)
    grid_co[:, 0] = grid[0].ravel()
    grid_co[:, 1] = grid[1].ravel()
    
    start = 0
    stop = total_shape
    step = ud_shape
    adder = np.arange(start, stop, step)# (not the poisonous kind)
    row = np.zeros((lr_shape, ud_shape - 1, 2), dtype=np.int32)
    ara = np.arange(ud_shape - 1)
    row += ara[:, None]
    row[:, :, 1] += 1
    row += adder[:, None][:, None]
    rs = row.shape

    last_col = row[:, :, 1][:, -1]
    row.shape = (rs[0] * rs[1], 2)

    # ud edges
    start = 0
    stop = total_shape
    step = ud_shape
    adder = np.arange(start, stop, step)# (This time it is the poisonous kind)

    col1 = adder[:-1]
    col2 = adder[1:]
    ed = np.empty((col1.shape[0], 2), dtype=np.int32)
    ed[:, 0] = col1
    ed[:, 1] = col2

    tiled = np.tile(ed.T, ud_shape).T
    ti_shape = tiled.shape
    tiled.shape = (ud_shape, ed.shape[0], 2)
    tiled += np.arange(ud_shape)[:, None][:, None]
    tiled.shape = ti_shape
    
    border.grid_edges = row.tolist() + tiled.tolist() + tri_edges.tolist()

    border.grid_co = grid_co
    border.edge_rows = row

    # consider rotating polyline object so the normal matches the z axis before doing this.


def grid_edge_stuff(border):
    """Generates the grid edges and edge co.
    Offsets every other row to make trianlges."""
    ge = np.array(border.grid_edges, dtype=np.int32)
    nb = border.new_border
    gco = border.grid_co

    if border.triangles:
        offset = (gco[1][0] - gco[0][0]) * 0.5
        gco_shape = border.grid_co.shape
        border.grid_co.shape = border.cols_rows
        border.grid_co[::2][:, :, 0] += offset
        border.grid_co.shape = gco_shape

    border.new_border_edges = np.empty((nb.shape[0], 2), dtype=np.int32)
    idxer = np.arange(nb.shape[0])
    border.new_border_edges[:, 0] = idxer
    border.new_border_edges[:, 1] = np.roll(idxer, -1)

    # border edge co
    border.b_edge_co = nb[border.new_border_edges]
    border.g_edge_co = border.grid_co[border.grid_edges]
    print(border.g_edge_co.shape, "g edge shape")
    print(border.b_edge_co.shape, "b edge shape")

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


def eliminate_duplicate_pairs(ar):
    """Finds unique index pairs assuming left
    and right side are different types:
    [[1, 2], [1, 2], [2, 1]] becomes:
    [[1, 2], [2, 1]]"""
    a = ar
    x = np.array(np.random.rand(a.shape[1]), dtype=np.float32)
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index


def join_objects(obs):
    """Put in a list of objects.
    Everything merges with last object
    in the list."""
    v_counts = [len(ob.data.vertices) for ob in obs]
    ctx = bpy.context.copy()
    ctx['active_object'] = obs[-1]
    #ctx['active_object'] = obs[0]
    ctx['selected_editable_objects'] = obs
    bpy.ops.object.join(ctx)
    return v_counts


def get_linked(obm, idx, op=None):
    """put in the index of a vert. Get everything
    linked just like 'select_linked_pick()'"""
    vboos = np.zeros(len(obm.verts), dtype=np.bool)
    cvs = [obm.verts[idx]]
    escape = False
    while not escape:
        new = []
        for v in cvs:
            if not vboos[v.index]:
                vboos[v.index] = True
                lv = [e.other_vert(v) for e in v.link_edges]
                culled = [v for v in lv if not vboos[v.index]]
                new += culled
        cvs = new
        if len(cvs) == 0:
            escape = True
    idxer = np.arange(len(obm.verts))[vboos]
    if op == "DELETE":    
        verts = [obm.verts[i] for i in idxer]
        bmesh.ops.delete(obm, geom=verts)    
    return idxer


def edge_collisions(border):
    #try:    
        #MC_pierce = bpy.data.texts['MC_pierce.py'].as_module()
    #except:
    #from . MC_29 import MC_pierce
    
    border.MC_pierce.detect_collisions(cloth=None, grid=border)

    #obm = get_bmesh(border.grid_ob, refresh=True)
    obm = border.grid_obm
    
    # find the edge collisions and delete collided edges    
    paired = np.zeros((border.eidx.shape[0], 2), dtype=np.int32)

    paired[:, 0] = border.eidx
    paired[:, 1] = border.tidx
    
    ar, index = eliminate_duplicate_pairs(paired)
    ar = paired
    
    b = border.b_edge_co[ar[:, 0]]
    g = border.g_edge_co[ar[:, 1]]    

    #b = border.b_edge_co[border.eidx]
    #g = border.g_edge_co[border.tidx]
    
    b1 = b[:, 0][:, :2]
    b2 = b[:, 1][:, :2]
    g1 = g[:, 0][:, :2]
    g2 = g[:, 1][:, :2]
            
    hits, boolies = edges_edges_intersect_2d(b1,b2, g1,g2, intersect=True)
    boolies = new_intersect(b1,b2, g1,g2)

    border.collide_locs = hits[boolies]
    border.collide_edges = ar[:, 1][boolies]
    uni_ed = np.unique(border.collide_edges)

    geom = [obm.edges[e] for e in uni_ed]
    bmesh.ops.delete(obm, geom=geom, context='EDGES')
    
    obm.verts.ensure_lookup_table()
    obm.edges.ensure_lookup_table()
    
    # delete everything outside the line. Certain shapes can create islands
    border.grid_co = np.array([v.co for v in obm.verts], dtype=np.float32)
    vidx = np.arange(len(obm.verts))
    idx_bool = np.ones(len(obm.verts), dtype=np.bool)
    del_bool = np.zeros(len(obm.verts), dtype=np.bool)
    out = get_linked(obm, 0)#, op="DELETE")
    idx_bool[out] = False
    del_bool[out] = True
    
    for i in range(len(obm.verts)):    
        rem = vidx[idx_bool]

        if len(rem) == 0:
            break

        v2 = rem[0]
        
        out_in = cull_outside(border, verts=[v2])
        deselect(border.grid_ob, v2)
        l2 = get_linked(obm, v2)

        if out_in:
            del_bool[l2] = True
        idx_bool[l2] = False
    verts = [obm.verts[i] for i in vidx[del_bool]]
    bmesh.ops.delete(obm, geom=verts)
    
    # write bmesh back to object
    if obm.is_wrapped:
        bmesh.update_edit_mesh(border.grid_ob.data, False, False)
    else:
        obm.to_mesh(border.grid_ob.data)
        border.grid_ob.data.update()
    
    # Add the border to the bmesh
    gl = len(border.grid_ob.data.edges)
    bl = len(border.border_ob.data.edges)
    g, b = join_objects([border.border_ob, border.grid_ob])

    if border.p1:    
        border.sew_relationships['border_verts'] += [[g, b]]
    
    border.border_edges = np.arange(bl) + gl


def cull_outside(border, verts=None):
    """Finds if points are inside a polyline"""
    co = border.grid_co
    
    bb_max = np.max(co, axis=0)
    eco = border.b_edge_co[:, :, :2]
    
    a = np.empty((eco.shape[0], 2), dtype=np.float32)
    a[:] = co[verts[0]][:2] 
    b = np.empty((eco.shape[0], 2), dtype=np.float32)
    b[:] = bb_max[:2] 
    c = eco[:, 0]
    d = eco[:, 1]
    
    ne = new_intersect(a, b, eco[:, 0], eco[:, 1])
    return np.sum(ne) % 2 == 0


def loner_perfect(v, v1, v2, fill):
    """Takes verts in the grid that are boundary
    and looks for places in the border to make edges"""
    obm = fill.obm

    bv1 = obm.verts[v1]
    bv2 = obm.verts[v2]
    ev1 = [e.other_vert(bv1).index for e in bv1.link_edges if e.index == -1]
    ev2 = [e.other_vert(bv2).index for e in bv2.link_edges if e.index == -1]
    
    unicorns = np.unique(ev1 + ev2)

    gap = False
    if unicorns.shape[0] >= 4:
        done = False
        for e in ev1:
            le = [ed for ed in obm.verts[e].link_edges if ed.other_vert(obm.verts[e]).index in ev2]
            if len(le) == 1:
                fill.perfect_edges += [[v, le[0].verts[0].index]]
                fill.perfect_edges += [[v, le[0].verts[1].index]]
                done = True
        if done:    
            return
        else:
            gap = True
        
    if unicorns.shape[0] == 3:
        special = [e for e in unicorns if (e in ev1) & (e in ev2)]    
        if len(special) == 1:    
            fill.perfect_edges += [[v, special[0]]]
            return

    if gap:
        reach2 = [[e.other_vert(obm.verts[u]).index for e in obm.verts[u].link_edges] for u in unicorns]
        merge = []
        for i in reach2:
            merge += i
        mergicorn, c = np.unique(merge, return_counts=True)
        multi = mergicorn[c > 1]
        in_b = [m for m in multi if m >= fill.border_verts[0]]
        
        if len(in_b) == 1:
            fill.perfect_edges += [[v, in_b[0]]]        
            return
            
    if (len(ev1) == 0) | (len(ev2) == 0):
        #print(unicorns, "unicorns")
        fill.gappy += [v]
        vle = [e.other_vert(obm.verts[v]).index for e in obm.verts[v].link_edges if len(e.link_faces) == 1]
        for ve in vle:
            nle = [e for e in obm.verts[ve].link_edges if e.index == -1]
            if len(nle) == 0:
                vee = [e.other_vert(obm.verts[ve]) for e in obm.verts[ve].link_edges if len(e.link_faces) == 1]    
                veev = [ver for ver in vee if ver.index != v]
                if len(veev) == 1:
                    ototv = veev[0]
                    brde = [e.other_vert(ototv).index for e in ototv.link_edges if e.index == -1]
                    match = np.array(brde)[np.isin(brde, unicorns)]
                    if len(match) == 1:
                        fill.perfect_edges += [[v, match[0]]]
                        return

    if len(ev1) == 1:
        fill.perfect_edges += [[v, ev1[0]]]
        return
        
    if len(ev2) == 1:
        fill.perfect_edges += [[v, ev2[0]]]


def make_edge_2(fill):
    """Find loner verts where surrounding verts
    both have two new edges."""

    obm = fill.obm
    npfb = np.array(fill.boundary_verts)

    e_counts = [[e.index for e in obm.verts[v].link_edges if len(e.link_faces) == 0] for v in fill.boundary_verts]
    otvs = [[e.other_vert(obm.verts[v]).index for e in obm.verts[v].link_edges if len(e.link_faces) == 0] for v in fill.boundary_verts]
        
    fill.original_loners = []
    fill.loners = []
    fill.double_loners = []
    fill.perfect_edges = [] # for perfect loner case

    boo = [len(otvs[i]) == 0 for i in range(len(fill.boundary_verts))]
    npb = np.array(fill.boundary_verts)
    loners = npb[boo]
    
    for v in loners:    
        #if len(otv) == 0: # this vert has no new edges connected to it. It's a loner
        v_otvs = [e.other_vert(obm.verts[v]).index for e in obm.verts[v].link_edges if len(e.link_faces) == 1]
        if len(v_otvs) == 2: # this vert is connected to only two boundary edges from the original grid
            v1 = v_otvs[0]
            v2 = v_otvs[1]
            loner_perfect(v, v1, v2, fill)
        else:
            print("found a weird loner with more than two linked boundary edges")            
            fill.loners += [v]
            fill.double_loners += [v]
        
    fill.original_loners = loners    


def make_edge(fill):
    """Walk around the boundary verts and try to make
    edges without crossing any edges."""
    size = fill.grid_size

    vvv = []
    close_vvv = []
    new_edges = []

    for v in fill.boundary_verts:
        dif = fill.border_co - fill.grid_co[v]
        dist = np.sqrt(np.einsum('ij, ij->i', dif, dif))
        edge_close = dist <= fill.merge_dist

        if np.any(edge_close):
            
            close_verts = fill.border_verts[edge_close]
            lee = []
            for bv in close_verts:
                lee += [[e.verts[0].index, e.verts[1].index,] for e in fill.obm.verts[bv].link_edges]
            nplee = np.array(lee)
                        
            edges = np.empty((close_verts.shape[0], 2), dtype=np.int32)
            edges[:, 0] = v
            edges[:, 1] = close_verts
            
                
            cull_edges = []
            for e in edges:
                otv = e[1]
                shared = (otv == nplee) | (v == nplee)
                share_cull = np.any(shared, axis=1)
                npleec = nplee[~share_cull]
                
                # collide this edge with nplee
                eco = fill.grid_co[e]
                edges_co = fill.grid_co[npleec]
                edges_co2 = np.empty(edges_co.shape, dtype=np.float32)
                edges_co2[:] = eco

                a1 = edges_co2[:, 0][:, :2]
                a2 = edges_co2[:, 1][:, :2]
                b1 = edges_co[:, 0][:, :2]
                b2 = edges_co[:, 1][:, :2]
                
                int2 = new_intersect(a1,a2,b1,b2)
                
                if np.all(~int2):
                    new_edges += [e.tolist()]
                                                
    return new_edges
    

class Fill():
    def __init__(self, border=None):
        
        self.name = 'fill'
        self.border_edges = border.border_edges
        self.smooth_iters = border.smooth_iters
        self.ob = border.grid_ob
        self.obm = get_bmesh(self.ob, refresh=True)
        self.v_count = len(self.obm.verts)
        self.vidx = np.arange(self.v_count)
        self.angle = border.angle
        self.triangles = border.triangles
        
        dif = self.v_count - border.border_v_count
        self.border_verts = np.arange(border.border_v_count) + dif
        
        self.boundary_verts = [v.index for v in self.obm.verts if v.is_boundary]
        
        self.v_boo = np.ones(self.v_count, dtype=np.bool)
        self.v_boo[self.boundary_verts] = False
        self.v_boo[dif:] = False
        
        self.floaters = [v for v in self.vidx[self.v_boo] if len(self.obm.verts[v].link_faces) == 0]
        self.boundary_verts = self.boundary_verts + self.floaters
        
        inner_boo = np.ones(self.vidx.shape[0], dtype=np.bool)
        inner_boo[self.border_verts] = False
        self.inner_verts = self.vidx[inner_boo]
        
        self.grid_co = np.zeros((self.vidx.shape[0], 3), dtype=np.float32)
        self.ob.data.vertices.foreach_get('co', self.grid_co.ravel())
        self.border_co = self.grid_co[self.border_verts]

        self.merge_dist = border.size * 1.5
        self.grid_size = border.size


def edge_delete_pass(fill):
    """Goes through the new edges and deletes certain ones
    so that none overlap."""
    re = fill.recent_edges
    obm = fill.obm
    eidx = np.array([[e.verts[0].index, e.verts[1].index] for e in obm.edges])    

    # this is the indices of the edges in the border
    beidx = fill.border_edges
    
    # this is the indices of the edges prior to adding the border
    first_edges = np.arange(beidx[0])

    # this is the indices of the new edges
    so_far = len(beidx) + len(first_edges)
    new_edges = np.arange(len(obm.edges) - so_far) + so_far
    
    edges_no_border = np.append(first_edges, new_edges)
    edge_tracking = np.copy(edges_no_border)

    grid_co = np.array([v.co for v in obm.verts], dtype=np.float32)
    
    # this is eidx for the edges excluding the border edges
    check_eidx = eidx[edges_no_border]
    check_edge_co = grid_co[check_eidx]
    
    # new_edges is the edge index for the edges we created
    fill.bad_edges = []
    fill.other_edges = []
    fill.removable_edges = []
    for i in range(new_edges.shape[0]):
        ne_idx = new_edges[i]
        
        # can eliminate checks of edges that share a vert
        this_eidx = eidx[ne_idx]    
        remaining_eidx = check_eidx#[edge_tracking]

        ed_share = np.any((this_eidx[0] == remaining_eidx) | (this_eidx[0] == remaining_eidx), axis=1)        
        cull1 = remaining_eidx[~ed_share]
        
        check_eco = grid_co[cull1]
        tiled = np.empty_like(check_eco)
        tiled[:] = grid_co[this_eidx]

        a1 = tiled[:, 0][:, :2]
        a2 = tiled[:, 1][:, :2]
        b1 = check_eco[:, 0][:, :2]
        b2 = check_eco[:, 1][:, :2]
        
        int2 = new_intersect(a1,a2,b1,b2)
        other_edges = edge_tracking[~ed_share][int2]
        
        if np.any(int2):
            fill.bad_edges += [ne_idx] * other_edges.shape[0]
            fill.other_edges += other_edges.tolist()

    obm.edges.ensure_lookup_table()
    
    # it ends up being unique becase the keys can't be duplicated
    cull = {i:True for i in fill.bad_edges + fill.other_edges}
    
    for i in range(len(fill.bad_edges)):
        be = fill.bad_edges[i]
        oe = fill.other_edges[i]
        
        if cull[be] & cull[oe]:
            # if there is a face edge delete the other one        
            if len(obm.edges[oe].link_faces) > 0:
                fill.removable_edges += [be]
                cull[be] = False
                continue
            
            # if they are both new edges delete the longer one
            be_co = grid_co[eidx[be]]
            oe_co = grid_co[eidx[oe]]
            bev = be_co[1] - be_co[0]
            oev = oe_co[1] - be_co[0]
            
            bel = bev @ bev
            oel = oev @ oev
            
            arso = np.argsort([bel, oel])[0]
            if arso == 0:
                fill.removable_edges += [be]
                cull[be] = False
                continue

            fill.removable_edges += [oe]
            cull[oe] = False


def get_inside_sharps(fill):
    """Identify sharps where the inside of the angle
        should be filled in by a face. Inside versus
        outside angles."""
    
    co = fill.border_co
    
    v1 = np.roll(co, 1, axis=0) - co
    v2 = np.roll(co, -1, axis=0) - co
    
    # use the vecs pointing away from each vertex later
    ls_dots = np.einsum('ij, ij->i', v1, v1)
    rs_dots = np.einsum('ij, ij->i', v2, v2)
    
    uv1 = v1 / np.sqrt(ls_dots)[:, None]
    uv2 = v2 / np.sqrt(rs_dots)[:, None]
    
    crosses = np.cross(uv1, uv2)
    cross_dots = np.einsum('ij, ij->i', crosses, crosses)
    
    con = np.pi / 180
    limit = np.cos(con * (180 - fill.angle))
    
    angle = np.einsum('ij, ij->i', uv1, uv2)
    sharps = angle > limit
    
    fill.sharp_normals = np.sign((crosses[sharps][:, 2]))
    fill.normals = np.sign((crosses[:, 2]))
    up_down = np.sum(cross_dots * fill.normals)    
    outies = (fill.sharp_normals * up_down) < 0
    
    sharp_idx = np.arange(co.shape[0])[sharps]
    fill.sharp_idx = sharp_idx
    fill.outies_idx = sharp_idx[outies]
    fill.innies_idx = sharp_idx[~outies]
            

def make_faces(fill):
    obm = fill.obm
    
    # net edges makes a giant face inside the whole border. Have to kill it.    
    net_edges = [e for e in obm.edges if len(e.link_faces) < 2]
    net_faces = bmesh.ops.edgenet_fill(obm, edges=net_edges, mat_nr=0, use_smooth=False, sides=0)
    big_face = [f for f in net_faces['faces'] if len(f.verts) == fill.border_co.shape[0]]
    
    for f in big_face:
        #fvs = [v.index for v in f.verts]
        #if np.all(np.isin(fvs, fill.border_verts)):
        obm.faces.remove(f)
        
    if fill.triangles:
        bmesh.ops.triangulate(obm, faces=obm.faces)    
    

def fill_border(border):
    
    fill = Fill(border)    
    new_edges = make_edge(fill)
    
    for e in new_edges:
        fill.obm.edges.new((fill.obm.verts[e[0]], fill.obm.verts[e[1]]))    
    
    perfect = False
    loners = []
    p_edges = []
    if perfect:
        fill.gappy = []
        make_edge_2(fill)
        if len(fill.loners) > 0:
            edss, edsidx = pairs_idx(np.array(fill.perfect_edges))
            for e in edss:
                fill.obm.edges.new((fill.obm.verts[e[0]], fill.obm.verts[e[1]]))
            loners = fill.original_loners
        
            if len(fill.perfect_edges) > 0:
                rlb = ~np.isin(fill.original_loners, np.array(fill.perfect_edges)[:, 0])
                loners = np.array(fill.original_loners)[rlb]
            p_edges = edss.tolist()
    
        fill.recent_edges = new_edges + p_edges
    else:    
        fill.recent_edges = new_edges
        
    smooth_v = [fill.obm.verts[v] for v in fill.inner_verts]

    edge_delete_pass(fill)
    bad_edges = [fill.obm.edges[e] for e in fill.removable_edges]
    bmesh.ops.delete(fill.obm, geom=bad_edges, context='EDGES')
    
    make_faces(fill)

    for i in range(fill.smooth_iters):
        bmesh.ops.smooth_vert(fill.obm, verts=smooth_v, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
            
    if fill.obm.is_wrapped:
        bmesh.update_edit_mesh(border.grid_ob.data, False, False)
    else:
        fill.obm.to_mesh(border.grid_ob.data)
        border.grid_ob.data.update()
        

def create_faces(border):
    c = border.cols_rows[0] # 73, 14 (cols rows)
    r = border.cols_rows[1] # 73, 14 (cols rows)
    if border.triangles:
        bottom_row = np.empty((r - 1, 3), dtype=np.int32)
        bridx = np.arange(r-1)
        bottom_row[:, 0] = bridx
        bottom_row[:, 1] = bridx + r
        bottom_row[:, 2] = bridx + r + 1

        bottom_row_2 = np.empty((r - 1, 3), dtype=np.int32)
        bottom_row_2[:, 0] = bridx + 1
        bottom_row_2[:, 1] = bridx
        bottom_row_2[:, 2] = bridx + r + 1

        br = np.empty(((r - 1) * 2, 3), dtype=np.int32)
        br[:r-1] = bottom_row
        br[r-1:] = bottom_row_2

        brsc = c//2 # bottom rows count
        brs = np.empty((brsc, (r - 1) * 2, 3), dtype=np.int32)
        brs[:, None] = br
        adder = np.arange(0, r * 2 * brsc, r * 2)
        brs += adder[:, None][:, None]
        
        # top row calc ------------------
        trsc = brsc + ((c % 2) -1) # column count - bottom rows count. Could be odd or even
        trs = np.empty((trsc, (r - 1) * 2, 3), dtype=np.int32)

        top_row = np.empty((r - 1, 3), dtype=np.int32)
        tridx = np.arange(r-1)
        top_row[:, 0] = tridx + r + 1
        top_row[:, 1] = tridx + r
        top_row[:, 2] = tridx + r * 2

        top_row_2 = np.empty((r - 1, 3), dtype=np.int32)
        top_row_2[:, 0] = tridx + r + 1
        top_row_2[:, 1] = tridx + r * 2
        top_row_2[:, 2] = tridx + r * 2 + 1

        tr = np.empty(((r - 1) * 2, 3), dtype=np.int32)
        tr[:r-1] = top_row
        tr[r-1:] = top_row_2
        trs[:, None] = tr
        adder = np.arange(0, r * 2 * trsc, r * 2)
        trs += adder[:, None][:, None]    
        
        # -------------------------------
        brs.shape = (brs.ravel().shape[0] //3 , 3)
        trs.shape = (trs.ravel().shape[0] //3 , 3)
        border.br = brs
        border.tr = trs
        return
    
    q_face = np.empty((c -1, r - 1, 4), dtype=np.int32)
    br = np.empty((r-1, 4), dtype=np.int32)
    aridx = np.arange(r - 1)
    br[:, 0] = aridx
    br[:, 1] = aridx + r
    br[:, 2] = aridx + r + 1
    br[:, 3] = aridx + 1
    
    adder = np.arange(0, (c - 1) * r, r)
    q_face[:, None] = br
    q_face += adder[:, None][:, None]
    q_face.shape = (q_face.ravel().shape[0] // 4, 4)
    border.q_face = q_face


# p1 stuff --------------------
def get_flap_verts():
    """Use json file to get the verts to exclude"""
    pass

def divide_into_panels():
    """Use vertex groups to identify panels"""
    pass


def generate_border():
    """For each panel create a border respecting
    sharp turns."""
    pass


def generate_grid_sew_lines():
    """Folds"""
    pass

# grid from border ------------
class Border():
    name = "Border"
    
    def __init__(self, ob, make=True, p_mod=None):
        # include the object when creating instance
        # -----------------------
        self.MC_pierce = p_mod
        self.p1 = False
        self.ob = ob
        self.size = ob.MC_props.grid_size
        self.angle = ob.MC_props.grid_angle_limit
        self.triangles = ob.MC_props.grid_triangles
        self.smooth_iters = ob.MC_props.grid_smoothing
        
        self.edges = np.empty((len(ob.data.edges), 2), dtype=np.int32)
        ob.data.edges.foreach_get('vertices', self.edges.ravel())
        self.ordered = get_ordered_loop(ob, self.edges)            

        cut_polyline = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
        ob.data.vertices.foreach_get('co', cut_polyline.ravel())
        self.ordered_co = cut_polyline[self.ordered]
        
        self.new_border = redistribute(self.ordered_co[:, :2], grid_size=self.size, angle=self.angle)
        
        generate_grid(self)
        
        self.border_v_count = self.new_border.shape[0]
        
        self.cull_verts = np.zeros(self.grid_co.shape[0], dtype=np.bool)
        self.cull_idxer = np.arange(self.grid_co.shape[0])
        
        # offset every other row and create border edges
        grid_edge_stuff(self)
        # ----------------------------------------------
        
        create_faces(self)
        if make:    
            make_objects(self)
        
        cut = False
        cut = True
        if cut:    
            edge_collisions(self)
        
        fill_border(self)
        bpy.context.view_layer.update()


class V_Border():
    name = "Border"
    
    def __init__(self, ob, make=True, size=0.5, ordered=None, border_co=None, grid=None):
        """the 'grid' arg is a class instance from the other module"""
        # include the object when creating instance
        # -----------------------
        print(len(ob.data.vertices), "a v count")
        self.p1 = True
        self.ob = ob
        self.new_ob = grid.new_ob
        self.new_obm = grid.new_obm
        self.new_sew_edges = grid.new_sew_edges
        self.test_attritbute = grid.test_attritbute
        print(self.test_attritbute, "test attribute")
        self.size = size
        self.inner_size = grid.inner_size
        self.angle = 10
        self.triangles = True
        self.smooth_iters = 7
        #self.edges = edges
        self.ordered = ordered
        self.sew_relationships = grid.sew_relationships
        self.iter_count = grid.iter_count
        self.accumulated_border_count = grid.accumulated_border_count
        self.ordered_co = border_co
        
        self.new_border = redistribute(self.ordered_co[:, :2], grid_size=self.size, angle=self.angle, v_border=self)
        generate_grid(self)
        
        self.border_v_count = self.new_border.shape[0]
        
        self.cull_verts = np.zeros(self.grid_co.shape[0], dtype=np.bool)
        self.cull_idxer = np.arange(self.grid_co.shape[0])
        
        # offset every other row and create border edges
        grid_edge_stuff(self)
        # ----------------------------------------------
        
        create_faces(self)
        if make:    
            make_objects(self, ob)
        
        cut = False
        cut = True
        if cut:
            edge_collisions(self)
        
        fill_border(self)
        
        # need this??
        bpy.context.view_layer.update()
        
            
        ob = self.grid_ob
        faces = [[v + grid.face_offset for v in p.vertices] for p in ob.data.polygons]
        co = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
        ob.data.vertices.foreach_get('co', co.ravel())
        grid.face_offset += co.shape[0]
        
        grid.full_faces += faces
        
        #this should be what I need to get the panels of the new grid
        grid.panel_indices[grid.panel] = np.arange(co.shape[0]) + len(grid.full_vertices)
        
        grid.full_vertices += co.tolist()        
        self.accumulated_border_count = len(grid.full_vertices)
        grid.accumulated_border_count = len(grid.full_vertices)
        grid.new_sew_edges += self.new_sew_edges
        
        
        if grid.done:
            idx = 0
            total_b_verts = []
            for g in grid.sew_relationships['border_verts']:
                total_b_verts += (np.arange(g[0]) + idx + g[1]).tolist()
                
                idx += (g[0] + g[1])
                
            npfv = np.array(grid.sew_relationships['no_fold_verts'])
            nptv = np.array(grid.sew_relationships['three_verts'])
            npbv = np.array(total_b_verts)
            bool = np.zeros(npfv.shape[0], dtype=np.bool)
            
            other_verts = []
            new_edges = []
            for i, j in zip(npfv, npbv):
                bool[:] = False
                vert = self.new_obm.verts[i]
                lv = [e.other_vert(vert).index for e in vert.link_edges if len(e.link_faces) == 0]
                other_verts += lv
                
                for v in lv:
                    bool[np.any(nptv == v, axis=1)] = True
                    
                match = npbv[bool]

                for m in match:
                    new_edges += [[j, m]]              
                
            ob = link_mesh(grid.full_vertices, edges=new_edges, faces=grid.full_faces, name='name', ob=ob)
            grid.new_grid_ob = ob
