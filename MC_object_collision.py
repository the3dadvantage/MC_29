import bpy
import numpy as np
import bmesh
import time

def timer(t, name='name'):
    ti = bpy.context.scene.timers
    if name not in ti:
        ti[name] = 0.0
    ti[name] += t


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
def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate"""
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] / np.array(ob.scale, dtype=np.float32) # rotates backwards without T
    #print(mat)
    return (co @ mat) / np.array(ob.scale, dtype=np.float32)


# universal ---------------------
def absolute_co(ob, co=None):
    """Get vert coords in world space with modifiers"""
    co, proxy = get_proxy_co(ob, co, return_proxy=True)
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc, proxy


def apply_transforms(ob, co):
    """Get vert coords in world space"""
    m = np.array(ob.matrix_world, dtype=np.float32)
    mat = m[:3, :3].T # rotates backwards without T
    loc = m[:3, 3]
    return co @ mat + loc


def select_edit_mode(sc, ob, idx, type='v', deselect=False, obm=None):
    """Selects verts in edit mode and updates"""
    
    if ob.data.is_editmode:
        if obm is None:
            obm = bmesh.from_edit_mesh(ob.data)
            obm.verts.ensure_lookup_table()
        
        if type == 'v':
            x = obm.verts
        if type == 'f':
            x = obm.faces
        if type == 'e':
            x = obm.edges
        
        if deselect:
            for i in x:
                i.select = False
        
        for i in idx:
            sc.select_counter[i] += 1
            x[i].select = True
        
        if obm is None:
            bmesh.update_edit_mesh(ob.data)
        #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


def bmesh_proxy(ob):
    """Get a bmesh contating modifier effects"""
    dg = bpy.context.evaluated_depsgraph_get()
    prox = ob.evaluated_get(dg)
    proxy = prox.to_mesh()
    obm = bmesh.new()
    obm.from_mesh(proxy)
    return obm
    

def get_proxy_co(ob, co=None, proxy=None, return_proxy=False, return_normals=False):
    """Gets co with modifiers like cloth"""
    if proxy is None:
        dg = bpy.context.evaluated_depsgraph_get()
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    if return_proxy:
        return co, proxy
    
    proxy.to_mesh_clear()
    #ob.to_mesh_clear()
    return co


def get_edges(ob, fake=False):
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


def get_faces(ob):
    """Only works on triangle mesh."""
    fa = np.empty((len(ob.data.polygons), 3), dtype=np.int32)
    ob.data.polygons.foreach_get('vertices', fa.ravel())
    return fa


def get_tridex(ob, tobm=None):
    """Return an index for viewing the verts as triangles"""
    free = True
    if ob.data.is_editmode:
        ob.update_from_editmode()
    if tobm is None:
        tobm = bmesh.new()
        tobm.from_mesh(ob.data)
        free = True
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:])
    tridex = np.array([[v.index for v in f.verts] for f in tobm.faces], dtype=np.int32)
    if free:
        tobm.free()
    return tridex


def inside_triangles(tris, points, margin=0.0):#, cross_vecs):
    """Checks if points are inside triangles"""
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

    w = 1 - (u+v)
    # !!!! needs some thought
    margin = 0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)
    
    return check, weights


def b2(sc, cloth, count):
    #print('running b2', count)
    if len(sc.big_boxes) == 0:
        print("ran out")
        return

    boxes = []
    for oct in sc.big_boxes:
        t = oct[0]
        e = oct[1]
        b = oct[2]
                
        tfull, efull, bounds = octree_et(sc, margin=0.0, idx=t, eidx=e, bounds=b, cloth=cloth)
        
        for i in range(len(tfull)):
            t = tfull[i]
            e = efull[i]
            bmin = bounds[0][i]
            bmax = bounds[1][i]
            
            if (t.shape[0] < sc.box_max) | (e.shape[0] < sc.box_max):
                sc.small_boxes.append([t, e])
            else:
                boxes.append([t, e, [bmin, bmax]])            
    sc.big_boxes = boxes
    

def generate_bounds(minc, maxc):
    """from a min corner and a max corner
    generate the min and max corner of 8 boxes"""

    diag = (maxc - minc) / 2
    mid = minc + diag
    mins = np.zeros((8,3), dtype=np.float32) 
    maxs = np.zeros((8,3), dtype=np.float32) 

    # blf
    mins[0] = minc
    maxs[0] = mid
    # brf
    mins[1] = minc
    mins[1][0] += diag[0]
    maxs[1] = mid
    maxs[1][0] += diag[0]
    # blb
    mins[2] = minc
    mins[2][1] += diag[1]
    maxs[2] = mid
    maxs[2][1] += diag[1]
    # brb
    mins[3] = mins[2]
    mins[3][0] += diag[0]
    maxs[3] = maxs[2]
    maxs[3][0] += diag[0]
    # tlf
    mins[4] = mins[0]
    mins[4][2] += diag[2]
    maxs[4] = maxs[0]
    maxs[4][2] += diag[2]
    # trf
    mins[5] = mins[1]
    mins[5][2] += diag[2]
    maxs[5] = maxs[1]
    maxs[5][2] += diag[2]
    # tlb
    mins[6] = mins[2]
    mins[6][2] += diag[2]
    maxs[6] = maxs[2]
    maxs[6][2] += diag[2]
    # trb
    mins[7] = mins[3]
    mins[7][2] += diag[2]
    maxs[7] = maxs[3]
    maxs[7][2] += diag[2]
    
    return mid, [mins, maxs]


def octree_et(sc, margin, idx=None, eidx=None, bounds=None, cloth=None):
    """Adaptive octree. Good for finding doubles or broad
    phase collision culling. et does edges and tris.
    Also groups edges in boxes.""" # first box is based on bounds so first box could be any shape rectangle

    T = time.time()
    margin = sc.M # might be faster than >=, <=
    margin = 0.0 #sc.M # might be faster than >=, <=

    #co = cloth.oc_co

    if bounds is None:
        #b_min = np.min(co, axis=0)
        #b_max = np.max(co, axis=0)
        b_min = np.min(np.min(cloth.traveling_edge_co, 0), 0)
        b_max = np.max(np.max(cloth.traveling_edge_co, 0), 0)
    
    
    else:
        b_min, b_max = bounds[0], bounds[1]

        #eco = co[sc.ed[eidx].ravel()]
        #b_min = np.min(eco, axis=0)
        #b_max = np.max(eco, axis=0)
        
    # bounds_8 is for use on the next iteration.
    mid, bounds_8 = generate_bounds(b_min, b_max)
    
    #mid = b_min + ((b_max - b_min) / 2)
    mid_ = mid #+ margin
    _mid = mid #- margin

    x_, y_, z_ = mid_[0], mid_[1], mid_[2]
    _x, _y, _z = _mid[0], _mid[1], _mid[2]

    # tris
    xmax = sc.txmax
    xmin = sc.txmin

    ymax = sc.tymax
    ymin = sc.tymin

    zmax = sc.tzmax
    zmin = sc.tzmin

    # edges
    exmin = sc.exmin
    eymin = sc.eymin
    ezmin = sc.ezmin
    
    exmax = sc.exmax
    eymax = sc.eymax
    ezmax = sc.ezmax

    # l = left, r = right, f = front, b = back, u = up, d = down
    if idx is None:
        idx = np.arange(sc.tris6.shape[0], dtype=np.int32)
        #idx = cloth.oc_indexer
    if eidx is None:    
        eidx = cloth.oc_eidx

    idx = np.array(idx, dtype=np.int32)
    eidx = np.array(eidx, dtype=np.int32)

    # -------------------------------
    B = xmin[idx] <= x_# + margin
    il = idx[B]

    B = xmax[idx] >= _x# - margin
    ir = idx[B]
    
    # edges
    eB = exmin[eidx] <= x_# + margin
    eil = eidx[eB]

    eB = exmax[eidx] >= _x# - margin
    eir = eidx[eB]

    # ------------------------------
    B = ymax[il] >= _y# - margin
    ilf = il[B]

    B = ymin[il] <= y_# + margin
    ilb = il[B]

    B = ymax[ir] >= _y# - margin
    irf = ir[B]

    B = ymin[ir] <= y_# + margin
    irb = ir[B]
    
    # edges
    eB = eymax[eil] >= _y# - margin
    eilf = eil[eB]

    eB = eymin[eil] <= y_# + margin
    eilb = eil[eB]

    eB = eymax[eir] >= _y# - margin
    eirf = eir[eB]

    eB = eymin[eir] <= y_# + margin
    eirb = eir[eB]

    # ------------------------------
    B = zmax[ilf] >= _z# - margin
    ilfu = ilf[B]
    B = zmin[ilf] <= z_# + margin
    ilfd = ilf[B]

    B = zmax[ilb] >= _z# - margin
    ilbu = ilb[B]
    B = zmin[ilb] <= z_# + margin
    ilbd = ilb[B]

    B = zmax[irf] >= _z# - margin
    irfu = irf[B]
    B = zmin[irf] <= z_# + margin
    irfd = irf[B]

    B = zmax[irb] >= _z# - margin
    irbu = irb[B]
    B = zmin[irb] <= z_# + margin
    irbd = irb[B]

    # edges
    eB = ezmax[eilf] >= _z# - margin
    eilfu = eilf[eB]
    eB = ezmin[eilf] <= z_# + margin
    eilfd = eilf[eB]

    eB = ezmax[eilb] >= _z# - margin
    eilbu = eilb[eB]
    eB = ezmin[eilb] <= z_# + margin
    eilbd = eilb[eB]

    eB = ezmax[eirf] >= _z# - margin
    eirfu = eirf[eB]
    eB = ezmin[eirf] <= z_# + margin
    eirfd = eirf[eB]

    eB = ezmax[eirb] >= _z# - margin
    eirbu = eirb[eB]
    eB = ezmin[eirb] <= z_# + margin
    eirbd = eirb[eB]    

    boxes = [ilbd, irbd, ilfd, irfd, ilbu, irbu, ilfu, irfu]
    eboxes = [eilbd, eirbd, eilfd, eirfd, eilbu, eirbu, eilfu, eirfu]
    
    bbool = np.array([i.shape[0] > 0 for i in boxes])
    ebool = np.array([i.shape[0] > 0 for i in eboxes])
    both = bbool & ebool
    
    full = np.array(boxes, dtype=np.object)[both]
    efull = np.array(eboxes, dtype=np.object)[both]

    return full, efull, [bounds_8[0][both], bounds_8[1][both]]
    

def total_bounds(sc, cloth):
    
    bool = cloth.tris6_bool
    bool[:] = True
    
    #box_min = np.min(cloth.oc_co, axis=0)# - sc.M
    #box_max = np.max(cloth.oc_co, axis=0)# + sc.M
    box_min = np.min(np.min(cloth.traveling_edge_co, axis=0), axis=0)# - sc.M
    box_max = np.max(np.max(cloth.traveling_edge_co, axis=0), axis=0)# + sc.M
    
    bool[sc.txmax <= box_min[0]] = False
    bool[sc.txmin >= box_max[0]] = False

    bool[sc.tymax <= box_min[1]] = False
    bool[sc.tymin >= box_max[1]] = False

    bool[sc.tzmax <= box_min[2]] = False
    bool[sc.tzmin >= box_max[2]] = False

    return bool


def object_collisions_7(sc, margin=0.0, cloth=None):
    
    margin = 0.0
    #margin = 0.0
    
    #print(cloth.traveling_edge_co[0])
    
    tx = cloth.oc_tris_six[:, :, 0]
    ty = cloth.oc_tris_six[:, :, 1]
    tz = cloth.oc_tris_six[:, :, 2]

    txmax = np.max(tx, axis=1)# + margin
    txmin = np.min(tx, axis=1)# - margin

    tymax = np.max(ty, axis=1)# + margin
    tymin = np.min(ty, axis=1)# - margin

    tzmax = np.max(tz, axis=1)# + margin
    tzmin = np.min(tz, axis=1)# - margin

    sc.txmax = txmax
    sc.txmin = txmin

    sc.tymax = tymax
    sc.tymin = tymin

    sc.tzmax = tzmax
    sc.tzmin = tzmin

    # cloth box cull
    cloth.boxboo = total_bounds(sc, cloth)
    tris6 = cloth.oc_tris_six[cloth.boxboo]

    sc.finished = False
    sc.tris6 = tris6
    if tris6.shape[0] == 0:
        sc.finished = True
        return
    
    tx = tris6[:, :, 0]
    ty = tris6[:, :, 1]
    tz = tris6[:, :, 2]

    margin = 0.00001

    txmax = np.max(tx, axis=1) + margin
    txmin = np.min(tx, axis=1) - margin

    tymax = np.max(ty, axis=1) + margin
    tymin = np.min(ty, axis=1) - margin

    tzmax = np.max(tz, axis=1) + margin
    tzmin = np.min(tz, axis=1) - margin

    sc.txmax = txmax
    sc.txmin = txmin

    sc.tymax = tymax
    sc.tymin = tymin

    sc.tzmax = tzmax
    sc.tzmin = tzmin

    # edge bounds:
    ex = cloth.traveling_edge_co[:, :, 0]
    ey = cloth.traveling_edge_co[:, :, 1]
    ez = cloth.traveling_edge_co[:, :, 2]

    #margin = 0.1

    sc.exmin = np.min(ex, axis=1) - margin
    sc.eymin = np.min(ey, axis=1) - margin
    sc.ezmin = np.min(ez, axis=1) - margin
    
    sc.exmax = np.max(ex, axis=1) + margin
    sc.eymax = np.max(ey, axis=1) + margin
    sc.ezmax = np.max(ez, axis=1) + margin
        
    tfull, efull, bounds = octree_et(sc, margin=0.0, cloth=cloth)

    for i in range(len(tfull)):
        t = tfull[i]
        e = efull[i]
        bmin = bounds[0][i]
        bmax = bounds[1][i]
        
        if (t.shape[0] < sc.box_max) | (e.shape[0] < sc.box_max):
            sc.small_boxes.append([t, e])
        else:
            sc.big_boxes.append([t, e, [bmin, bmax]]) # using a dictionary or class might be faster !!!
            # !!! instead of passing bounds could figure out the min and max in the tree every time
            #       we divide. So divide the left and right for example then get the new bounds for
            #       each side and so on...

    sizes = [b[1].shape[0] for b in sc.big_boxes]
    if len(sizes) > 0:    
        check = max(sizes)
    
    limit = 6
    count = 1

    done = False
    while len(sc.big_boxes) > 0:
        b2(sc, cloth, count)

        sizes2 = [b[1].shape[0] for b in sc.big_boxes]
        if len(sizes2) > 0:
            if check / max(sizes2) < 1.5:
                done = True
        
        if count == limit:
            done = True
                    
        if done:
            for b in sc.big_boxes:
                sc.small_boxes.append(b)
            break
        count += 1
        
    for en, b in enumerate(sc.small_boxes):
        trs = np.array(b[0], dtype=np.int32)
        ed = np.array(b[1], dtype=np.int32) # can't figure out why this becomes an object array sometimes...
        
        if ed.shape[0] == 0:
            continue

        rse = np.tile(ed, trs.shape[0])
        rse.shape = (trs.shape[0], ed.shape[0])
        rst = np.repeat(trs, ed.shape[0])
        rst.shape = (trs.shape[0], ed.shape[0])
        
        re = rse#[~ab] # repeated edges with link faces removed
        rt = rst#[~ab] # repeated triangles to match above edges
                
        in_x = txmax[rt] >= sc.exmin[re]
        rt, re = rt[in_x], re[in_x]

        in_x2 = txmin[rt] <= sc.exmax[re]
        rt, re = rt[in_x2], re[in_x2]

        in_y = tymax[rt] >= sc.eymin[re]
        rt, re = rt[in_y], re[in_y]

        in_y2 = tymin[rt] <= sc.eymax[re]
        rt, re = rt[in_y2], re[in_y2]

        in_z = tzmin[rt] <= sc.ezmax[re]
        rt, re = rt[in_z], re[in_z]
        
        in_z2 = tzmax[rt] >= sc.ezmin[re]
        rt, re = rt[in_z2], re[in_z2]
        
        if rt.shape[0] > 0:
            
            sc.ees += re.tolist()
            sc.trs += rt.tolist()
        

def ray_check(sc, ed, trs, cloth):
    
    # ed is a list object so we convert it for indexing the points
    # trs indexes the tris
    #edidx = np.array(ed, dtype=np.int32)

    #cloth.oc_tris_six[:, :3] = (cloth.last_co - cloth.inner_norms)[cloth.oc_total_tridex]
    #cloth.oc_tris_six[:, :3] -= cloth.inner_norms[cloth.oc_total_tridex]

    edidx = np.array(ed, dtype=np.int32)
    trs = np.array(trs, dtype=np.int32)
    
    
    # e is the start co and current co of the cloth paird in Nx2x3    
    e = cloth.traveling_edge_co[edidx]

    
    start_co = e[:, 0]

    #start_co -= cloth.inner_norms[cloth.boxboo][edidx]

    co = e[:, 1]
    
    #sc.tris6[:, :3] = (cloth.last_co + cloth.inner_norms)[cloth.oc_total_tridex][cloth.boxboo]        
    cloth.oc_tris_six[:, :3] = cloth.last_co[cloth.oc_total_tridex]
    t = cloth.oc_tris_six[cloth.boxboo][trs]
    #t = sc.tris6[trs]
    
    ori = t[:, 3]
    t1 = t[:, 4] - ori
    t2 = t[:, 5] - ori
    
    norms = np.cross(t1, t2)
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    
    vecs = co - ori
    dots = np.einsum('ij,ij->i', vecs, un)
    
    switch = dots <= 0 # why does this work????
    #switch = dots <= cloth.ob.MC_p# why does this work????
    
    check, weights = inside_triangles(t[:, :3][switch], co[switch])
    
    start_check, start_weights = inside_triangles(t[:, :3][switch], start_co[switch], margin= 0.0)

    #pcols = edidx[switch][check]
    #cloth.static = True

    if cloth.static:    
        travel = un[switch][check] * -dots[switch][check][:, None]
        weight_plot = t[:, 3:][switch][check] * start_weights[check][:, :, None]
        #weight_plot = t[:, 3:][switch][check] * weights[check][:, :, None]
        loc = np.sum(weight_plot, axis=1)
        pcols = edidx[switch][check]

    else:
        travel = un[switch] * -dots[switch][:, None]
        weight_plot = t[:, 3:][switch] * start_weights[:, :, None]
        loc = np.sum(weight_plot, axis=1)
        pcols = edidx[switch]

    cco = sc.fco[pcols]
    pl_move = loc - cco

    # static friction
    ob_settings = not cloth.ob.MC_props.override_settings
    if ob_settings:
        #trsidx = np.array(trs, dtype=np.int32)
        if cloth.static:    
            tcols = trs[switch][check]
        else:    
            tcols = trs[switch]
    
        fr = cloth.total_friction[cloth.boxboo][tcols] # put in a static friction method !!! when the force is greater than a length it pulls otherwise it sticks.

        move = (travel * (1 - fr)) + (pl_move * fr)
        
        st = (cloth.move_dist[pcols] < cloth.total_static[cloth.boxboo][tcols])
        move[st] = pl_move[st]

        rev = revert_rotation(cloth.ob, move)

        if False:
            lens = np.sqrt(np.einsum('ij,ij->i', rev, rev))
            uni, inv, counts = np.unique(pcols, return_inverse=True, return_counts=True)
            stretch_array = np.zeros(uni.shape[0], dtype = np.float32)
            np.add.at(stretch_array, inv, lens)
            weights = lens / stretch_array[inv]
            rev *= (weights[:, None])# * .777)    

            ntn = np.nan_to_num(rev)
            np.add.at(cloth.co, pcols, ntn)
            
        cloth.co[pcols] += rev    

        return

    fr = cloth.object_friction # put in a static friction method !!! when the force is greater than a length it pulls otherwise it sticks.
    move = (travel * (1 - fr)) + (pl_move * fr)
            
    st = (cloth.move_dist[pcols] < cloth.static_threshold)
    move[st] = pl_move[st]

    rev = revert_rotation(cloth.ob, move)
    cloth.co[pcols] += rev
    

class ObjectCollide():
    name = "oc"
    
    def __init__(self, cloth):
        

        sco = apply_transforms(cloth.ob, cloth.select_start)
        fco = apply_transforms(cloth.ob, cloth.co)
        self.fco = fco
        
        #cloth.oc_co[:cloth.v_count] = sco
        #cloth.oc_co[cloth.v_count:] = fco

        cloth.traveling_edge_co[:, 0] = sco
        cloth.traveling_edge_co[:, 1] = fco
        # if static:
            # should make tris six into tries three here using just cloth.total_co
        
        cloth.oc_tris_six[:, :3] = (cloth.last_co + cloth.inner_norms)[cloth.oc_total_tridex]
        #cloth.oc_tris_six[:, :3] = cloth.last_co[cloth.oc_total_tridex]
        cloth.oc_tris_six[:, 3:] = cloth.total_co[cloth.oc_total_tridex]
        
        #print(cloth.oc_tris_six.shape, "shape")
        # -----------------------

        #self.box_max = cloth.ob.MC_props.sc_box_max        
        self.box_max = 500#cloth.ob.MC_props.sc_box_max        
        
        self.M = cloth.OM
        #self.force = cloth.ob.MC_props.self_collide_force
        self.tris = cloth.oc_tris_six
        #self.edges = cloth.oc_co[cloth.sc_edges]
        #print(self.edges[0], "original")
        #print(cloth.traveling_edge_co[0], "traveling")
        #self.edges = cloth.traveling_edge_co
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size
        self.trs = []
        self.ees = []


def detect_collisions(cloth):
    
    sc = ObjectCollide(cloth)

    object_collisions_7(sc, sc.M, cloth)
    if sc.finished:
        return
    
    ray_check(sc, sc.ees, sc.trs, cloth)
    

def register():
    pass


def unregister():
    pass


if __name__ == "__main__":
    register()
