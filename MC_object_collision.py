
try:
    import bpy
    import bmesh
    import numpy as np
    import time

except ImportError:
    pass


big_t = 0.0
def rt_(num=None):
    global big_t
    t = time.time()
    if num is not None:    
        print(t - big_t, "timer", num)
    big_t = t


def revert_rotation(ob, co):
    """When reverting vectors such as normals we only need
    to rotate"""
    m = np.array(ob.matrix_world)
    mat = m[:3, :3] / np.array(ob.scale, dtype=np.float32) # rotates backwards without T
    return (co @ mat) / np.array(ob.scale, dtype=np.float32)


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
    #print('running b2 self_collisions', count)
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
        

def generate_bounds(minc, maxc, margin):
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
    
    co = cloth.ob_co
    
    if bounds is None:
        b_min = np.min(co, axis=0)
        b_max = np.max(co, axis=0)
    else:
        b_min, b_max = bounds[0], bounds[1]
        
    # bounds_8 is for use on the next iteration.
    mid, bounds_8 = generate_bounds(b_min, b_max, margin)
    
    mid_ = mid
    _mid = mid

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
        idx = cloth.oc_indexer
    if eidx is None:    
        eidx = cloth.verts_idx

    idx = np.array(idx, dtype=np.int32)
    eidx = np.array(eidx, dtype=np.int32)

    # -------------------------------
    B = xmin[idx] <= x_
    il = idx[B]

    B = xmax[idx] >= _x
    ir = idx[B]
    
    # edges
    eB = exmin[eidx] <= x_
    eil = eidx[eB]

    eB = exmax[eidx] >= _x
    eir = eidx[eB]

    # ------------------------------
    B = ymax[il] >= _y
    ilf = il[B]

    B = ymin[il] <= y_
    ilb = il[B]

    B = ymax[ir] >= _y
    irf = ir[B]

    B = ymin[ir] <= y_
    irb = ir[B]
    
    # edges
    eB = eymax[eil] >= _y
    eilf = eil[eB]

    eB = eymin[eil] <= y_
    eilb = eil[eB]

    eB = eymax[eir] >= _y
    eirf = eir[eB]

    eB = eymin[eir] <= y_
    eirb = eir[eB]

    # ------------------------------
    B = zmax[ilf] >= _z
    ilfu = ilf[B]
    B = zmin[ilf] <= z_
    ilfd = ilf[B]

    B = zmax[ilb] >= _z
    ilbu = ilb[B]
    B = zmin[ilb] <= z_
    ilbd = ilb[B]

    B = zmax[irf] >= _z
    irfu = irf[B]
    B = zmin[irf] <= z_
    irfd = irf[B]

    B = zmax[irb] >= _z
    irbu = irb[B]
    B = zmin[irb] <= z_
    irbd = irb[B]

    # edges
    eB = ezmax[eilf] >= _z
    eilfu = eilf[eB]
    eB = ezmin[eilf] <= z_
    eilfd = eilf[eB]

    eB = ezmax[eilb] >= _z
    eilbu = eilb[eB]
    eB = ezmin[eilb] <= z_
    eilbd = eilb[eB]

    eB = ezmax[eirf] >= _z
    eirfu = eirf[eB]
    eB = ezmin[eirf] <= z_
    eirfd = eirf[eB]

    eB = ezmax[eirb] >= _z
    eirbu = eirb[eB]
    eB = ezmin[eirb] <= z_
    eirbd = eirb[eB]    

    boxes = [ilbd, irbd, ilfd, irfd, ilbu, irbu, ilfu, irfu]
    eboxes = [eilbd, eirbd, eilfd, eirfd, eilbu, eirbu, eilfu, eirfu]
    
    bbool = np.array([i.shape[0] > 0 for i in boxes])
    ebool = np.array([i.shape[0] > 0 for i in eboxes])
    both = bbool & ebool
    
    full = np.array(boxes, dtype=np.object)[both]
    efull = np.array(eboxes, dtype=np.object)[both]

    return full, efull, [bounds_8[0][both], bounds_8[1][both]]
    

def recursive_boxes(sc, cloth=None):

    tx = sc.tris[:, :, 0]
    ty = sc.tris[:, :, 1]
    tz = sc.tris[:, :, 2]
    
    txmax = np.max(tx, axis=1)
    txmin = np.min(tx, axis=1)

    tymax = np.max(ty, axis=1)
    tymin = np.min(ty, axis=1)

    tzmax = np.max(tz, axis=1)
    tzmin = np.min(tz, axis=1)

    sc.txmax = txmax
    sc.txmin = txmin

    sc.tymax = tymax
    sc.tymin = tymin

    sc.tzmax = tzmax
    sc.tzmin = tzmin

    # edge bounds:
    ex = sc.edges[:, :, 0]
    ey = sc.edges[:, :, 1]
    ez = sc.edges[:, :, 2]

    sc.exmin = np.min(ex, axis=1)
    sc.eymin = np.min(ey, axis=1)
    sc.ezmin = np.min(ez, axis=1)
    
    sc.exmax = np.max(ex, axis=1)
    sc.eymax = np.max(ey, axis=1)
    sc.ezmax = np.max(ez, axis=1)
    
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
    
    limit = 10
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
        
        tris = sc.tris[trs]
        eds = sc.edges[ed]
        
        # detect link faces and broadcast
        #nlf_0 = cloth.sc_edges[ed][:, 0] == cloth.tridex[trs][:, :, None]
        #ab = np.any(nlf_0, axis=1)
        
        rse = np.tile(ed, trs.shape[0])
        rse.shape = (trs.shape[0], ed.shape[0])
        rst = np.repeat(trs, ed.shape[0])
        rst.shape = (trs.shape[0], ed.shape[0])
        
        re = rse#[~ab] # repeated edges with link faces removed
        rt = rst#[~ab] # repeated triangles to match above edges
        
        if True:        
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


def basic_collide(sc, ed, trs, cloth):
    """Simple collision based on bounding boxes
    with no ray check or friction"""
    ed = np.array(ed, dtype=np.int32)
    trs = np.array(trs, dtype=np.int32)

    e = sc.edges[ed]
    t = sc.tris[trs]
    
    s_norms = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    norms = np.cross(t[:, 4] - t[:, 3], t[:, 5] - t[:, 3])
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    
    s_compare = np.einsum('ij,ij->i', e[:, 0] - t[:, 0], s_norms)
    compare = np.einsum('ij,ij->i', e[:, 1] - t[:, 3], un)
    
    switch = (s_compare * compare) <= 0.0
    
    cloth.co[ed[switch]] -= (un * compare[:, None])[switch]


def ray_two(sc, ed, trs, cloth):
    """Supports friction, static friction
    and inside traingle check. Works better
    when using low-poly objects because
    outer margin with triangles bends
    normals weird directions."""
    # -----------------------------------
    ed = np.array(ed, dtype=np.int32)
    trs = np.array(trs, dtype=np.int32)

    e = sc.edges[ed]
    t = sc.tris[trs]

    norms = np.cross(t[:, 4] - t[:, 3], t[:, 5] - t[:, 3])
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    
    s_compare = np.einsum('ij,ij->i', e[:, 0] - t[:, 0], norms)
    compare = np.einsum('ij,ij->i', e[:, 1] - t[:, 3], un)
    
    switch = (s_compare * compare) <= 0.0
        
    st = sc.fr_tris[trs]
    start_check, start_weights = inside_triangles(st[switch], e[:, 0][switch], margin= 0.0)

    c_check, c_weights = inside_triangles(t[:, 3:][switch], e[:, 1][switch], margin=0.0)
    switch[switch] = c_check

    so_far = ed[switch]
    fr_idx = trs[switch]
    weight_plot = t[:, 3:][switch] * start_weights[:, :, None][c_check]

    cl = e[:, 1][switch]
    tf_so_far = cloth.total_friction[fr_idx]
    
    fr_move = np.sum(weight_plot, axis=1) - cl
    fr = fr_move * tf_so_far

    no_fr = (-un[switch] * compare[switch][:, None]) * (1 - tf_so_far)
    mixed = fr + no_fr
    
    stat = (cloth.move_dist[so_far] < cloth.total_static[fr_idx])
    mixed[stat] = fr_move[stat]
    
    rev = revert_rotation(cloth.ob, mixed)
    cloth.co[so_far] += rev
    

class ObjectCollide():
    name = "ob"
    
    def __init__(self, cloth):

        # -----------------------
        ob = cloth.ob
        
        tris_six = cloth.oc_tris_six # empty float array shape N x 6 x 3 for start end triangles of collide objects combined
        tridex = cloth.oc_total_tridex # indexer of combined objects with offset based on vert counts of each object
        
        # shift is outer margin
        # ishift is outer margin minus inner margin
        shift = cloth.ob_v_norms * cloth.outer_margins        
        ishift = cloth.ob_v_norms * cloth.inner_margins
        
        self.shift = shift
        self.ishift = ishift
                
        tris_six[:, :3] = (cloth.last_co - ishift)[tridex]
        tris_six[:, 3:] = (cloth.total_co + shift)[tridex]
        
#        tco = cloth.total_co + shift
#        test_ob = bpy.data.objects['test']
#        test_ob.data.vertices.foreach_set('co', tco.ravel())
#        test_ob.data.update()

#        tco2 = cloth.last_co - ishift
#        test_ob_2 = bpy.data.objects['test2']
#        test_ob_2.data.vertices.foreach_set('co', tco2.ravel())
#        test_ob_2.data.update()

#        tco3 = cloth.total_co
#        test_ob_3 = bpy.data.objects['t3']
#        test_ob_3.data.vertices.foreach_set('co', tco3.ravel())
#        test_ob_3.data.update()
        
        self.fr_tris = (cloth.last_co + shift)[tridex]

        self.box_max = 150 #cloth.ob.MC_props.sc_box_max
        
        self.tris = tris_six
        self.edges = cloth.ob_co[cloth.sc_edges]

        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        self.trs = []
        self.ees = []
        

def detect_collisions(cloth):
    
    sc = ObjectCollide(cloth)
    recursive_boxes(sc, cloth)
    
    if bpy.context.scene.MC_props.basic_collide:
        basic_collide(sc, sc.ees, sc.trs, cloth)

    else:
        ray_two(sc, sc.ees, sc.trs, cloth)
