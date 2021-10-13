
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


def inside_triangles(tris, points, margin=0.0):#, cross_vecs): # could plug these in to save time...
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
    #margin = -cloth.ob.MC_props.sc_expand_triangles
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)
    
    return check, weights


# universal ---------------
def eliminate_pairs(ar1, ar2):
    """Remove pairs from ar1 that are in ar2"""
    x = np.array(np.random.rand(ar1.shape[1]), dtype=np.float32)
    y = ar1 @ x
    z = ar2 @ x
    booly = np.isin(y, z, invert=True)
    return ar1[booly], booly


def eliminate_duplicate_pairs(ar, sort=True):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns an Nx2 array."""
    # no idea how this works (probably sorcery) but it's really fast
    a = ar
    if sort:
        a = np.sort(ar, axis=1)
    x = np.array(np.random.rand(a.shape[1]), dtype=np.float32)
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index


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
    
    co = cloth.sc_co
    
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
        idx = cloth.sc_indexer
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
    

def self_collisions_7(sc, margin=0.1, cloth=None):

    tx = sc.tris[:, :, 0]
    ty = sc.tris[:, :, 1]
    tz = sc.tris[:, :, 2]

    margin = cloth.ob.MC_props.min_max_margin
    margin = 0.0
    
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
    ex = sc.edges[:, :, 0]
    ey = sc.edges[:, :, 1]
    ez = sc.edges[:, :, 2]

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
        
        #tris = sc.tris[trs]
        #eds = sc.edges[ed]
        
        # detect link faces and broadcast
        nlf_0 = cloth.sc_edges[ed][:, 0] == cloth.tridex[trs][:, :, None]
        ab = np.any(nlf_0, axis=1)
        
        rse = np.tile(ed, trs.shape[0])
        rse.shape = (trs.shape[0], ed.shape[0])
        rst = np.repeat(trs, ed.shape[0])
        rst.shape = (trs.shape[0], ed.shape[0])
        
        re = rse[~ab] # repeated edges with link faces removed
        rt = rst[~ab] # repeated triangles to match above edges
        
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
    

def compare_sew_v_norms(cloth):
    """Check sew edges against the vertex norms.
    If they are perpindicular we probably want
    to sew tight. Otherwise they are part of a fold...
    probably..."""


def ray_check_oc(sc, ed, trs, cloth, sort_only=False):
    
    """Need to fix selected points by removing them
    from the weights. (For working in edit mode)
    Need to join ob collide and self collide weights
    to improve stability."""
    
    eidx = np.array(ed, dtype=np.int32)
    tidx = np.array(trs, dtype=np.int32)
    s1 = tidx.shape[0]
    if cloth.has_butt_edges:
        #print('running')
        eliminator = np.empty((eidx.shape[0], 2), dtype=np.int32)
        eliminator[:, 0] = eidx
        eliminator[:, 1] = tidx
        new, boo = eliminate_pairs(eliminator, cloth.butt_tri_pairs)
        #new, boo = eliminate_pairs(eliminator, eliminator[:4])
        eidx = new[:, 0]
        tidx = new[:, 1]
        #for i in eliminator:
            #print(i)
        #print(eliminator)
        #print(cloth.butt_tri_pairs)
        if False:    
            if s1 - tidx.shape[0] != 0:
                print()
                print()
                print('different!!!')
            #print(s1, tidx.shape[0])  
        #print()
        #print(tidx.shape[0])
        #print(s1)
    if sort_only:
        cloth.oct_tris = tidx
        cloth.oct_points = eidx
        return
    
    # undo the offset:
    sc.tris[:, :3] = cloth.select_start[cloth.tridex]# - shift)[cloth.tridex]
    sc.tris[:, 3:] = cloth.co[cloth.tridex]# + shift)[cloth.tridex]
         
    if cloth.ob.MC_props.sew_tight:
        #pass
    #if cloth.ob.MC_props.butt_sew_force > 0.0:
        #print(cloth.sew_verts[ed])
        cloth.sew_edges
        ignore = cloth.sew_verts[eidx]
        eidx = eidx[~ignore]
        tidx = tidx[~ignore]
        #when a vert sews an overlap like a fold the sew edge is more or less paralell
        #to the vert normals
        #when its end to end its perp to the normals.
        #so a dot of less than .707 is less than 45. 
        #We probably want to err on the side of caution so like a dot of 0.25 or less
        #should mean we can sew all the way and ignore self collide forces on that vert
        
        
    e = sc.edges[eidx]
    t = sc.tris[tidx]        

    start_co = e[:, 0]
    co = e[:, 1]

    M = cloth.ob.MC_props.self_collide_margin

    start_ori = t[:, 0]
    st1 = t[:, 1] - start_ori
    st2 = t[:, 2] - start_ori
    start_norms = np.cross(st1, st2)
    u_start_norms = start_norms / np.sqrt(np.einsum('ij,ij->i', start_norms, start_norms))[:, None]
    start_vecs = start_co - start_ori
    start_dots = np.einsum('ij,ij->i', start_vecs, u_start_norms)
    
    # normals from cloth.co (not from select_start)
    ori = t[:, 3]
    t1 = t[:, 4] - ori
    t2 = t[:, 5] - ori
    norms = np.cross(t1, t2)
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]

    vecs = co - ori
    dots = np.einsum('ij,ij->i', vecs, un)
    
    # new ================================
    new = False
    #new = True
    if new:    
        #switch = np.sign(dots * start_dots)
        #in_margin = (abs_dots <= M) | (switch <= 0.0)
        #start_check, start_weights_1 = inside_triangles(t[:, :3][in_margin], start_co[in_margin], margin= -cloth.ob.MC_props.sc_expand_triangles)        
        switch = np.sign(dots * start_dots)
        abs_dots = np.abs(dots)
        in_margin = (abs_dots <= M)
        direction = M - abs_dots
        direction *= switch
    
        #return
    else:
        switch = np.sign(dots * start_dots)
        direction = np.sign(dots)
        abs_dots = np.abs(dots)
        
        # !!! if a point has switched sides, direction has to be reversed !!!    
        direction *= switch
        in_margin = (abs_dots <= M) | (switch == -1)
        #in_margin = (abs_dots <= M)# | (switch == -1)

    
    check_1, weights = inside_triangles(t[:, 3:][in_margin], co[in_margin], margin= -0.1)
    start_check, start_weights_1 = inside_triangles(t[:, :3][in_margin], start_co[in_margin], margin= -cloth.ob.MC_props.sc_expand_triangles)

    check = check_1 | start_check
    #check = check_1# | start_check

    #check = check_1 | start_check
    #check[:] = True
    start_weights = start_weights_1[check]

    weight_plot = t[:, 3:][in_margin][check] * start_weights[:, :, None]

    loc = np.sum(weight_plot, axis=1) + ((un[in_margin][check] * M) * direction[in_margin][check][:, None])
    
    mean = True
    if mean:
        co_idx = eidx[in_margin][check]
        uni, counts = np.unique(co_idx, return_counts=True)
        cloth.sc_meaner[:] = 0.0
        np.add.at(cloth.sc_meaner, co_idx, loc)
        mean_move = cloth.sc_meaner[uni] / counts[:, None]
        dif = mean_move - cloth.co[uni]
        scf = cloth.ob.MC_props.self_collide_force

        cloth.co[uni] += np.nan_to_num(dif * scf)
        
        fr = cloth.ob.MC_props.sc_friction
        if fr > 0:
            cloth.velocity[uni] *= (1 - fr)
        return
    
    T = time.time()    
    move = False
    move = True
    if move:    
        co_idx = eidx[in_margin][check]
        #print(np.sum(co_idx == 722))
        dif = (loc - cloth.co[co_idx])
        uni, inv, counts = np.unique(co_idx, return_inverse=True, return_counts=True)
        ntn = np.nan_to_num(dif / counts[inv][:, None])
        #ntn = np.nan_to_num(dif)# / counts[inv][:, None])
        scf = cloth.ob.MC_props.self_collide_force
        np.add.at(cloth.co, co_idx, ntn * scf)
        #np.subtract.at(cloth.velocity, co_idx, ntn * scf)
        #tp = cloth.tridex[tidx[in_margin][check]].ravel()
        #tri_travel = np.repeat(-ntn, 3, axis=0)
        #np.add.at(cloth.co, tp, tri_travel * 0.1)
        
        fr = cloth.ob.MC_props.sc_friction
        if fr > 0:
            cloth.velocity[co_idx] *= (1 - fr)        
        
        #cloth.velocity[co_idx] *= 0
        #np.subtract.at(cloth.velocity, co_idx, ntn)
        print(time.time() - T)
        return
    co_idx = eidx[in_margin][check]

    travel = -(u_start_norms[in_margin][check] * dots[in_margin][check][:, None]) + ((u_start_norms[in_margin][check] * M) * direction[in_margin][check][:, None])
    #start_check, start_weights = inside_triangles(t[:, :3][in_margin][check], co[in_margin][check], margin= -0.1)
    #move = cloth.co[co_idx] - start_co_loc
    
    #now in theory I can use the weights from start tris
    combined_weights = True # weights with points and tris
    if combined_weights:

        tp = cloth.tridex[tidx[in_margin][check]].ravel()
        joined = np.append(co_idx, tp)
        
        uni, inv, counts = np.unique(joined, return_inverse=True, return_counts=True)
        
        lens = np.sqrt(np.einsum('ij,ij->i', travel, travel))
        tri_lens = np.repeat(lens, 3)
        joined_lens = np.append(lens, tri_lens)
        
        stretch_array = np.zeros(uni.shape[0], dtype = np.float32)
        np.add.at(stretch_array, inv, joined_lens)
        weights = joined_lens / stretch_array[inv]
        tri_travel = np.repeat(-travel, 3, axis=0)
        joined_travel = np.append(travel, tri_travel, axis=0)
        
        scf = cloth.ob.MC_props.self_collide_force
        
        joined_travel *= (weights[:, None] * scf)

        ntn = np.nan_to_num(joined_travel)

        np.add.at(cloth.co, joined, ntn)
        #cloth.velocity[joined] *= 0
        #np.add.at(cloth.velocity, joined, -ntn)# * -0.5)

        fr = cloth.ob.MC_props.sc_friction
        if fr > 0:
            cloth.velocity[co_idx] *= (1 - fr)
    

class SelfCollide():
    name = "sc"
    
    def __init__(self, cloth):

        # -----------------------
        ob = cloth.ob
        tris_six = cloth.tris_six

        tridex = cloth.tridex
        
        cloth.sc_co[:cloth.v_count] = cloth.select_start
        cloth.sc_co[cloth.v_count:] = cloth.co

        offset_tris = True
        #offset_tris = False
        if offset_tris:        
            M = cloth.ob.MC_props.self_collide_margin# * .5
            #shift = cloth.v_norms * M# * 0
            shift = M# * 0
            self.shift = shift
            tris_six[:, :3] = (cloth.select_start - shift)[cloth.tridex]
            tris_six[:, 3:] = (cloth.co + shift)[cloth.tridex]
        else:
            tris_six[:, :3] = cloth.select_start[cloth.tridex]
            tris_six[:, 3:] = cloth.co[cloth.tridex]
        
        # -----------------------

        self.box_max = cloth.ob.MC_props.sc_box_max
        #self.box_max = 190#cloth.ob.MC_props.sc_box_max

        self.M = cloth.ob.MC_props.self_collide_margin
        
        self.tris = tris_six
        self.edges = cloth.sc_co[cloth.sc_edges]

        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        self.trs = []
        self.ees = []
        

def detect_collisions(cloth, sort_only=False):
    
    sc = SelfCollide(cloth)
    self_collisions_7(sc, sc.M, cloth)
    ray_check_oc(sc, sc.ees, sc.trs, cloth, sort_only=sort_only)
