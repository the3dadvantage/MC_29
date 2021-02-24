import bpy
import numpy as np
import bmesh
import time


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
    
    sh = cloth.pierce_co.shape[0]
    shh = cloth.pierce_co.shape
    
    if bounds is None:
        cloth.pierce_co.shape = (shh[0] * 2, 3)
        b_min = np.min(cloth.pierce_co, axis=0)
        b_max = np.max(cloth.pierce_co, axis=0)
        cloth.pierce_co.shape = shh
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
        eidx = cloth.peidx

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
    

def self_collisions_7(sc, cloth=None):

    tx = sc.tris[:, :, 0]
    ty = sc.tris[:, :, 1]
    tz = sc.tris[:, :, 2]

    margin = 0.0
    
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
        eds = cloth.eidx[ed]
        
        # detect link faces and broadcast
        ttt = cloth.tridex[trs][:, :, None]
        nlf_0 = cloth.eidx[ed][:, 0] == ttt
        nlf_1 = cloth.eidx[ed][:, 1] == ttt
        ab1 = np.any(nlf_0, axis=1)
        ab2 = np.any(nlf_1, axis=1)
        ab = ab1 | ab2
        
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


def side_sort(points, tridex, cloth):
    # get the unique pairs
    unip, p_idx = eliminate_duplicate_pairs(points)
    
    tri_p = tridex[p_idx]    
    
    
    # boundary check
    if False:
        cloth.boundary_verts = np.array([[v.index for v in t.verts if v.is_boundary] for t in cloth.triobm.faces], dtype=np.object)
        cloth.boundary_bool = np.array([[v.is_boundary for v in t.verts] for t in cloth.triobm.faces], dtype=np.bool)
        cloth.boundary_tris = np.array([np.any(b) for b in cloth.boundary_bool], dtype=np.bool)
        
        
        print(tri_p, "this is tri-p")
        ct = tri_p[cloth.boundary_tris[tri_p]]
        print(cloth.boundary_verts[ct], "this is the verts")
#        clo
#        
#        
#        now i need a way to move towards the outside edge...
#        I suppose I could just move past the boundary vert...
#        Maybe by the sc margin?
#        so... 
        
        for i, t in enumerate(cloth.boundary_bool):
            if np.any(t):
                bpy.context.object.data.vertices[cloth.tridex[i][0]].select = True
                
        cloth.ob.data.update()
        #cloth.bt_edges = [[[e.verts[0].index, e.verts[1].index] for e in t.edges] for t in cloth.triobm.faces]
        
        #print(cloth.boundary_bool)
        
        
    #print(unip.shape)
    #print(tri_p)
        
        is_boundary = cloth.boundary_tris[tri_p]
        if np.any(is_boundary):
            print("found one")
        #print(tri_p[is_boundary], "b tris")
        #print(unip[is_boundary], "index here?")
        
        
        
        #cloth.boundary_tris[tri_p]
#        problem: a boundary tri
#        could just have a vert
#        and not an edge.
#        When this happens
#        could move towards the vert
#        also could move towards the eges
#        a boundary edge connected
#        to the vert.
#        Need to change boundary 
#        tris to vert.is_boundary
#        find the other vert that
#        is boundary.
#        Should always be at least
#        one (I think two) edge that is boundary
#        connected to the vert
        
    
    
#    So the idea is where a tri is on the boundary
#    if an edge pierces it, we need to find
#    That tri's' edge verts. will also need
#    the point on the plane so we can measure.
#    could spit that out and do it in the other
#    function
    
    # get the edges and tris where boundary happens
    

    uni, idx, inv, counts = np.unique(unip, return_inverse=True, return_counts=True, return_index=True)
        
    cinv = counts[inv]
    sh = cinv.shape[0]
    cinv.shape = (sh//2, 2)
    


    # multi edges
    culprits = counts > 1

    verts = uni[culprits]

    e_tiled_tris = np.repeat(tri_p, 2)[idx]
    tris = e_tiled_tris[culprits]


    
    
    return verts, tris


def ray_check_oc(sc, ed, trs, cloth):
    
    """Need to fix selected points by removing them
    from the weights. (For working in edit mode)
    Need to join ob collide and self collide weights
    to improve stability."""
    
    eidx = np.array(ed, dtype=np.int32)
    tidx = np.array(trs, dtype=np.int32)

    e = sc.edges[eidx]
    t = sc.tris[tidx]

    start_co = e[:, 0]
    co = e[:, 1]

    ori = t[:, 0]
    t1 = t[:, 1] - ori
    t2 = t[:, 2] - ori
    norms = np.cross(t1, t2)
    #u_norms = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    start_vecs = start_co - ori
    vecs = co - ori
    start_dots = np.einsum('ij,ij->i', start_vecs, norms)
    #dots = np.einsum('ij,ij->i', vecs, u_norms)
    
    #es = cloth.eidx.shape
    #switch = (start_dots * dots) < -0.0001
    #print(cloth.eidx[eidx][switch])
    #print(tidx[switch])
    
    #if not np.any(switch):
        #return
    #print(eidx[switch])
    #print(es)
    #print(cloth.eidx[es:][eidx][switch])
    e_vec = co - start_co
    or_vec = start_vecs
    e_n_dot = start_dots
    e_dot = np.einsum('ij,ij->i', e_vec, norms)
    scale = e_n_dot / e_dot
    
    through = (scale < 0) & (scale > -1)

    #if not np.any(through):
        #return
    #print(through[switch])
    
    plot = (or_vec - e_vec * scale[:, None]) + ori
    plots = plot[through]

    rt = t[through]
    check, weights = inside_triangles(rt[:, :3], plots, margin= 0.0)
    
    if np.any(check):
            


        points = cloth.eidx[ed][through][check]
        #print(points)
        
        #cloth.co[cloth.eidx[ed][through][check]] += np.array([0.0, 0.0, 0.001])
        #print(eidx[through][check]])
        verts, tris = side_sort(points, tidx[through][check], cloth)

        
        if tris.shape[0] > 0:            
            nor = cloth.tri_normals[tris]
            ori = cloth.co[cloth.tridex[tris]][:, 0]
            vec = cloth.co[verts] - ori
            
            dot = np.einsum('ij,ij->i', vec, nor)
            
            move = nor * dot[:, None] * -(1 + cloth.ob.MC_props.self_collide_margin)
                
            
            #testing = True
            testing = False
            if not testing:    
                cloth.co[verts] += (move * 1)
            
        
def slide_point_to_plane(e1, e2, normal, origin, intersect=False):
    
    e_vec = e2 - e1
    or_vec = e1 - origin
    e_dot = np.dot(normal, e_vec)
    e_n_dot = np.dot(normal, or_vec)
    scale = e_n_dot / e_dot  
    if intersect:
        return (or_vec - e_vec * scale) + origin, (scale < 0) & (scale > -1)
    else:    
        return (or_vec - e_vec * scale) + origin

    
    return
    # normals from cloth.co (not from select_start)
    ori = t[:, 3]
    t1 = t[:, 4] - ori
    t2 = t[:, 5] - ori
    norms = np.cross(t1, t2)
    un = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]

    vecs = co - ori
    dots = np.einsum('ij,ij->i', vecs, un)

    switch = np.sign(dots * start_dots)
    direction = np.sign(dots)
    abs_dots = np.abs(dots)
    
    # !!! if a point has switched sides, direction has to be reversed !!!    
    direction *= switch
    in_margin = (abs_dots <= M) | (switch == -1)
    
    check_1, weights = inside_triangles(t[:, 3:][in_margin], co[in_margin], margin= -0.1)
    start_check, start_weights_1 = inside_triangles(t[:, :3][in_margin], start_co[in_margin], margin= 0.0)

    #check = check_1 | start_check
    check = check_1 | start_check
    #check[:] = True
    start_weights = start_weights_1[check]

    weight_plot = t[:, 3:][in_margin][check] * start_weights[:, :, None]

    loc = np.sum(weight_plot, axis=1) + ((un[in_margin][check] * M) * direction[in_margin][check][:, None])
    
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

        fr = cloth.ob.MC_props.sc_friction
        if fr > 0:
            cloth.velocity[co_idx] *= (1 - fr)
    

class PierceCollide():
    name = "sc"
    
    def __init__(self, cloth):

        # -----------------------
        ob = cloth.ob
        tridex = cloth.tridex
        
        # -----------------------

        self.box_max = cloth.ob.MC_props.sc_box_max
        
        self.tris = cloth.co[tridex]
        self.edges = cloth.pierce_co#[cloth.pc_edges]
        
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        self.trs = []
        self.ees = []
        

def detect_collisions(cloth):
    
    sc = PierceCollide(cloth)
    self_collisions_7(sc, cloth)
    ray_check_oc(sc, sc.ees, sc.trs, cloth)


def register():
    pass


def unregister():
    pass


if __name__ == "__main__":
    register()
    
    
"""
could detect if an object is both a collider and cloth
and switch to using self collision code in this case.

!!! Use the duplicate pairs code for stability in sc


!!! virtual edge faces for self collisions. Think cloth thickness.

!!! The inside trianlges margin helped with collision fails
        maybe tri this on the quarter zip.

Maybe have a real world size value

do a closest point on mesh check for every vert for the wrap force
    See if that runs at a decent speed.
    
For the p1 sims... Could create a much simplified mesh
    simulate that around the avatar then pick up on the
    high res sim once the low res is in place.

prolly should go back to bounds check for object collide
    seems like I'm getting slowdons like the sparse object

need to debug sparse objects.
    
If all the forces are separated I could multi-thread
cloth.force_stretch
cloth.force_bend
cloth.force_sew
cloth.force_ob_collide
cloth.force_sc_collide
cloth.force_detnagle

could I iterate at the end of collision working on
    the same set of eidx and tidx without repeating broad phase?

Could I put a box max on the octree, especially at the first?
    that should make all eight boxes empty if a collide
    object is way out there. Maybe doesn't make sense
    for self collide? have to think about it.
    
wind ui should have one label and three side by side
Maybe do the same for gravity

Could experiment with shrink/fatten on the folds in the p1 sims
Maybe create a mix of prerwap and flat for the bend targets?
Get the springs for both and run both with bend springs.

Having mutiple source shape keys might allow some intersting things...

for the simplified mesh: think of the hi_res mesh moving
towards it with a force. Could use the simplified mesh
as a sort of guide for the springs and sewing...
    
"""
