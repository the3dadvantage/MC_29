
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
    margin = 0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)
    
    return check, weights


def b2(sc, cloth, count):

    boxes = []
    for oct in sc.big_boxes:
        e = oct[0]
        b = oct[1]
                
        efull, bounds = octree_et(sc, margin=0.0, eidx=e, bounds=b, cloth=cloth)
        
        for i in range(len(efull)):
            e = efull[i]
            bmin = bounds[0][i]
            bmax = bounds[1][i]
            
            if e.shape[0] < sc.box_max:
                sc.small_boxes.append([e])
            else:
                boxes.append([e, [bmin, bmax]])
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
    
    sh = cloth.four_edge_co.shape[0]
    shh = cloth.four_edge_co.shape
    
    if bounds is None:
        cloth.four_edge_co.shape = (shh[0] * 4, 3)
        b_min = np.min(cloth.four_edge_co, axis=0)
        b_max = np.max(cloth.four_edge_co, axis=0)
        cloth.four_edge_co.shape = shh
    else:
        b_min, b_max = bounds[0], bounds[1]
        
    # bounds_8 is for use on the next iteration.
    mid, bounds_8 = generate_bounds(b_min, b_max, margin)
    
    x, y, z = mid[0], mid[1], mid[2]

    # edges
    exmin = sc.exmin
    eymin = sc.eymin
    ezmin = sc.ezmin
    
    exmax = sc.exmax
    eymax = sc.eymax
    ezmax = sc.ezmax

    # l = left, r = right, f = front, b = back, u = up, d = down
    if eidx is None:    
        eidx = cloth.peidx

    eidx = np.array(eidx, dtype=np.int32)

    # -------------------------------
    # edges
    eB = exmin[eidx] <= x
    eil = eidx[eB]

    eB = exmax[eidx] >= x
    eir = eidx[eB]

    # ------------------------------    
    # edges
    eB = eymax[eil] >= y
    eilf = eil[eB]

    eB = eymin[eil] <= y
    eilb = eil[eB]

    eB = eymax[eir] >= y
    eirf = eir[eB]

    eB = eymin[eir] <= y
    eirb = eir[eB]

    # ------------------------------
    # edges
    eB = ezmax[eilf] >= z
    eilfu = eilf[eB]
    eB = ezmin[eilf] <= z
    eilfd = eilf[eB]

    eB = ezmax[eilb] >= z
    eilbu = eilb[eB]
    eB = ezmin[eilb] <= z
    eilbd = eilb[eB]

    eB = ezmax[eirf] >= z
    eirfu = eirf[eB]
    eB = ezmin[eirf] <= z
    eirfd = eirf[eB]

    eB = ezmax[eirb] >= z
    eirbu = eirb[eB]
    eB = ezmin[eirb] <= z
    eirbd = eirb[eB]    

    eboxes = [eilbd, eirbd, eilfd, eirfd, eilbu, eirbu, eilfu, eirfu]
    
    both = np.array([i.shape[0] > 0 for i in eboxes])
    
    efull = np.array(eboxes, dtype=np.object)[both]

    return efull, [bounds_8[0][both], bounds_8[1][both]]
    

def self_collisions_7(sc, cloth=None, flood=False):

    # edge bounds:
    ex = sc.edges[:, :, 0]
    ey = sc.edges[:, :, 1]
    ez = sc.edges[:, :, 2]

    # This margin should be the edge margin property.
    # It should cause the edges to overlap where needed
    #   even for the first octree iteration.
    M = 0.01
    sc.exmin = np.min(ex, axis=1) - M
    sc.eymin = np.min(ey, axis=1) - M
    sc.ezmin = np.min(ez, axis=1) - M
    
    sc.exmax = np.max(ex, axis=1) + M
    sc.eymax = np.max(ey, axis=1) + M
    sc.ezmax = np.max(ez, axis=1) + M
    
    efull, bounds = octree_et(sc, margin=0.0, cloth=cloth)

    for i in range(len(efull)):
        e = efull[i]
        bmin = bounds[0][i]
        bmax = bounds[1][i]
        
        if e.shape[0] < sc.box_max:
            sc.small_boxes.append([e])
        else:
            sc.big_boxes.append([e, [bmin, bmax]]) # using a dictionary or class might be faster !!!
            # !!! instead of passing bounds could figure out the min and max in the tree every time
            #       we divide. So divide the left and right for example then get the new bounds for
            #       each side and so on...
    
    sizes = [b[0].shape[0] for b in sc.big_boxes]
    if len(sizes) > 0:    
        check = max(sizes)
    
    limit = 10
    count = 1

    done = False
    while len(sc.big_boxes) > 0:
        b2(sc, cloth, count)

        sizes2 = [b[0].shape[0] for b in sc.big_boxes]
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
        ed = np.array(b[0], dtype=np.int32) # can't figure out why this becomes an object array sometimes...

        if ed.shape[0] == 0:
            continue
        
        eds = cloth.eidx[ed]
        
        # detect link faces and broadcast
        ttt = cloth.eidx[ed][:, :, None]
        nlf_0 = cloth.eidx[ed][:, 0] == ttt
        nlf_1 = cloth.eidx[ed][:, 1] == ttt
        ab1 = np.any(nlf_0, axis=1)
        ab2 = np.any(nlf_1, axis=1)
        ab = ab1 | ab2
        
        rse = np.tile(ed, ed.shape[0])
        rse.shape = (ed.shape[0], ed.shape[0])
        rst = np.repeat(ed, ed.shape[0])
        rst.shape = (ed.shape[0], ed.shape[0])
        
        re = rse[~ab] # repeated edges with link faces removed
        rt = rst[~ab] # repeated triangles to match above edges
        
        if True:        
            in_x = sc.exmax[rt] >= sc.exmin[re]
            rt, re = rt[in_x], re[in_x]

            in_x2 = sc.exmin[rt] <= sc.exmax[re]
            rt, re = rt[in_x2], re[in_x2]

            in_y = sc.eymax[rt] >= sc.eymin[re]
            rt, re = rt[in_y], re[in_y]

            in_y2 = sc.eymin[rt] <= sc.eymax[re]
            rt, re = rt[in_y2], re[in_y2]

            in_z = sc.ezmin[rt] <= sc.ezmax[re]
            rt, re = rt[in_z], re[in_z]
            
            in_z2 = sc.ezmax[rt] >= sc.ezmin[re]
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
        #cloth.boundary_bool = np.array([[v.is_boundary for v in t.verts] for t in cloth.triobm.faces], dtype=np.bool)
        cloth.boundary_tris = np.array([np.any([v.is_boundary for v in t.verts]) for t in cloth.triobm.faces], dtype=np.bool)
        #cloth.boundary_tris = np.array([np.any(b) for b in cloth.boundary_bool], dtype=np.bool)
        
        
        #print(tri_p, "this is tri-p")
        ct = tri_p[cloth.boundary_tris[tri_p]]
        print(ct, "this is ct")
        
        print(cloth.boundary_verts[ct], "this is the verts")
        
        #when there is a boundary edge it makes
        #sense to just move perp to the edge.
        #when there is only a vert I could
        #1. move towards the vert ()
        #2. get the connected edges and a direction
        #    based on the two edge angles. 
        
        #in a case where there is 3 boundary verts, 
        #ideally we would move towards one 
    #    
        #probably simpler to just do edge edge for everything
        #Could use this module with some mods.
#        
#        
#        now i need a way to move towards the outside edge...
#        I suppose I could just move past the boundary vert...
#        Maybe by the sc margin?
#        so... 
        if False:
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


def ray_check_oc(sc, ed, trs, cloth, flood=False):
    
    """Need to fix selected points by removing them
    from the weights. (For working in edit mode)
    Need to join ob collide and self collide weights
    to improve stability."""
    i = 1
    eidx = np.array(ed, dtype=np.int32)
    tidx = np.array(trs, dtype=np.int32)
    
    if False:
    #if True:
        #print(eidx[i])
        #print(tidx[i])
        cloth.ob.data.edges[eidx[i]].select = True
        print(eidx[i], "eidx i")
        print(tidx[i], "tidx i")
        cloth.ob.data.edges[tidx[i]].select = True
        
    
    e = cloth.four_edge_co[eidx]
    t = cloth.four_edge_co[tidx]
    
    if flood:
        print(eidx)
        print("flood in edge collide")
        return
    
    sh = e.shape[0]
    shh = e.shape
    #print(shh, "should be the shape")
    e.shape = (sh * 4, 3)
    t.shape = (sh * 4, 3)
    
    #evecs = cloth.four_edge_co[1::2] - cloth.four_edge_co[::2]
    eevecs = e[1::2] - e[::2]
    tevecs = t[1::2] - t[::2]
    
    #print(eevecs.shape, "ee vecs shape")
    #print(eevecs.shape, "ee vecs shape")
    cross1 = np.cross(eevecs, tevecs)
    cross2 = np.cross(tevecs, cross1)
    #cross2 = np.cross
    #bpy.data.objects['e'].location = e[i]
    #print(t.shape)
    #print(cross1.shape, "cross 1 shape")
    if False:
        bpy.data.objects['e'].location = t[4] + cross1[2]
        bpy.data.objects['ee'].location = t[4] + cross2[2]

    
  

    # ==============================
    or_vec = e[::2] - t[::2]

    e_dot = np.einsum('ij,ij->i', cross2, eevecs)
    e_n_dot = np.einsum('ij,ij->i', cross2, or_vec)
    scale = (e_n_dot / e_dot)[:, None]  
    

    
    #if intersect:
    this = (or_vec - eevecs * scale) + t[::2]
    if False:    
        bpy.data.objects['eee'].location = this[2]
    
    # now get spit (as in where the spit lands when you stand on the top edge and spit on the bottom)
    this_ori_vec = this - t[::2]
    c1d = np.einsum('ij,ij->i', cross1, cross1)
    norm = cross1 / np.sqrt(c1d)[:, None]
    
    spit_dot = -np.einsum('ij,ij->i', this_ori_vec, norm)
    #spit_l = spit_dot / c1d
    
    spit = this + (norm * spit_dot[:, None])
    #spit[1::2]
    if False:    
        bpy.data.objects['s'].location = spit[2]

    #tevecs[1::2]
    sp_vecs = spit[1::2] - t[::4]
    #scale2 = 
    #print(tevecs[1::2].shape, "meh")
    #print(sp_vecs.shape, "meh")
    
    spit_t_dot = np.einsum('ij,ij->i', tevecs[1::2], sp_vecs)
    te_dot = np.einsum('ij,ij->i', tevecs[1::2], tevecs[1::2])
    #scale2 = np.einsum('ij,ij->i', tevecs[1::2], sp_vecs)
    in_edge_2 = (spit_t_dot >= 0.0) & (spit_t_dot <= te_dot) 
    #in_edge_2 = spit_t_dot <= te_dot 
    #have to get the length of spit.
    #might be better to use the u-norm
    
    # start_spit_l = spit_l[::2]
    # current_spit_l = spit_l[1::2]
    M = 0.04
    absolute_spit = np.abs(spit_dot[1::2])
    in_margin = absolute_spit < M
    crossed = (spit_dot[::2] * spit_dot[1::2]) <= 0.0
    #print(crossed[0], "crossed")
    #print()
    
    hit = crossed | in_margin
    
    
    #eidx[hit]
    
    in_edge = (scale[1::2] < 0) & (scale[1::2] > -1)
    #print(in_edge[0])
    #print(in_edge_2[0])
    #print(eidx)
    #print(tidx)
    #in_edge_2 = (scale2 < 0) & (scale2 > -1)
    #print(in_edge.shape, "in_edge shape")
    #print(in_edge_2.shape, "in_edge_2 shape")
    #print(hit.shape, "hit shape")
    #print(move.shape, "move shape")
    
    
    hit[~in_edge.ravel() | ~in_edge_2] = False
    
    
    #move1 = norm * (spit_dot + (M - spit_dot))[:, None]
    move1 = norm * (M - spit_dot)[:, None]
    move = move1[1::2]

    spit_dir = this - spit
    
    spit_dir_dot = np.einsum('ij,ij->i', norm, spit_dir)
    
    direction = np.sign(spit_dir_dot)
    print(direction[1::2][hit].shape)
    move_re = np.repeat(direction[1::2][hit][:, None] * norm[1::2][hit], 2, axis=0)
    
    #move_re = np.repeat(norm[1::2][hit], 2, axis=0)
    #print(move_re.shape, "shape here too")
    #print(cloth.eidx[eidx[hit]].ravel().shape, "shape here")
    
    e_hit = cloth.eidx[eidx[hit]].ravel()    
    uni, inv, counts = np.unique(e_hit, return_inverse=True, return_counts=True)
    
    div = counts[inv]
    div_move = move_re / div[:, None]

    np.add.at(cloth.co, e_hit, div_move * .02)
    #cloth.co[e_hit] += move_re * .02
    return
    #print(hit[0])
    #print(scale.shape, "scale shape")
    #print(hit.shape, "hit shape")
    
    
    
    #print(scale[][in_edge])
    if False:    
        print(cloth.eidx[eidx[hit]])
        print(cloth.eidx[tidx[hit]])
        
    
    
    #have to do in margin or crossed.
    
    
    #If crossed also flip direction
    
    #have to get the spit_l for start and current
    ##if the dot of start and current is negative flip the move
    #direction because its on the wrong side.
    
    
    #print(spit_l[2] <)
    
    #so now I have this and spit.
    
    #print(cloth.four_edge_co[0])
    
    id = 1
    bpy.data.objects['ef'].location = this[id] + (cross1[id] * -spit_dot[id])
    bpy.data.objects['efg'].location = this[id - 1] + (cross1[id - 1] * -spit_dot[id - 1])
    
    ar = np.empty((eidx.shape[0], 2), dtype=np.int32)
    ar[:, 0] = eidx
    ar[:, 1] = tidx
    aru = eliminate_duplicate_pairs(ar, sort=False)
        
    #print(aru[0][41])
    #print(spit_dot[2])

#    I suppose every edge is both a tri and an edge
#    since we're' doubling things I wonder if I don't'
#    need to be doubling things. is there another way??
#    
#    the reason we are doubling things is because the tris
#    are a copy of the edges here. Doesnt that mean that the
#    same indexing used to tile could be used to index the boxes?
#    
#    is there a way to pair the edge edge with the tri edge?
#    do we even need to? 


    return (or_vec - eevecs * scale) + t[::2], (scale < 0) & (scale > -1)
    print(e_dot)
    return

    e.shape = shh
    t.shape = shh

    #els4 = cloth.four_edge_co
    #ers4
    
    #print(cloth.four_edge_co.shape)
    #print(cloth.four_edge_co[:2])
    
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
                
            
            testing = True
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
    

class EdgeCollide():
    name = "ed"
    
    def __init__(self, cloth, flood=False):

        # -----------------------
        ob = cloth.ob
        # -----------------------

        self.box_max = cloth.ob.MC_props.sc_box_max
        
        self.edges = cloth.four_edge_co # cloth.start_co[cloth.eidx], cloth.co[cloth.eidx]
        
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        self.ees = []
        self.trs = []
        

def detect_collisions(cloth):
    
    sc = EdgeCollide(cloth)
    self_collisions_7(sc, cloth)
    ray_check_oc(sc, sc.ees, sc.trs, cloth)


def flood_collisions(cloth, flood=True):
    
    sc = EdgeCollide(cloth, flood=True)
    self_collisions_7(sc, cloth, flood=True)
    ray_check_oc(sc, sc.ees, sc.trs, cloth, flood=True)


def register():
    pass


def unregister():
    pass


if __name__ == "__main__":
    register()
