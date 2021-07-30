
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

glob_iters = 0


# ==============================================
# ==============================================
def self_collisions_mem(sc, cloth=None, mem=False):

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
        
        if (t.shape[0] < sc.box_max) & (e.shape[0] < sc.box_max):
            sc.small_boxes.append([t, e])
        else:
            sc.big_boxes.append([t, e, [bmin, bmax]]) # using a dictionary or class might be faster !!!
            # !!! instead of passing bounds could figure out the min and max in the tree every time
            #       we divide. So divide the left and right for example then get the new bounds for
            #       each side and so on...
    
    sizes = [b[1].shape[0] for b in sc.big_boxes]
    if len(sizes) > 0:    
        check = max(sizes)
    
    limit = 20
    count = 1

    done = False
    while len(sc.big_boxes) > 0:
        b2(sc, cloth, count)

        sizes2 = [b[1].shape[0] for b in sc.big_boxes]
        if len(sizes2) > 0:
            if check / max(sizes2) < 1.5:
                done = True
        
        if count == limit:
            print("warning hit box limit")
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

        if mem:
            sc.ed_shapes += [ed.shape[0]]
            sc.trs_shapes += [trs.shape[0]]
        
        #tris = sc.tris[trs]
        #eds = cloth.eidx[ed]
        
        # detect link faces and broadcast
        ttt = cloth.tridex[trs][:, :, None]
        nlf_0 = cloth.eidx[ed][:, 0] == ttt
        
        sc.broadcast_shapes += [[nlf_0.shape[0], nlf_0.shape[1], nlf_0.shape[2]]]
        
        ab1 = np.any(nlf_0, axis=1)
        nlf_1 = cloth.eidx[ed][:, 1] == ttt
        
        sc.broadcast_shapes += [[nlf_1.shape[0], nlf_1.shape[1], nlf_1.shape[2]]]
        
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
            if mem:
                joined = np.empty((rt.shape[0], 2), dtype=np.int32)
                joined[:, 0] = re
                joined[:, 1] = rt
                ar, idx = eliminate_duplicate_pairs_keep_mirrors(joined)
                sc.ees += ar[:, 0].tolist()
                sc.trs += ar[:, 1].tolist()
            else:                
                sc.ees += re.tolist()
                sc.trs += rt.tolist()


def ray_check_mem(sc, ed, trs, cloth, mem=False):
    global glob_iters
    #margin = cloth.ob.MC_props.self_collide_margin
    margin = -0.001#cloth.ob.MC_props.self_collide_margin

    if mem:
        joined = np.empty((len(ed), 2), dtype=np.int32)
        joined[:, 0] = ed
        joined[:, 1] = trs
        ar, idx = eliminate_duplicate_pairs_keep_mirrors(joined)
        eidx = ar[:, 0]
        tidx = ar[:, 1]
    else:        
        eidx = np.array(ed, dtype=np.int32)
        tidx = np.array(trs, dtype=np.int32)
    
    if cloth.ob.MC_props.sew_tight:
        pass
    if False: # doesn't help because sew boundaries often need to flip
        ignore = cloth.sew_vert_edges[eidx]
        eidx = eidx[~ignore]
        tidx = tidx[~ignore]
        
    e = sc.edges[eidx]
    t = sc.tris[tidx]

    ls = e[:, 0]
    rs = e[:, 1]

    ori = t[:, 0]
    t1 = t[:, 1] - ori
    t2 = t[:, 2] - ori
    norms = np.cross(t1, t2)
    #u_norms = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    ls_vecs = ls - ori
    #rs_vecs = rs - ori
    start_dots = np.einsum('ij,ij->i', ls_vecs, norms)

    e_vec = rs - ls
    or_vec = ls_vecs
    e_n_dot = start_dots
    e_dot = np.einsum('ij,ij->i', e_vec, norms)
    scale = e_n_dot / e_dot
    
    #through = (scale < 0) & (scale > -1)
    through = (scale < margin) & (scale > (-1 + margin))
        
    plot = (or_vec - e_vec * scale[:, None]) + ori
    plots = plot[through]

    rt = t[through]
    check, weights = inside_triangles(rt, plots, margin= 0.0)
    u_norms = norms / np.sqrt(np.einsum('ij,ij->i', norms, norms))[:, None]
    #p_dirs = np.sign(e_dot[through][check])

    glob_iters += 1
    if np.any(check):
        points = cloth.eidx[eidx][through][check]
        #edp, edpidx = eliminate_duplicate_pairs(points)
        #points = edp
        
        p_norms = u_norms[through][check]#[edpidx]
        #p_e_vecs = e_vec[through][check]#[edpidx]
        
        cull_plots = plots[check]#[edpidx]
        edge_idx = cloth.peidx[eidx][through][check]#[edpidx]
        p_tris = rt[check]#[edpidx]
        p_tidx = tidx[through][check]#[edpidx]
        
        #if cloth.ob.MC_props.p1_final_enhance:    
        if cloth.ob.MC_props.flood_debug:
            if cloth.ob.data.is_editmode:
                se = [e for e in edge_idx if not cloth.sew_edges[e]]
                select_edit_mode(cloth.ob, se, type='e', deselect=True, obm=cloth.obm)
                #select_edit_mode(cloth.ob, edge_idx, type='e', deselect=True, obm=cloth.obm)
                #select_edit_mode(cloth.ob, [e.index for e in cloth.obm.edges if cloth.sew_edges[e.index]], type='e', deselect=True, obm=cloth.obm)
        
        panel = False
        #panel = True
        if panel:
            """
            problems with doing a panel sort:
                It's exclusive to p1 only working were vertex groups identify panels
            If the problem is being cause by linked search checking only the
            collide face:
                Say a tri on the outside of the cuff has a vert poking
                through. the linked verts would be on the correct
                side of the normal and it should work.
            What if the tri was on the opposite side. Like in the cuff.
            as the search increased if the sleeve curves around we
            would find an increasing number of verts on
            the wrong side of the normal.     
            
            
            What would it take to do the smarter search...
            Currently I find an edge that passes through a tri
            
            
            issues with panel:
                Where a panel folds over onto itself the normal is
                flipped. If the intersecting edge is part
                of one of these folds it might:
                    be colliding with a tri in the same panel
                    be colliding with a tri in a different panel.
                        in which case it shouldn't make a difference
                        because we are sorting based on which side
                        of other panels each vert is on.
            
            """
            
            
            f = cloth.is_fold[p_tidx]
            #is_fold = np.all(f1, axis=1)
            #tri_verts = cloth.tridex[p_tidx]
            at = cloth.group_indicies[cloth.tridex[p_tidx][:, 0]]
            lap = cloth.group_indicies[points[:, 0]]
            rap = cloth.group_indicies[points[:, 1]]
            l_final_sides = cloth.panel_matrix[at, lap] #* is_fold
            #l_final_sides[f] *= -1        
            r_final_sides = cloth.panel_matrix[at, rap] #* is_fold        
            #r_final_sides[f] *= -1

        movers = []
        moves = []

        cloth.obm.verts.ensure_lookup_table()
        cloth.generic_bool[:] = False # for folded panels/same panel sew edges
        
        for i in range(points.shape[0]):
            p_edge = edge_idx[i]

            if cloth.ob.MC_props.flood_skip_sew_edges:
                if cloth.sew_edges[p_edge]:
                    continue
                        
            
            # boundary edge collistion -----------------------
            pti = p_tidx[i]
            if cloth.boundary_tris[pti]:

                tri_verts = cloth.tridex[pti][cloth.v_boundary_bool[pti]]
                
                boundary_link_edges = [[e.index for e in cloth.obm.verts[v].link_edges if e.is_boundary] for v in tri_verts]    
                flat_list = [item for sublist in boundary_link_edges for item in sublist]    
                ec = cloth.co[cloth.eidx[flat_list]]
                elc = ec[:, 0]
                erc = ec[:, 1]
                #le_vecs = erc - elc
                p_ori = cull_plots[i] - elc
                
                evecs = erc - elc
                uev = evecs / np.sqrt(np.einsum('ij,ij->i', evecs, evecs))[:, None]
                cpoe = (uev * np.einsum('ij,ij->i', uev, p_ori)[:, None]) + elc
                e_moves = cpoe - cull_plots[i]
                shortest = np.argmin(np.einsum('ij,ij->i', e_moves, e_moves))
                short_moves = e_moves[[shortest, shortest]] * 0.5# * (1 + cloth.ob.MC_props.flood_overshoot) # one for each vert in the edge
                moves += short_moves.tolist()
                movers += points[i].tolist()

                # !!! need to test if we should continue !!!
                #continue
                # probably should not continue because boundaries can get stuck
                # continue seemed like a bad idea last time I checked
                # end boundary edge collistion -----------------------
                        
            pe = points[i]
            pt = p_tris[i]
            pn = p_norms[i]
            ptidx = cloth.tridex[p_tidx][i]
            # a dynamic search size might be in order here...
            if cloth.ob.MC_props.p1_final_enhance:
                for k, v in cloth.panel_bools.items():
                    if cloth.panel_types[k] == 'folded':
                        if v[ptidx[0]] & v[pe[0]]:
                            direction = cloth.same_panel_sew_edges[k].direction
                            if direction != 0.0:
                                evecs = cull_plots[i] - cloth.co[pe]
                                wrong_side = ((evecs @ pn) * direction) < 0.0
                                if np.sum(wrong_side) == 1:         
                                    #print(pe[wrong_side] not in used, "not in used")
                                    if not cloth.generic_bool[pe[wrong_side]]:
                                    #if pe[wrong_side] not in used:
                                        t_move = pn * (evecs[wrong_side] @ (pn * 1.01))
                                        #if move.shape[0] != 3:

                                        #cloth.co[pe[wrong_side]] += t_move
                                        moves += [t_move.tolist()]
                                        movers += pe[wrong_side].tolist()

                                cloth.generic_bool[pe[wrong_side]] = True                                    
                                continue

                        if cloth.panel_types[k] == "outer":
                            pass
                            # if triangle verts are outer panel the edge
                            #   verts need to move to the underside.
                            # if the tri is part of a fold it's opposite
                            # if the edge vert is in the outer and the tri
                            #   is a different panel the edge needs to move
                            #   to the outside.                     
                            #continue
            
            if not panel:    
                idx, summy = get_linked(cloth, pe, limit=cloth.ob.MC_props.flood_search_size, dynamic=True, pt=pt[0], pn=pn)
            #link_vecs = cloth.co[idx] - pt[0]
            # are most on the positive side or the negative side or neither?
            #link_norm_dot = np.einsum('j,ij->i', pn, link_vecs)
            #link_norm_dots = link_vecs @ pn
            #flippy = cloth.v_norms[idx] @ cloth.v_norms[pe[0]]# not the chihuahua
            #print(np.sign(flippy))
            #summy = np.sum(np.sign(link_norm_dots))
            ignore = True
            if panel:
                lfs = l_final_sides[i]
                rfs = r_final_sides[i]
                print('need a way to check if its a collision within the same panel. Maybe diagonal?')
                print('still need to sort into folded sections')
                print('if lfs/rfs is 0 revert to get_linked')
                print(lfs, rfs, "lfs rfs")
                summy = lfs
                ignore = False

            p0nd = (cloth.co[pe[0]] - pt[0]) @ pn
            compare0 = summy * p0nd
            if compare0 < 0:
                side = 0
            else:
                side = 1
            
            if panel:
                idx2 = pe[side]
            
            if abs(summy) > cloth.ob.MC_props.flood_bias:
                ignore = False
                idx2 = pe[side]

            if not ignore:
                movers += [idx2]
                m_or_vec = cloth.co[idx2] - pt[0]
                dist = m_or_vec @ (pn * (1 + cloth.ob.MC_props.flood_overshoot))
                #dist = m_or_vec @ pn# * cloth.ob.MC_props.flood_overshoot)
                #moves += [(pn * -dist) + (pn * -margin)]
                moves += [pn * -dist]# + (pn * -cloth.ob.MC_props.flood_overshoot)]
                    
        if len(moves) > 0:    
            uni, inv, counts = np.unique(movers, return_inverse=True, return_counts=True)
            ntn = np.nan_to_num(np.array(moves) / counts[inv][:, None])
            np.add.at(cloth.co, movers, ntn)


# ==============================================
# ==============================================


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


def select_edit_mode(ob, idx, type='v', deselect=True, obm=None):
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
            obm.edges.ensure_lookup_table()
            x = obm.edges
        
        if deselect:
            for i in x:
                i.select = False
        
        for i in idx:
            #sc.select_counter[i] += 1
            x[i].select = True
        
        if obm is None:
            bmesh.update_edit_mesh(ob.data)


def get_linked(cloth, idx, limit=10, dynamic=False, pt=None, pn=None):
    """put in the index of a vert. Get everything
    linked just like 'select_linked_pick()'.
    'dynamic' looks for a sum of verts on a side
    of the normal and keeps growing until it
    finds a bias."""
    obm = cloth.obm
    
    
    # !!! print("you will have a bug on small meshes if you run out of linked geometry !!!")
    
    vboos = np.zeros(len(obm.verts), dtype=np.bool)
    cvs = [obm.verts[i] for i in idx]
    #cvs = [obm.verts[idx]]
    escape = False
    while not escape:
        new = []
        for v in cvs:
            if not vboos[v.index]:
                vboos[v.index] = True

                if dynamic:                    
                    link_vecs = cloth.co[vboos] - pt

                    # are most on the positive side or the negative side or neither?
                    link_norm_dot = link_vecs @ pn
                    summy = np.sum(np.sign(link_norm_dot))
                    flips = True
                    flips = False
                    if flips:    
                        flippy = np.sign(cloth.v_norms[vboos] @ cloth.v_norms[idx[0]])
                        summy = np.sum(np.sign(link_norm_dot * flippy))
                    
                    if abs(summy) > cloth.ob.MC_props.flood_bias:
                        return np.arange(len(obm.verts))[vboos], summy
                    
                    if np.sum(vboos) >= limit:
                        return np.arange(len(obm.verts))[vboos], summy


                lv = [e.other_vert(v) for e in v.link_edges if not cloth.sew_edges[e.index]]
                #lv = [e.other_vert(v) for e in v.link_edges]
                culled = [v for v in lv if not vboos[v.index]]
                new += culled
        cvs = new
        
        if len(cvs) == 0:
            escape = True
        if np.sum(vboos) >= limit:
            escape = True    
            
    idxer = np.arange(len(obm.verts))[vboos]
    return idxer, summy


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
    margin = -0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)
    
    return check, weights


def b2(sc, cloth, count):

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
            
            if (t.shape[0] < sc.box_max) & (e.shape[0] < sc.box_max):
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

    #sh = cloth.pierce_co.shape[0]
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


def eliminate_pairs(ar1, ar2):
    """Remove pairs from ar1 that are in ar2"""
    x = np.array(np.random.rand(ar1.shape[1]), dtype=np.float32)
    y = ar1 @ x
    z = ar2 @ x
    booly = np.isin(x, y, invert=True)
    return ar1[booly], booly


def eliminate_duplicate_pairs_keep_mirrors(ar):
    """Finds unique index pairs assuming left
    and right side are different types:
    [[1, 2], [1, 2], [2, 1]] becomes:
    [[1, 2], [2, 1]]"""
    a = ar
    x = np.array(np.random.rand(a.shape[1]), dtype=np.float32)
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index


def eliminate_duplicate_pairs(ar):
    """Eliminates duplicates and mirror duplicates.
    for example, [1,4], [4,1] or duplicate occurrences of [1,4]
    Returns an Nx2 array."""
    # no idea how this works (probably sorcery) but it's really fast
    a = np.sort(ar, axis=1)
    x = np.array(np.random.rand(a.shape[1]), dtype=np.float32)
    y = a @ x
    unique, index = np.unique(y, return_index=True)
    return a[index], index


def side_sort(points, tridex):
    # get the unique pairs
    unip, p_idx = eliminate_duplicate_pairs(points)
    
    uni, idx, inv, counts = np.unique(unip, return_inverse=True, return_counts=True, return_index=True)
        
    cinv = counts[inv]
    sh = cinv.shape[0]
    cinv.shape = (sh//2, 2)

    culprits = counts > 2

    verts = uni[culprits]
    tris = np.repeat(tridex[p_idx], 2)[idx][culprits]

    if False:
        cloth.boundary_bool = np.array([[e.is_boundary for e in t.edges] for t in cloth.triobm.faces], dtype=np.bool)
        cloth.boundary_tris = np.array([np.any(b) for b in cloth.boundary_bool], dtype=np.bool)
        cloth.bt_edges = np.array([[[e.verts[0].index, e.verts[1].index] for e in t.edges] for t in cloth.triobm.faces], dtype=np.int32)

    return verts, tris

        
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
    

class PierceCollide():
    name = "sc"
    
    def __init__(self, cloth):
        
        # -----------------------
        tridex = cloth.tridex
        # -----------------------

        self.box_max = 300        
        testing = False
        testing = True
        if testing:    
            self.box_max = cloth.ob.MC_props.sc_box_max
        
        self.tris = cloth.co[tridex]
        self.edges = cloth.pierce_co#[cloth.pc_edges]
            
        self.big_boxes = [] # boxes that still need to be divided
        self.small_boxes = [] # finished boxes less than the maximum box size

        self.trs = []
        self.ees = []
        
        self.ed_shapes = []
        self.trs_shapes = []
        self.broadcast_shapes = []
        

def flood_collisions(cloth):
    sc = PierceCollide(cloth)
    
    # in p1 we are working with huge data sets so we need to conserve memory    
    # Note:
    # Flood produces mem errors where MC_self collide doesn't
    #   probably because you get more box overlap with edges
    #   and tris than with points and tris.
    if cloth.ob.MC_props.p1_final_enhance:
        self_collisions_mem(sc, cloth, mem=True)        
        ed = np.array(sc.ed_shapes)
        trs = np.array(sc.trs_shapes)
        brd = np.array(sc.broadcast_shapes)
        ray_check_mem(sc, sc.ees, sc.trs, cloth, mem=True)
        return    

    self_collisions_mem(sc, cloth, mem=False)
    ray_check_mem(sc, sc.ees, sc.trs, cloth, mem=True)
