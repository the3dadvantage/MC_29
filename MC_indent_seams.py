try:
    import bpy
    import bmesh

except ImportError:
    pass

import numpy as np
import time


# universal ---------------
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


# universal ---------------
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


# universal ---------------
def get_panel_groups(Bobj, vg=None):
    """Creates a dictionary of boolean arrays
    for verts in each panel"""

    obm = get_bmesh(Bobj)
    count = len(obm.verts)
    groups = [i.name for i in Bobj.vertex_groups if i.name.startswith('P_')]
    bools = {}

    for i in groups:
        g_idx = Bobj.vertex_groups[i].index
        obm.verts.layers.deform.verify()
        dvert_lay = obm.verts.layers.deform.active
        boo = np.zeros(count, dtype=np.bool)
        bools[i] = boo

        relevant = obm.verts
        if vg is not None:
            relevant = [obm.verts[v] for v in vg]

        for v in relevant:
            idx = v.index
            dvert = v[dvert_lay]

            if g_idx in dvert:

                boo[idx] = True

    return bools


# universal ---------------
def create_spread_key(Bobj, key="flat", margin=5.0, hide_sew_edges=False):
    """Creates a shape key that spreads out
    the panels based on their bounding boxes.
    " margin " is the space between the panels.
    " hide_sew_edges " hides sew edges in edit mode.
    Use "alt + H" to unhide."""

    new_key = key + "_spread"
    keys = Bobj.data.shape_keys.key_blocks
    v_count = len(Bobj.data.vertices)
    co = np.empty((v_count, 3), dtype=np.float32)

    if new_key in keys:
        Bobj.data.shape_keys.key_blocks[new_key].data.foreach_get('co', co.ravel())
        return co

    co = np.empty((v_count, 3), dtype=np.float32)
    Bobj.data.shape_keys.key_blocks[key].data.foreach_get('co', co.ravel())

    panels = get_panel_groups(Bobj, vg=None)

    ls = 0.0
    for k, v in panels.items():
        x = np.min(co[:, 0][v])
        move = ls - x
        co[:, 0][v] += move

        ls = np.max(co[:, 0][v]) + margin

    new_key = key + "_spread"

    if new_key not in keys:
        Bobj.shape_key_add(name=new_key)

    if hide_sew_edges:
        obm = get_bmesh(Bobj)
        se = [e.index for e in obm.edges if len(e.link_faces) == 0]
        eboo = np.zeros(len(obm.edges), dtype=np.bool)
        eboo[se] = True
        Bobj.data.edges.foreach_set('hide', eboo)

    Bobj.data.shape_keys.key_blocks[new_key].data.foreach_set('co', co.ravel())
    Bobj.data.update()
    return co


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


# universal ---------------
def get_proxy_co_mods(ob, co=None, proxy=None, return_proxy=False, types=["SOLIDIFY"]):
    """Gets co with modifiers like cloth exculding mods in types"""
    mods = [m for m in ob.modifiers]
    views = [m.show_viewport for m in mods]

    for m in mods:
        if m.type in types:
            m.show_viewport = False

    if proxy is None:

        dg = bpy.context.evaluated_depsgraph_get()
        prox = ob.evaluated_get(dg)
        proxy = prox.to_mesh()

    if co is None:
        vc = len(proxy.vertices)
        co = np.empty((vc, 3), dtype=np.float32)

    proxy.vertices.foreach_get('co', co.ravel())
    if return_proxy:
        return co, proxy, prox

    ob.to_mesh_clear()

    for m, v in zip(mods, views):
        m.show_viewport = v

    return co


def test_sort_idx(obm):
    """For testing a line indent
    (had to subdivide mesh lines where linked edges
    led to more than one possible next vert)"""

    idx = [421, 595, 618, 992, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169]
    start = 1129
    end = 1102

    escape = False
    count = 0
    v = start
    reordered = [start]
    while not escape:
        vert = obm.verts[v]
        new = [e.other_vert(vert).index for e in vert.link_edges if e.other_vert(vert).index in idx and e.other_vert(vert).index not in reordered]
        reordered += new
        v = new[0]
        if v == end:
            escape = True
        if count > 100:
            escape = True


    #return reordered
    return reordered, np.array(reordered) + 1170


def generate_offset_lines(ob=None, vs=None):
    """experiment with select history.
    Not used"""

    if True:
        if ob is None:
            ob = bpy.data.objects['b2']
        obm = get_bmesh(ob, refresh=True)
        #ob = bpy.context.object
        #if ob.mode != 'OBJECT':
        obm.select_history.clear()
        if vs is None:
            vs = [0, 948, 250]
        for v in vs:
            obm.select_history.add(obm.verts[v])
            obm.verts[v].select = True

        obm.to_mesh(ob.data)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.vert_connect_path()



def connect_path(ob, obm, verts):
    """Used blender's built-in vert connect
    with the verts assuming correct selection
    order."""

    obm.verts.ensure_lookup_table()
    obm.select_history.clear()

    for v in verts:
        obm.select_history.add(obm.verts[v])
        obm.verts[v].select = True

    bmesh.ops.connect_verts(obm, verts=[obm.verts[v] for v in verts])
    return


# universal ---------------
def edges_edges_intersect_2d(a1,a2, b1,b2, return_scale=False):
    '''simple 2d line intersect'''
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = da[:, ::-1] * np.array([1,-1])
    denom = np.einsum('ij,ij->i', dap, db)
    num = np.einsum('ij,ij->i', dap, dp)
    scale = (num / denom)
    if return_scale:
        return b1 + db * scale[:, None], scale
    else:
        return b1 + db * scale[:, None]


def ccw(A,B,C):
    return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])


# Return true if line segments AB and CD intersect
def new_intersect(A,B,C,D):
    return (ccw(A,C,D) != ccw(B,C,D)) & (ccw(A,B,C) != ccw(A,B,D))


def get_mids(v1, v2, c1, c2):
    """Get the distance needed to move
    at an angle to be parallel to the
    offset line"""

    uv1 = v1 / np.sqrt(np.einsum('ij,ij->i', v1, v1))[:, None]

    mid_point = c1 + ((c2 - c1) * 0.5)
    ump = mid_point / np.sqrt(np.einsum('ij,ij->i', mid_point, mid_point))[:, None]

    dd = np.einsum('ij,ij->i', mid_point, uv1)
    scale = 1/ np.sin(np.arccos(np.abs(dd)))
    return ump * scale[:, None]

    d = np.einsum('ij,ij->i', uv1, ump)
    new_mid = ump * (1 / d)[:, None]
    return new_mid


def get_intersections(offset, co, cross):
    """Use offset vecs with intersect to
    get new plot points. Append both ends
    with perp line. Mnanage nans (mananage)"""
    vc = co.shape[0]
    move1 = cross * offset
    co2 = co[:, :2]
    move = move1[:, :2]

    a1 = co2[:-2] + move[: -1]
    a2 = co2[1: -1] + move[: -1]

    b1 = co2[1: -1] + move[1:]
    b2 = co2[2:] + move[1:]

    intersect = edges_edges_intersect_2d(a1,a2, b1,b2)
    nans = np.isnan(intersect)
    infs = np.isinf(intersect)

    # manage nans (mananage)
    intersect[nans] = co2[1:-1][nans] + move[:-1][nans]
    intersect[infs] = co2[1:-1][infs] + move[:-1][infs]

    inter_3 = np.zeros((intersect.shape[0], 3), dtype=np.float32)
    inter_3[:, :2] = np.nan_to_num(intersect)

    # fill in the start and end
    ends = np.zeros((inter_3.shape[0] + 2, 3), dtype=np.float32)
    ends[1:-1] = inter_3
    ends[0] = co[0] + move1[0]
    ends[-1] = co[-1] + move1[-1]

    return ends


def manage_selection_states(states=None):
    """First time get select, active and modes
    for everything. Second time provide
    generated states to restore."""
    if states is None:
        obs = [ob for ob in bpy.data.objects]
        modes = [ob.mode for ob in bpy.data.objects]
        select = [ob.select_get() for ob in bpy.data.objects]
        active = bpy.context.view_layer.objects.active
        return obs, modes, select, active

    for i, ob in enumerate(states[0]):
        try:
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.mode_set(mode=states[1][i])
            ob.select_set(states[2][i])
        except:
            print(ob.name, "didn't work")
    bpy.context.view_layer.objects.active = states[3]


def get_tridex_2(ob, obm, mesh_mode=True):
    """Return an index for viewing the
    verts as triangles and a triangulated
    bmesh. 'mesh_mode' uses bpy.ops.mesh
    instead of bmesh.ops. Bmesh triangulate
    creates bad geometry. Mesh triangulate
    appears to be superior."""

    if ob.data.is_editmode:
        ob.update_from_editmode()

    pc = np.array([len(p.vertices) for p in ob.data.polygons], dtype=np.int32)
    if np.all(pc == 3):
        p_count = len(me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        me.polygons.foreach_get('vertices', tridex.ravel())
        return tridex, obm

    if mesh_mode:
        states = manage_selection_states()
        new_ob = ob.copy()
        new_me = ob.data.copy()
        new_ob.data = new_me
        bpy.context.collection.objects.link(new_ob)
        bpy.context.view_layer.objects.active = new_ob

        for sob in bpy.data.objects:
            sob.select_set(False)

        p_sel = np.ones(len(new_ob.data.polygons), dtype=np.bool)
        new_ob.data.polygons.foreach_set('select', p_sel)
        new_ob.data.update()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode="OBJECT")
        triobm = get_bmesh(new_ob)
        p_count = len(new_me.polygons)
        tridex = np.empty((p_count, 3), dtype=np.int32)
        new_me.polygons.foreach_get('vertices', tridex.ravel())
        bpy.data.objects.remove(new_ob)
        bpy.data.meshes.remove(new_me)

        manage_selection_states(states)

        return tridex, triobm

    tobm = bmesh.new()
    tobm.from_mesh(ob.data)
    bmesh.ops.triangulate(tobm, faces=tobm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    me = bpy.data.meshes.new('tris')
    tobm.to_mesh(me)

    p_count = len(me.polygons)
    tridex = np.empty((p_count, 3), dtype=np.int32)
    me.polygons.foreach_get('vertices', tridex.ravel())

    # clear unused tri mesh
    bpy.data.meshes.remove(me)

    return tridex, tobm


def inside_triangles(tris, points):
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
    margin = 0.0
    # !!!! ==================
    weights = np.array([w, u, v]).T
    check = (u >= margin) & (v >= margin) & (w >= margin)

    return check, weights


def poke_all(LG, debugging=False):

    test_sharps = np.zeros(LG.sharps.shape[0] + 2, dtype=np.bool)
    test_sharps[1:-1] = LG.sharps
    # LG.offset_points is the coordinates for each offset line


    # !!! testing:
    trimguy = bpy.data.objects['trimguy']
    testobm = get_bmesh(trimguy, refresh=True)
    test_idx = []
    # !!! -------

    # don't forget this is supposed to be the whol list !!!!!
    #for l in LG.offset_points: # !!!!!!!!!!!!!!!!!!!

    obm = get_bmesh(LG.ob, refresh=True)
    obm_shape = False
    if obm_shape:    
        spread_shape = obm.verts.layers.shape['flat_spread']
    tridex, triobm = get_tridex_2(LG.ob, obm)

    # -----------------------------------------------
    LG.pokes = [] # list of poked face vert index sets
    # -----------------------------------------------

    for l in LG.offset_points: # !!!!!!!!!!!!!!!!!!
        sharpies = l[test_sharps]
        this_poke = []

        s_count = 0
        for s in sharpies:
            tridex = np.array([[v.index for v in f.verts] for f in triobm.faces], dtype=np.int32)
            tridexer = np.arange(tridex.shape[0])

            trico = LG.spread_co[tridex]

            inxmin = np.min(trico[:, :, 0], 1) <= s[0]
            trico = trico[inxmin]
            inxmax = np.max(trico[:, :, 0], 1) >= s[0]
            trico = trico[inxmax]
            inymin = np.min(trico[:, :, 1], 1) <= s[1]
            trico = trico[inymin]
            inymax = np.max(trico[:, :, 1], 1) >= s[1]
            trico = trico[inymax]

            in_bounds = tridexer[inxmin][inxmax][inymin][inymax]
            points = np.tile(s, in_bounds.shape[0])
            points.shape = (in_bounds.shape[0], 3)
            check, weights = inside_triangles(trico, points)
            in_tris = in_bounds[check]

            # !!! testing:
            if np.any(in_tris):

                #test_idx += in_tris.tolist()
                triobm.faces.ensure_lookup_table()

                pokey = True
                #pokey = False

                if pokey:
                    po = bmesh.ops.poke(triobm, faces=[triobm.faces[in_tris[0]]], center_mode='MEAN')#, offset, center_mode, use_relative_offset)
                triobm.faces.ensure_lookup_table()
                triobm.verts.ensure_lookup_table()
                
                if obm_shape:
                    po['verts'][0][spread_shape] = s
                po['verts'][0].co = s

                this_poke += [po['verts'][0].index]
                test_idx += [po['verts'][0].index]
                LG.spread_co = np.append(LG.spread_co, [s], axis=0)
                triobm.to_mesh(trimguy.data)

                debugging = True
                #debugging = False
                if debugging:

                    if s_count == 0:
                        bpy.data.objects['ee'].location = s

                    if s_count == 1:
                        bpy.data.objects['eee'].location = s

                    if s_count == 2:
                        bpy.data.objects['e'].location = s

                s_count += 1

        LG.pokes += [this_poke]
        #break # !!! take the break out once done testing !!!
        #error
    if debugging:    
        triobm.to_mesh(trimguy.data)
        LG.triobm = triobm
        #print(test_idx, "this is test idx")
        print(LG.pokes, "LG.pokes???")
        deselect(trimguy, test_idx, 'vert')


def measure_angles(co):
    """Takes a set of coordinates and gives
    the angle in degrees where it's possible
    to measure angles. Fixes floating point
    errors."""

    if co.shape[0] < 3:
        print("!!! not enough points to measure angle !!!")
        return

    mids = co[1:-1]
    ls = co[:-2] - mids
    rs = co[2:] - mids
    uls = ls / np.sqrt(np.einsum('ij,ij->i', ls, ls))[:, None]
    urs = rs / np.sqrt(np.einsum('ij,ij->i', rs, rs))[:, None]
    dots = np.einsum('ij,ij->i', uls, urs)
    dots[dots < -1] = -1 # might be float errors but we're using unit vecs so it's close enough
    angle = np.arccos(dots)
    con = 180 / np.pi
    return angle * con


def create_offset_points(LG):
    """Find the lines that follow the existing
    lines with offset. One offset point for each
    point in the line. Number of lines comes from
    LG.lines which is a list of offsets.
    Retuns a set of new coordinats. One
    for each point for each line."""

    spread_co = LG.spread_co
    idx = LG.idx
    co = spread_co[idx]
    LG.angles = measure_angles(co)
    LG.sharps = (180 - LG.angles) > LG.angle_limit
    vecs = co[1:] - co[: -1]

    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross = np.cross(vecs, z)
    u_cross = cross / np.sqrt(np.einsum('ij,ij->i', cross, cross))[:, None]

    offset_points = []
    for i in LG.lines: # i is the offset
        nl = get_intersections(i, co, u_cross)
        offset_points.append(nl)

    LG.offset_points = offset_points

    # for boundaries at angles where the ends are inside the boundary:
    os = LG.overshoot
    for i in LG.offset_points:

        v1 = i[0] - i[1]
        uv1 = v1 / np.sqrt(v1 @ v1)
        i[0] += (uv1 * os)

        v2 = i[-1] - i[-2]
        uv2 = v2 / np.sqrt(v2 @ v2)
        i[-1] += (uv2 * os)

    debugging = True
    
    if np.any(LG.sharps):
        poke_all(LG, debugging)

    if debugging:
        line = bpy.data.objects['mm_guy']
        line.data.vertices.foreach_set('co', offset_points[0].ravel())
        line.data.update()
        test_sharps = np.zeros(LG.sharps.shape[0] + 2, dtype=np.bool)
        test_sharps[1:-1] = LG.sharps
        deselect(line, np.arange(test_sharps.shape[0])[test_sharps])

        line = bpy.data.objects['red_shirt']
        line.data.vertices.foreach_set('co', offset_points[1].ravel())
        line.data.update()

    return offset_points


class LineGroup():

    def __init__(self):
        self.lines = [0.003, -0.003]


def indent_seams(self=None, col_mod=None):
    """ "points" comes from:
    self.cg.single_needle_full_offsets_vert_pointers()
    It's a dictionary with each key corresponding
    to the name of a curve object and each item
    as a 1d array of vertex indicies the curve is
    generated from."""

    print("running indent_seams")

    if self is not None:
        points = self.cg.single_needle_full_offsets_vert_pointers()
        ob = self.cg_Bobj

    else:
        ob = bpy.context.object
        ob = bpy.data.objects['base']
        points = {'some string': [1,2,3,4,5]}

    obm = get_bmesh(ob, refresh=True)
    co = get_proxy_co_mods(ob)
    spread_co = create_spread_key(ob, margin=0.1)

    # --------------------------------------------------
    if self is None:
        idx, idx2 = test_sort_idx(obm)
        #create_offset_points(spread_co, idx, offset=0.01)
        points = {'some string': idx, 'some other string': idx2}
        points = {'some string': idx}#, 'some other string': idx2}

    for key, idx in points.items():
        LG = LineGroup()
        LG.overshoot = 0.1 # because offset lines might not intersect the boundary when at an angle to the boundary
        LG.ob = ob
        LG.angle_limit = 10 # 10 degrees before poke face
        LG.spread_co = spread_co
        LG.idx = idx
        LG.obm = obm
        LG.offset_points = create_offset_points(LG)

        LG.line_eidx = np.empty((LG.offset_points[0].shape[0] -1, 2), dtype=np.int32)
        LG.line_eidx[:, 0] = np.arange(LG.offset_points[0].shape[0] - 1)
        LG.line_eidx[:, 1] = np.arange(LG.offset_points[0].shape[0] - 1) + 1

        test_all_pokes = []

        debugging = True
        #debugging = False
        if not debugging:
            LG.triobm.to_mesh(ob.data)
            ob.data.update()
            LG.grid_eidx = np.empty((len(ob.data.edges), 2), dtype=np.int32)
            ob.data.edges.foreach_get('vertices', LG.grid_eidx.ravel())
            LG.g_edge_co = spread_co[LG.grid_eidx]

        if debugging:
            #LG.grid_eidx = np.empty((len(ob.data.edges), 2), dtype=np.int32)
            LG.grid_eidx = np.array([[e.verts[0].index, e.verts[1].index] for e in LG.triobm.edges], dtype=np.int32)
            #ob.data.edges.foreach_get('vertices', LG.grid_eidx.ravel())
            new_co = np.array([v.co for v in LG.triobm.verts], dtype=np.float32)
            LG.g_edge_co = new_co[LG.grid_eidx]
            spread_co = new_co

        ft = 0
        #for i, l in enumerate(LG.offset_points[::-1]):
            #i = 1
        for i, l in enumerate(LG.offset_points):
            pokes = LG.pokes[i]
            print(pokes, "pokes here !!!")

#            if I find all the sharps
#            at each sharp I find the face its in
#            I poke that face and move to the coordinate.
#            Get a new bmesh and new mesh coords rinse repeate
#            The new verts need to be indexed with largest numbers.
#            I guess the only way to find out if it works is to do it.

            LG.b_edge_co = l[LG.line_eidx]

            col_mod.detect_collisions(None, LG)
            ar = np.empty((LG.eidx.shape[0], 2), dtype=np.int32)
            ar[:, 0] = LG.eidx
            ar[:, 1] = LG.tidx
            new, pidx = eliminate_duplicate_pairs_keep_mirrors(ar)

            LG.eidx = new[:, 0]
            LG.tidx = new[:, 1]

            lxy = l[:, :2]
            spxy = spread_co[:, :2]

            a = lxy[LG.eidx]
            b = lxy[LG.eidx + 1] # because edges are like [[0, 1],[1, 2], [2, 3]]
            c = spxy[LG.grid_eidx[:, 0][LG.tidx]]
            d = spxy[LG.grid_eidx[:, 1][LG.tidx]]

            # check for overlap like where the end of an edge lands on a poke
            ovl1 = np.all(a-c == 0, axis=1)
            ovl2 = np.all(a-d == 0, axis=1)
            ovl3 = np.all(b-c == 0, axis=1)
            ovl4 = np.all(b-d == 0, axis=1)
            ovl = ovl1 + ovl2 + ovl3 + ovl4
            print(a, 'this is a')
            print(ovl.shape)
            print("///////////")
            print(LG.grid_eidx[:, 0][LG.tidx][ovl])
            print("///////////")
            print()

            # was counting on edge intersections where edge ends overlap pokes.
            # Doesnt always work so adding those to the interesct bool array below

            #print(LG.grid_eidx[np.arange(ovl1.shape[0])[ovl1]], "ovl1")

            locs, scale = edges_edges_intersect_2d(c, d, a, b, return_scale=True) # cdab because scale is from the second pair
            booly = new_intersect(a, b, c, d)# + ovl
            close = scale == 0
            booly[close] = True
            #booly[ovl] = True
            #print(booly, 'booly shape')
            #print(ovl, 'booly shape')
            print(np.sum(scale == 0), 'scale')
            print(np.sum(booly), 'scale')
            locs = locs[booly]
            scale = scale[booly]

            ge = LG.tidx[booly]
            ee = LG.eidx[booly]

            ags = np.argsort(ee)

            if ft == 2:
                # new edge segments sorted. Duplicates now need to be sorted by scale
                sharps = np.zeros(LG.sharps.shape[0] + 2, dtype=np.bool)
                sharps[1:-1] = LG.sharps
                sidx = np.arange(sharps.shape[0])[sharps]

            arg_scale = scale[ags]
            arg_ge = ge[ags]
            arg_locs = locs[ags]

            uni, inv, counts = np.unique(ee, return_inverse=True, return_counts=True)

            tidx = LG.tidx[booly]
            #deselect(ob, tidx, 'edge')

            stepper = 0
            scale_sort = []
            for i, c in enumerate(counts):
                if c > 1:
                    sarg = np.argsort(arg_scale[stepper:stepper+c])
                    idxer = (np.arange(c) + stepper)[sarg]
                    scale_sort += idxer.tolist()
                    stepper += c
                else:
                    scale_sort += [stepper]
                    stepper += 1

            sorted_edges = arg_ge[scale_sort]
            sorted_locs = arg_locs[scale_sort]

            tester = np.zeros((sorted_locs.shape[0], 3), dtype=np.float32)
            tester[:, :2] = sorted_locs
            miners = bpy.data.objects['miners_not_minors']

            print()
            print("-----------------------")


            with_pokes = [-1]

            # to avoid bisecting the same edge twice causing weird flips:
            bisect_boo = {}#np.zeros(len(LG.triobm.edges), dtype=bool)
            bisect_key = {}
            new_edge_idx = len(LG.triobm.edges) - 1

            if np.any(LG.sharps):
                for it, se in enumerate(LG.grid_eidx[sorted_edges]):
                    if (se[0] in pokes) | (se[1] in pokes):
                        print(se, "1188 in here?")

                        LG.triobm.verts.ensure_lookup_table()
                        eco = np.array([LG.triobm.verts[se[0]].co, LG.triobm.verts[se[1]].co], dtype=np.float32)
                        same = np.all((eco[:, :2] - sorted_locs[it]) == 0, axis=1)
                        #print(eco[:, :2] - sorted_locs[it])
                        #print(same, "same here???")
                        #same = np.all(np.abs(eco[:, :2] - sorted_locs[it]) < 0.0001, axis=1)
                        #samer = se[same].tolist()[0]
                        samer = se[same][0]
                        #print(samer, "samer")
                        if samer != with_pokes[-1]:
                            #print(samer, "samer 2")
                            #print()
                            #with_pokes += se[same].tolist()
                            with_pokes += [samer]
                    else:
                        LG.triobm.edges.ensure_lookup_table()
                        e = sorted_edges[it]

                        if e in bisect_boo:
                            eds = bisect_key[e]
                            vco = sorted_locs[it]
                            # figure out which edge's center is closer to the location
                            eco1 = np.array(LG.triobm.edges[eds[0]].verts[0].co, dtype=np.float32)[:2]
                            eco2 = np.array(LG.triobm.edges[eds[0]].verts[1].co, dtype=np.float32)[:2]
                            mid = (eco1 + eco2) / 2
                            dif = vco - mid
                            dist1 = dif @ dif
                            #print(dist1, "dist1 ???")


                            eco3 = np.array(LG.triobm.edges[eds[1]].verts[0].co, dtype=np.float32)[:2]
                            eco4 = np.array(LG.triobm.edges[eds[1]].verts[1].co, dtype=np.float32)[:2]
                            mid2 = (eco3 + eco4) / 2
                            dif2 = vco - mid2
                            dist2 = dif2 @ dif2

                            #print(dist2, "dist2")

                            e = eds[1]
                            if dist1 < dist2:
                                e = eds[0]


                            #e = bisect_key[e]
                            bisect_boo[e] = True
                        #print(e, "this is e")
                        bisect_boo[sorted_edges[it]] = True

                        #else:
                            #bisect_boo[e] = True
                        #bisect_key[e] = len(LG.triobm.edges) - 2
                        #if True:
                        if e in [3568, 3545, 2579]:

                            print(sorted_edges[it], e)

                            #if bisect_boo[e]:

                                #LG.triobm.edges.ensure_lookup_table()
                                #print(e, "prev e")
                                #e = bisect_key[e]
                                #print(len(LG.triobm.edges) - 1, "this was the e")
                            #else:
                                #bisect_boo[e] = True
                        bisecter = True
                        #bisecter = False
                        #print(e, "basic e")
                        if bisecter:
                            bisect = bmesh.ops.bisect_edges(LG.triobm, edges=[LG.triobm.edges[e]], cuts=1)
                            bi_vert = bisect['geom_split'][0]
                            escape = False
                            #if e in [3568, 3545, 2579]:
                                #escape = True
                                #print()
                                #print()
                                #print(bi_vert.index, "verts?")
                                #print()
                            bisect_key[e] = [ed.index for ed in bi_vert.link_edges]
                            new_edge_idx += 1
                            #print(bisect['geom_split'][0].index, "is this a vert index???")
                            xyz = np.zeros(3, dtype=np.float32)
                            xyz[:2] = sorted_locs[it]
                            bi_vert.co = xyz

                            with_pokes += [bi_vert.index]
                            if escape:
                                trimguy = bpy.data.objects['trimguy']
                                LG.triobm.to_mesh(trimguy.data)
                                deselect(trimguy, sel=with_pokes[1:])
                                #error

                            #if sorted_edges[it] == 2577:
                                #print(sorted_edges[it], "sorted edges")
                            #we bisect and get the new index
            print("-------------------")
            print("-------------------")
            print("-------------------")
            #print(with_pokes[1:], "this is with_pokes")
            print()
            print()
            print()
            test_all_pokes += with_pokes[1:]

            trimguy = bpy.data.objects['trimguy']

            if True:
                connect_path(trimguy, LG.triobm, with_pokes[1:])
                #print(with_pokes[1:])

#                for some reason with_pokes does not contain the poked faces
#                So connect path is missing the pokes.
#                Only seems to be happening on the second line segment
#                There also seems to be a bunch of verts where there should
#                only be one poked face. Separate those verts and there
#                is a hole in the mesh.


            LG.triobm.to_mesh(trimguy.data)
            #print(with_pokes[1:])
            if True:
                deselect(trimguy, sel=[])
                #generate_offset_lines(ob=trimguy, vs=with_pokes[1:])
            deselect(trimguy, sel=with_pokes[1:])
            #print(trimguy.mode, "mode??")
            #error
            #tob = bpy.data.objects['trimguy']
            #connect_path(tob, LG.triobm, with_pokes[1:])
            #what do I freaking have here....
            #so sorted locs comes from the mesh thats already been poked.
            #sorted edges is the intersections.
            #looks like the intersect edges where a vert matches the poke index
            #is always an intersection where an edge is not really intersecting
            #but shares a coordinate.
            #Should prolly verify this by testing a bunch of cases.
            #I need bisect the grid edge and get a new vertex index except at pokes.
            #at a poke I can insert the poke index instead of the bisect index.



                        #print(sorted_locs[it], "sorted locs")
                        #print(LG.triobm.verts[se[0]].co, "co 0")
                        #print(LG.triobm.verts[se[1]].co, "co 1")
                        #print(se, "this is the edges")

            #print(np.max(LG.grid_eidx[sorted_edges]), "this is sorted edges")
            print("-----------------------")
            print()

            #if ft == 1:
            if False:
                miners.data.vertices.foreach_set('co', tester.ravel())
                miners.data.update()

            ft += 1
        if True:
            trimguy = bpy.data.objects['trimguy']
            LG.triobm.to_mesh(trimguy.data)
            deselect(trimguy, test_all_pokes)

        #print(test_all_pokes, "test all pokes")
        #now we need to do the vert connect thing based on the selection order
            # !!! don't forget the last offset point might be inside
            #   the outer boundary line. Maybe bisect the last edge/edges
            #   on a final pass?

#            I can drop the poke faces but I need to find out
#            if I can bisect an edge, then another then poke
#            a face and get a predictable vert order from
#            the bmesh.

#            we have the edge collisions in order now.
#
#            sorted_edges
#            sorted_locs
#            Could bisect edges then place the new vert at the loc.
#            have to figure poke faces into the sorting.
#            maybe just do poke faces after the fact?
#            crap. still have to sort selection order for connected verts.



#            !!! have to accumulate indicies to keep track as we add each edge to the mesh
#
#
#            havu keep track of seamies.
#            so... sort after getting collisions indices based on border
#            where there are duplicates I will need the scale of the collision
#            to sort them. This means I need to return scales from the edge
#            intersect. Might also be able to debug the problem with detecting
#            intersections... The scale should be positive and less than
#            the scale of either of the two edge segs involved.
#
#
#            collisions gives me a list of things to check.
#            edge_edge gives me collision locs
#            duplicate pairs does its thing.
#
#
#
#
#            could go along and check every point and tri.
#            the trick is keeping the selection order of making
#            faces using the bmesh op.
#            so... say I get all the edge intersections.


    #generate_offset_lines(idx, obm)








    return

    self_collide() # to know which face the point is in

    poke_faces()

    position_pokes()

    make_cuts()

    barycentric_transform() # to get from spread to final enhance or shape with value of 1

    offset_on_normals()







    # write your own loop_cut that respects the angle at
    #   jointa so the the cut line will be offset correctly
    #   even when it makes a sharp turn

    # Once the lines are cut use the vertex normals to
    #   lower the line along the seam slightly rais
    #   the lines next to the seame. This should give
    #   the effect but keep the threads the right depth
    #   in the surface.

    # issues:
    # 1. not sure how to tell which way is out.
    #   I think that issue had to be solved in the rotate so I'll dig there.

#generate_offset_lines(None, None, None)

    
