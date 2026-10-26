"""Procedural Tello-class quadrotor mesh (bpy). Built once, shared by every drone object.

True size (DJI/Ryze Tello: 98 x 92.5 x 41 mm incl. prop guards, 3-inch props), body frame x forward / y left / z up,
which matches the Isaac Lab / EnvWorld body frame, so a sim quaternion (wxyz) drives the object directly.
Material slots: 0 team shell (per object, red attacker / green defender), 1 dark frame, 2 prop blur, 3 lens, 4 status LED
(per object). The props are drawn as static motion-blur discs: rotor speed is not part of the recorded sim state.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

SLOT_TEAM, SLOT_FRAME, SLOT_PROP, SLOT_LENS, SLOT_LED = range(5)
ARM = 0.034        # [m] motor offset along x and y (diagonal 0.048)
PROP_R = 0.032     # [m] 3-inch prop ~ 0.038 radius; slightly smaller so the guard ring reads
BODY = (0.062, 0.040, 0.026)


def _add(bm, geom_fn, mat, M):
    """Run a bmesh.ops.create_* into a temporary bmesh, transform by M, append to bm with material index mat."""
    tmp = bmesh.new()
    geom_fn(tmp)
    bmesh.ops.transform(tmp, matrix=M, verts=tmp.verts)
    for f in tmp.faces:
        f.material_index = mat
        f.smooth = True
    me = bpy.data.meshes.new("_tmp")
    tmp.to_mesh(me)
    tmp.free()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)


def _rounded_box(bm, size, bevel, mat, M=Matrix()):
    tmp = bmesh.new()
    bmesh.ops.create_cube(tmp, size=1.0)
    bmesh.ops.scale(tmp, vec=Vector(size), verts=tmp.verts)
    bmesh.ops.bevel(tmp, geom=list(tmp.edges), offset=bevel, segments=3, affect="EDGES", profile=0.5)
    bmesh.ops.transform(tmp, matrix=M, verts=tmp.verts)
    for f in tmp.faces:
        f.material_index = mat
        f.smooth = True
    me = bpy.data.meshes.new("_tmp")
    tmp.to_mesh(me)
    tmp.free()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)


def _torus(bm, R, r, mat, M, seg=32, rseg=8):
    verts = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        ring = []
        for j in range(rseg):
            b = 2 * math.pi * j / rseg
            p = Vector(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)))
            ring.append(bm.verts.new(M @ p))
        verts.append(ring)
    for i in range(seg):
        for j in range(rseg):
            f = bm.faces.new((verts[i][j], verts[(i + 1) % seg][j], verts[(i + 1) % seg][(j + 1) % rseg], verts[i][(j + 1) % rseg]))
            f.material_index = mat
            f.smooth = True


def build_mesh(name="viz_drone_mesh"):
    if name in bpy.data.meshes:
        return bpy.data.meshes[name]
    bm = bmesh.new()
    # fuselage: team-coloured shell over a dark core, slight nose-down wedge via two boxes
    _rounded_box(bm, BODY, 0.008, SLOT_TEAM, Matrix.Translation((0, 0, 0.004)))
    _rounded_box(bm, (BODY[0] * 0.92, BODY[1] * 1.04, 0.010), 0.003, SLOT_FRAME, Matrix.Translation((0, 0, -0.011)))
    # battery (rear, dark) and front camera lens
    _rounded_box(bm, (0.020, 0.030, 0.012), 0.003, SLOT_FRAME, Matrix.Translation((-0.030, 0, 0.002)))
    _add(bm, lambda t: bmesh.ops.create_cone(t, cap_ends=True, segments=20, radius1=0.0055, radius2=0.0055, depth=0.006),
         SLOT_LENS, Matrix.Translation((BODY[0] / 2 + 0.001, 0, 0.004)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    # status LED strip on the nose (team colour, emissive)
    _rounded_box(bm, (0.004, 0.022, 0.004), 0.0012, SLOT_LED, Matrix.Translation((BODY[0] / 2 - 0.002, 0, 0.016)))
    for sx in (1, -1):
        for sy in (1, -1):
            c = Vector((sx * ARM, sy * ARM, 0.0))
            ang = math.atan2(sy, sx)
            # arm
            _rounded_box(bm, (ARM * 1.414, 0.008, 0.007), 0.002, SLOT_FRAME,
                         Matrix.Translation(c / 2) @ Matrix.Rotation(ang, 4, "Z"))
            # motor bell
            _add(bm, lambda t: bmesh.ops.create_cone(t, cap_ends=True, segments=16, radius1=0.0065, radius2=0.0058, depth=0.012),
                 SLOT_FRAME, Matrix.Translation(c + Vector((0, 0, 0.004))))
            # prop blur disc + hub
            _add(bm, lambda t: bmesh.ops.create_circle(t, cap_ends=True, segments=40, radius=PROP_R),
                 SLOT_PROP, Matrix.Translation(c + Vector((0, 0, 0.0115))))
            # prop guard ring (team colour) joined to the arm
            _torus(bm, PROP_R + 0.004, 0.0022, SLOT_TEAM, Matrix.Translation(c + Vector((0, 0, 0.008))))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me
