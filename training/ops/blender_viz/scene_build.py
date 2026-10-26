"""Build the Blender scene for one recorded episode (bpy). Called by render.py; importable for interactive use.

Layout (all under collection `DroneViz`, rebuilt from scratch on every run, nothing else in the file is touched):
  DroneViz/Environment   ground (1 m grid), world, sun
  DroneViz/Regions       protected site / keep-out sphere / touchdown zone / spawn ring (only those the data defines)
  DroneViz/Attackers     red drones       DroneViz/Defenders   green drones     DroneViz/Agents  single-team drones
  DroneViz/Targets       goals, gates, pads (from the recording)
  DroneViz/Trajectories  per-segment path curves (history grows with time; optional faint full path)
  DroneViz/Markers       start / end markers, drop lines
  DroneViz/Events        capture bursts + persistent capture marks, proximity lines
  DroneViz/Cameras       Overview (default), Cinematic, TopDown, each with its own HUD sub-collection

Timing: sample i (sim time i*dt) is keyed on frame 1 + i*dt*fps/speed, sub-frame keys allowed, LINEAR interpolation
between samples (no smoothing / resampling / extrapolation). Scrubbing, pausing and replay are plain timeline actions.
Each drone root empty carries the recorded position (+ recorded orientation when the sim logged one). The visible model is
a child, so its display scale never changes the position. Optional `visual_heading` (velocity yaw for drones the sim
logged without orientation) lives on a separate child empty named `VisualHeading (not sim state)`.
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

import drone_model

ROLE_RGB = {"attacker": (0.90, 0.06, 0.05), "defender": (0.05, 0.78, 0.12), "agent": (0.05, 0.78, 0.12)}
ROLE_COLL = {"attacker": "Attackers", "defender": "Defenders", "agent": "Agents"}
TARGET_RGB = {"goal": (1.0, 0.62, 0.05), "gate": (1.0, 0.55, 0.0), "pad": (0.85, 0.85, 0.85)}
NEUTRAL = (0.45, 0.6, 0.85)
CAPTURE_RGB = (1.0, 0.78, 0.22)
HUD_TEXT, HUD_DIM = (0.93, 0.94, 0.96), (0.62, 0.65, 0.7)
ROOT = "DroneViz"

DEFAULTS = dict(
    fps=30, speed=1.0, hold_seconds=1.5,
    drone_scale=0.0,            # 0 = auto (model enlarged so it stays readable at scene scale; positions unaffected)
    trails="history",           # history | full | both | none
    trail_width=0.0,            # [m] 0 = auto
    start_markers=True, end_markers=True, drop_lines=True,
    capture_bursts=True, capture_marks=True,
    proximity_lines=False, proximity_radius=1.0,   # [m] draw defender-attacker line when closer than this
    fit_ground=True,            # keep the ground under every drone in frame (off for close-ups of high-altitude tasks)
    regions=True, targets=True, target_trails=True, grid=1.0, axes=True,
    visual_heading=False, hud=True, title=None, roles=None,
    overview_elev=34.0, overview_azim=-62.0, cinematic_elev=16.0, cinematic_orbit=40.0, cinematic_follow=0.3,
)


# ------------------------------------------------------------------------------------------ housekeeping
def clear():
    """Remove a previous DroneViz build (objects, collections, viz_* data). Leaves the rest of the file alone."""
    root = bpy.data.collections.get(ROOT)
    if root:
        def walk(c):
            for ch in list(c.children):
                walk(ch)
            for ob in list(c.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.collections.remove(c)
        walk(root)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.texts,
                 bpy.data.node_groups, bpy.data.worlds):
        for d in list(coll):
            if d.name.startswith("viz") and d.users == 0:
                coll.remove(d)
    for h in list(bpy.app.handlers.frame_change_pre):
        if getattr(h, "__name__", "") == "viz_hud_update":
            bpy.app.handlers.frame_change_pre.remove(h)


def coll(path):
    parent = bpy.context.scene.collection
    for i, name in enumerate(path.split("/")):
        full = name if i == 0 else f"{name}"
        c = bpy.data.collections.get(full)
        if c is None:
            c = bpy.data.collections.new(full)
            parent.children.link(c)
        parent = c
    return parent


def link(ob, c):
    c.objects.link(ob)
    return ob


# ------------------------------------------------------------------------------------------ materials
def principled(name, rgb, rough=0.45, metal=0.0, emit=0.0, alpha=1.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    if emit > 0:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        _blend(m)
    return m


def _blend(m):
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    elif hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    if hasattr(m, "use_transparency_overlap"):
        m.use_transparency_overlap = False


def mix_rgba(nt):
    """ShaderNodeMix in RGBA mode; returns (node, factor, A, B, result) sockets (names are shared across data types)."""
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    fac = next(i for i in mix.inputs if i.name == "Factor" and i.type == "VALUE")
    A = next(i for i in mix.inputs if i.name == "A" and i.type == "RGBA")
    B = next(i for i in mix.inputs if i.name == "B" and i.type == "RGBA")
    R = next(o for o in mix.outputs if o.name == "Result" and o.type == "RGBA")
    return mix, fac, A, B, R


def state_material(name, rgb, dead_rgb=(0.12, 0.12, 0.13), rough=0.38, emit=0.0, coat=0.6, prop="downed"):
    """Team material whose colour fades to `dead_rgb` when the object's custom property `prop` (keyframed) is 1."""
    m = principled(name, rgb, rough=rough, emit=emit, coat=coat)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = prop
    mix, fac, A, B, R = mix_rgba(nt)
    A.default_value = (*rgb, 1)
    B.default_value = (*dead_rgb, 1)
    nt.links.new(at.outputs["Fac"], fac)
    nt.links.new(R, b.inputs["Base Color"])
    if emit > 0:
        nt.links.new(R, b.inputs["Emission Color"])
    return m


def emission_material(name, rgb, strength=2.0, alpha=1.0, fade_prop=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1)
    em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = alpha
    if fade_prop:
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "OBJECT"
        at.attribute_name = fade_prop
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = alpha
        nt.links.new(at.outputs["Fac"], mul.inputs[0])
        nt.links.new(mul.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if alpha < 1 or fade_prop:
        _blend(m)
    return m


def ground_material(cell, extent):
    m = bpy.data.materials.new("viz_ground")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.9
    base, line = (0.105, 0.115, 0.13, 1), (0.20, 0.215, 0.24, 1)
    if cell <= 0:
        b.inputs["Base Color"].default_value = base
        return m
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    w = 0.012 / cell * max(1.0, extent / 12.0)  # line half-width as a fraction of a cell, thicker on big scenes

    def axis_line(sock):
        d = nt.nodes.new("ShaderNodeMath"); d.operation = "DIVIDE"; d.inputs[1].default_value = cell
        nt.links.new(sock, d.inputs[0])
        f = nt.nodes.new("ShaderNodeMath"); f.operation = "FRACT"; nt.links.new(d.outputs[0], f.inputs[0])
        o = nt.nodes.new("ShaderNodeMath"); o.operation = "SUBTRACT"; o.inputs[0].default_value = 1.0
        nt.links.new(f.outputs[0], o.inputs[1])
        mn = nt.nodes.new("ShaderNodeMath"); mn.operation = "MINIMUM"
        nt.links.new(f.outputs[0], mn.inputs[0]); nt.links.new(o.outputs[0], mn.inputs[1])
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = w
        nt.links.new(mn.outputs[0], lt.inputs[0])
        return lt.outputs[0]

    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
    nt.links.new(axis_line(sep.outputs["X"]), mx.inputs[0])
    nt.links.new(axis_line(sep.outputs["Y"]), mx.inputs[1])
    mix, fac, A, B, R = mix_rgba(nt)
    A.default_value = base
    B.default_value = line
    nt.links.new(mx.outputs[0], fac)
    nt.links.new(R, b.inputs["Base Color"])
    return m


# ------------------------------------------------------------------------------------------ keyframing helpers
def _fcurve(id_data, path, index):
    """F-curve for (path, index) on id_data, created through keyframe_insert so Blender 4.4+ layered actions get their
    slot / channelbag set up the same way as on 3.x-4.3 legacy actions."""
    ad = id_data.animation_data
    act = ad.action if ad else None

    def find():
        if act is None:
            return None
        if getattr(act, "layers", None) and len(act.layers) and ad.action_slot is not None:
            for st in act.layers[0].strips:
                cb = st.channelbag(ad.action_slot)
                if cb:
                    return cb.fcurves.find(path, index=index)
            return None
        return act.fcurves.find(path, index=index)

    fc = find()
    if fc is None:
        try:
            is_array = id_data.path_resolve(path).__len__() > 0 if hasattr(id_data.path_resolve(path), "__len__") else False
        except Exception:
            is_array = False
        if is_array:
            id_data.keyframe_insert(data_path=path, index=index, frame=1.0)
        else:
            id_data.keyframe_insert(data_path=path, frame=1.0)
        ad, act = id_data.animation_data, id_data.animation_data.action
        fc = find()
        fc.keyframe_points.clear()
    return fc


INTERP = {}


def set_keys(id_data, path, frames, values, index=-1, interp="LINEAR", group=None):
    """Fast keyframe insertion: values (N,) for one channel or (N, C) for all C channels of `path`."""
    values = np.asarray(values, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not INTERP:
        INTERP.update({e.identifier: e.value for e in bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items})
    chans = range(values.shape[1]) if index < 0 else [index]
    for c, ch in enumerate(chans):
        fc = _fcurve(id_data, path, ch)
        n = len(frames)
        fc.keyframe_points.clear()
        fc.keyframe_points.add(n)
        co = np.empty(2 * n)
        co[0::2] = frames
        co[1::2] = values[:, c]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [INTERP[interp]] * n)
        fc.extrapolation = "CONSTANT"
        fc.update()


def key_bool_visibility(ob, frames, visible):
    """hide_render / hide_viewport keyed with CONSTANT interpolation at the samples where visibility changes."""
    visible = np.asarray(visible, bool)
    idx = np.r_[0, np.flatnonzero(visible[1:] != visible[:-1]) + 1]
    f = np.asarray(frames)[idx]
    v = (~visible[idx]).astype(float)
    set_keys(ob, "hide_render", f, v, interp="CONSTANT")
    set_keys(ob, "hide_viewport", f, v, interp="CONSTANT")


def key_prop(ob, prop, frames, values, interp="CONSTANT"):
    ob[prop] = float(values[0])
    set_keys(ob, f'["{prop}"]', frames, values, interp=interp)


# ------------------------------------------------------------------------------------------ geometry helpers
def mesh_object(name, verts, faces, mat, c, smooth=False):
    me = bpy.data.meshes.new(f"viz_{name}")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), c)


def disc(name, center, r, mat, c, z=0.003, seg=96):
    v = [(center[0], center[1], z)] + [(center[0] + r * math.cos(a), center[1] + r * math.sin(a), z)
                                       for a in np.linspace(0, 2 * math.pi, seg, endpoint=False)]
    return mesh_object(name, v, [(0, i + 1, (i + 1) % seg + 1) for i in range(seg)], mat, c)


def ring(name, center, r, width, mat, c, z=0.004, seg=128, dashes=0):
    v, f = [], []
    for i, a in enumerate(np.linspace(0, 2 * math.pi, seg, endpoint=False)):
        for rr in (r - width / 2, r + width / 2):
            v.append((center[0] + rr * math.cos(a), center[1] + rr * math.sin(a), z))
    for i in range(seg):
        if dashes and (i * dashes * 2 // seg) % 2:
            continue
        j = (i + 1) % seg
        f.append((2 * i, 2 * i + 1, 2 * j + 1, 2 * j))
    return mesh_object(name, v, f, mat, c)


def cylinder_wall(name, center, r, h, mat, c, seg=96):
    v, f = [], []
    for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
        v += [(center[0] + r * math.cos(a), center[1] + r * math.sin(a), 0.0), (center[0] + r * math.cos(a), center[1] + r * math.sin(a), h)]
    for i in range(seg):
        j = (i + 1) % seg
        f.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    return mesh_object(name, v, f, mat, c, smooth=True)


def hemisphere(name, center, r, mat, c, seg=64, rings=24):
    v, f = [], []
    for k in range(rings + 1):
        el = (math.pi / 2) * k / rings
        for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
            v.append((center[0] + r * math.cos(el) * math.cos(a), center[1] + r * math.cos(el) * math.sin(a), center[2] + r * math.sin(el)))
    for k in range(rings):
        for i in range(seg):
            j = (i + 1) % seg
            f.append((k * seg + i, k * seg + j, (k + 1) * seg + j, (k + 1) * seg + i))
    return mesh_object(name, v, f, mat, c, smooth=True)


def uv_sphere(name, r, mat, c, seg=24, rings=12):
    v, f = [], []
    for k in range(rings + 1):
        el = -math.pi / 2 + math.pi * k / rings
        for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
            v.append((r * math.cos(el) * math.cos(a), r * math.cos(el) * math.sin(a), r * math.sin(el)))
    for k in range(rings):
        for i in range(seg):
            j = (i + 1) % seg
            f.append((k * seg + i, k * seg + j, (k + 1) * seg + j, (k + 1) * seg + i))
    return mesh_object(name, v, f, mat, c, smooth=True)


def torus_object(name, R, r, mat, c, seg=48, rseg=10):
    v, f = [], []
    for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
        for b in np.linspace(0, 2 * math.pi, rseg, endpoint=False):
            v.append(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)))
    for i in range(seg):
        for j in range(rseg):
            i2, j2 = (i + 1) % seg, (j + 1) % rseg
            f.append((i * rseg + j, i2 * rseg + j, i2 * rseg + j2, i * rseg + j2))
    return mesh_object(name, v, f, mat, c, smooth=True)


def poly_curve(name, pts, width, mat, c):
    cu = bpy.data.curves.new(f"viz_{name}", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = width
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    co = np.ones((len(pts), 4))
    co[:, :3] = pts
    sp.points.foreach_set("co", co.ravel())
    cu.materials.append(mat)
    return link(bpy.data.objects.new(name, cu), c)


def no_shadow(ob):
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False


# ------------------------------------------------------------------------------------------ build
class Builder:
    def __init__(self, S, cfg):
        self.S, self.cfg = S, {**DEFAULTS, **(cfg or {})}
        c = self.cfg
        self.fps, self.speed = int(c["fps"]), float(c["speed"])
        self.frames = 1.0 + S["t"] * self.fps / self.speed           # sample i -> (fractional) frame
        self.frame_end = int(math.ceil(self.frames[-1] + c["hold_seconds"] * self.fps))
        lo, hi = S["bounds"]
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        pts = np.asarray(self._box_points(ground=False))
        self.dlo, self.dhi = pts.min(0), pts.max(0)  # recorded data (+ regions), without the ground plane
        self.extent = float(max(np.ptp(pts, axis=0).max(), 0.5))
        self.dscale = c["drone_scale"] or float(np.clip(self.extent / 2.2, 1.0, 6.0))   # Tello is ~0.1 m wide
        self.tw = c["trail_width"] or 0.0055 * self.extent ** 0.75
        roles = c.get("roles") or {}
        for a in S["agents"]:
            a["role"] = roles.get(a["name"], roles.get(a["role"], a["role"]))
        self.ob = {}

    # ------------------------------------------------------------------ environment
    def environment(self):
        sc = bpy.context.scene
        root = coll(ROOT)
        env = bpy.data.collections.new("Environment"); root.children.link(env)
        span = self.extent * 4 + 20
        cx, cy = (self.lo[:2] + self.hi[:2]) / 2
        g = mesh_object("Ground", [(cx - span, cy - span, 0), (cx + span, cy - span, 0), (cx + span, cy + span, 0), (cx - span, cy + span, 0)],
                        [(0, 1, 2, 3)], ground_material(self.cfg["grid"], self.extent), env)
        # world: soft studio sky (no HDRI dependency)
        w = bpy.data.worlds.new("viz_world")
        w.use_nodes = True
        bg = w.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (0.20, 0.23, 0.28, 1)
        bg.inputs["Strength"].default_value = 1.0
        sc.world = w
        sun = bpy.data.lights.new("viz_sun", "SUN")
        sun.energy = 3.5
        sun.angle = math.radians(4)
        so = link(bpy.data.objects.new("Sun", sun), env)
        so.rotation_euler = (math.radians(38), math.radians(8), math.radians(-35))
        if self.cfg["axes"]:
            self.axes(env)
        return env

    def axes(self, c):
        """Small grey x / y arrows at the sim origin (frame check: identity mapping sim -> Blender)."""
        m = principled("viz_axes", (0.55, 0.57, 0.6), rough=0.6)
        L = max(0.5, round(self.extent / 10, 1))
        w = L * 0.03
        for name, d in (("x", (1, 0)), ("y", (0, 1))):
            ux, uy = d
            px, py = -uy * w, ux * w
            v = [(px, py, 0.006), (-px, -py, 0.006), (ux * L - px, uy * L - py, 0.006), (ux * L + px, uy * L + py, 0.006),
                 (ux * L + 3 * px, uy * L + 3 * py, 0.006), (ux * L - 3 * px, uy * L - 3 * py, 0.006), (ux * (L + 6 * w), uy * (L + 6 * w), 0.006)]
            mesh_object(f"Axis_{name}", v, [(0, 1, 2, 3), (4, 5, 6)], m, c)
            cu = bpy.data.curves.new(f"viz_axis_lbl_{name}", "FONT"); cu.body = name; cu.size = L * 0.25; cu.align_x = "CENTER"
            cu.materials.append(m)
            t = link(bpy.data.objects.new(f"Axis_{name}_label", cu), c)
            t.location = (ux * (L + 12 * w) , uy * (L + 12 * w) - (0.1 * L if name == "x" else 0), 0.007)

    # ------------------------------------------------------------------ regions
    def regions(self):
        if not self.cfg["regions"] or not self.S["regions"]:
            return
        c = bpy.data.collections.new("Regions"); coll(ROOT).children.link(c)
        for r in self.S["regions"]:
            k, ctr, rad = r["kind"], np.asarray(r["center"], float), float(r["radius"])
            if k == "site":  # protected cylinder: breach = live attacker within `radius` horizontally
                rgb = ROLE_RGB["defender"]
                disc("Site_floor", ctr, rad, principled("viz_site_floor", rgb, rough=0.8, alpha=0.28), c)
                ring("Site_edge", ctr, rad, 0.025 * max(1, self.extent / 10), emission_material("viz_site_edge", rgb, 1.5), c, z=0.006)
                if r.get("height"):
                    w = cylinder_wall("Site_volume", ctr, rad, r["height"], principled("viz_site_wall", rgb, rough=0.3, alpha=0.08), c)
                    no_shadow(w)
                    ring("Site_top", (*ctr[:2], 0), rad, 0.012 * max(1, self.extent / 10), emission_material("viz_site_top", rgb, 1.0, alpha=0.6), c,
                         z=r["height"])
            elif k == "keepout":
                h = hemisphere("Keepout_sphere", ctr, rad, principled("viz_keepout", NEUTRAL, rough=0.25, alpha=0.10), c)
                no_shadow(h)
                ring("Keepout_edge", ctr, rad, 0.02, emission_material("viz_keepout_edge", NEUTRAL, 1.0), c, z=0.006)
            elif k == "touchdown":
                disc("Touchdown_zone", ctr, rad, principled("viz_touchdown", (0.95, 0.95, 0.95), rough=0.7, alpha=0.5), c, z=0.004)
            elif k == "ring":
                ring("Spawn_ring", ctr, rad, 0.02 * max(1, self.extent / 10), principled("viz_spawn_ring", (0.45, 0.47, 0.5), rough=0.8), c,
                     dashes=48)

    # ------------------------------------------------------------------ targets (goal / gates / pads)
    def targets(self):
        if not self.cfg["targets"] or not self.S["targets"]:
            return
        c = bpy.data.collections.new("Targets"); coll(ROOT).children.link(c)
        mats = {}
        for t in self.S["targets"]:
            k = t["kind"]
            if k not in mats:
                done = ROLE_RGB["agent"]
                mats[k] = state_material(f"viz_target_{k}", TARGET_RGB[k], dead_rgb=done, emit=0.6 if k != "pad" else 0.0,
                                         coat=0.0, prop="reached")
            P = t["pos"]
            if k == "goal":
                ob = uv_sphere(t["name"], 0.012 * max(1.0, self.dscale * 0.7), mats[k], c)
                halo = uv_sphere(t["name"] + "_halo", 0.03 * max(1.0, self.dscale * 0.7),
                                 principled("viz_goal_halo", TARGET_RGB["goal"], alpha=0.12), c)
                halo.parent = ob; no_shadow(halo)
            elif k == "gate":  # ring facing the base (gate normal = radial direction), radius = env gate_radius 0.4 m
                rg = float(self.S["meta"].get("gate_radius", 0.4))
                ob = torus_object(t["name"], rg, 0.012 * max(1, self.extent / 10), mats[k], c)
                n = P[0] / max(np.linalg.norm(P[0]), 1e-6)
                ob.rotation_mode = "QUATERNION"
                ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(n))
            else:  # pad: landing spot disc, colour switches to agent green once that drone has landed
                ob = mesh_object(t["name"], [(0, 0, 0)] + [(0.12 * math.cos(a), 0.12 * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)],
                                 [(0, i + 1, (i + 1) % 48 + 1) for i in range(48)], mats[k], c)
                P = P + np.array([0, 0, 0.005])
            ob["reached"] = 0.0
            if t.get("moving", np.ptp(P, axis=0).max() > 1e-4):
                set_keys(ob, "location", self.frames, P)
            else:
                ob.location = P[0]
            if t.get("reached") is not None:
                key_prop(ob, "reached", self.frames, t["reached"].astype(float))
            self.ob[t["name"]] = ob
            if k == "goal" and self.cfg["target_trails"] and t.get("moving"):
                tr = bpy.data.collections.get("Trajectories") or self._trajcoll()
                self._trail(t["name"], P, [(0, len(P) - 1)], principled("viz_goal_trail", TARGET_RGB["goal"], emit=0.3, alpha=0.55), tr, width=0.6)

    def _trajcoll(self):
        c = bpy.data.collections.new("Trajectories"); coll(ROOT).children.link(c)
        return c

    # ------------------------------------------------------------------ drones
    def drones(self):
        me = drone_model.build_mesh()
        while len(me.materials) < 5:
            me.materials.append(None)
        frame_m = principled("viz_frame", (0.06, 0.06, 0.07), rough=0.5)
        prop_m = principled("viz_prop_blur", (0.08, 0.08, 0.09), rough=0.4, alpha=0.32)
        lens_m = principled("viz_lens", (0.01, 0.01, 0.012), rough=0.05, metal=0.3)
        me.materials[drone_model.SLOT_FRAME] = frame_m
        me.materials[drone_model.SLOT_PROP] = prop_m
        me.materials[drone_model.SLOT_LENS] = lens_m
        team = {r: state_material(f"viz_team_{r}", rgb, coat=0.7) for r, rgb in ROLE_RGB.items()}
        led = {r: state_material(f"viz_led_{r}", rgb, emit=6.0, coat=0.0) for r, rgb in ROLE_RGB.items()}
        colls = {}
        for a in self.S["agents"]:
            role = a["role"]
            if role not in colls:
                colls[role] = bpy.data.collections.new(ROLE_COLL[role]); coll(ROOT).children.link(colls[role])
            c = colls[role]
            root = link(bpy.data.objects.new(a["name"], None), c)
            root.empty_display_type = "PLAIN_AXES"
            root.empty_display_size = 0.05
            root["role"] = role
            root["orientation_source"] = a["quat_source"]
            set_keys(root, "location", self.frames, a["pos"], group="sim state")
            root.rotation_mode = "QUATERNION"
            if a["quat"] is not None:
                q = np.asarray(a["quat"], float)
                q = q / np.linalg.norm(q, axis=-1, keepdims=True)
                # q and -q are the same rotation; flipping keeps neighbouring keys on one hemisphere for interpolation
                for i in range(1, len(q)):
                    if np.dot(q[i], q[i - 1]) < 0:
                        q[i] = -q[i]
                set_keys(root, "rotation_quaternion", self.frames, q, group="sim state")
            key_bool_visibility(root, self.frames, a["visible"])
            parent = root
            if a["quat"] is None and self.cfg["visual_heading"]:
                parent = self._visual_heading(a, root, c)
            body = link(bpy.data.objects.new(a["name"] + "_model", me), c)
            body.parent = parent
            body.scale = (self.dscale,) * 3
            for slot, mat in ((drone_model.SLOT_TEAM, team[role]), (drone_model.SLOT_LED, led[role])):
                body.material_slots[slot].link = "OBJECT"
                body.material_slots[slot].material = mat
            key_prop(body, "downed", self.frames, a["downed"].astype(float))
            key_bool_visibility(body, self.frames, a["visible"])
            self.ob[a["name"]] = root
            if self.cfg["drop_lines"]:
                self._drop_line(a, root, c)

    def _visual_heading(self, a, root, c):
        P = a["pos"]
        v = np.gradient(P[:, :2], axis=0)
        yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        e = link(bpy.data.objects.new(a["name"] + " VisualHeading (not sim state)", None), c)
        e.parent = root
        e.empty_display_size = 0.02
        e["note"] = "yaw from finite-difference velocity, display only; the sim logged no orientation for this drone"
        set_keys(e, "rotation_euler", self.frames, yaw, index=2)
        return e

    def _drop_line(self, a, root, c):
        m = bpy.data.materials.get("viz_dropline") or principled("viz_dropline", (0.6, 0.62, 0.66), rough=0.8, alpha=0.35)
        r = 0.004 * max(1, self.extent / 8)
        seg = 8
        v = [(r * math.cos(t), r * math.sin(t), z) for z in (0.0, 1.0) for t in np.linspace(0, 2 * math.pi, seg, endpoint=False)]
        f = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
        ob = mesh_object(a["name"] + "_dropline", v, f, m, c)
        no_shadow(ob)
        con = ob.constraints.new("COPY_LOCATION"); con.target = root; con.use_z = False
        ob.scale = (1, 1, 1)
        set_keys(ob, "scale", self.frames, np.maximum(a["pos"][:, 2], 1e-4), index=2)
        key_bool_visibility(ob, self.frames, a["visible"] & ~a["downed"])

    # ------------------------------------------------------------------ trajectories + markers
    def _trail(self, name, P, segs, mat, c, width=1.0, grow=True):
        obs = []
        for s, (i0, i1) in enumerate(segs):
            if i1 - i0 < 1:
                continue
            ob = poly_curve(f"{name}_trail_{s}", P[i0:i1 + 1], self.tw * width, mat, c)
            no_shadow(ob)
            if grow:  # bevel_factor_end over a POLY spline maps linearly onto the sample index = sim time
                cu = ob.data
                cu.bevel_factor_mapping_end = "RESOLUTION"
                set_keys(cu, "bevel_factor_end", [self.frames[i0], self.frames[i1]], [0.0, 1.0])
            obs.append(ob)
        return obs

    def trajectories(self):
        mode = self.cfg["trails"]
        if mode == "none":
            return
        c = bpy.data.collections.get("Trajectories") or self._trajcoll()
        for a in self.S["agents"]:
            rgb = ROLE_RGB[a["role"]]
            if mode in ("history", "both"):
                self._trail(a["name"], a["pos"], a["segments"], bpy.data.materials.get(f"viz_trail_{a['role']}") or
                            principled(f"viz_trail_{a['role']}", rgb, rough=0.5, emit=0.8, alpha=0.9), c)
            if mode in ("full", "both"):
                self._trail(a["name"] + "_full", a["pos"], a["segments"], bpy.data.materials.get(f"viz_ghost_{a['role']}") or
                            principled(f"viz_ghost_{a['role']}", rgb, rough=0.6, emit=0.3, alpha=0.25), c, width=0.6, grow=False)

    def markers(self):
        if not (self.cfg["start_markers"] or self.cfg["end_markers"]):
            return
        c = bpy.data.collections.new("Markers"); coll(ROOT).children.link(c)
        caps = {(e["agent"], e["step"]) for e in self.S["events"] if e["kind"] == "capture"}
        for a in self.S["agents"]:
            rgb = ROLE_RGB[a["role"]]
            sm = bpy.data.materials.get(f"viz_start_{a['role']}") or principled(f"viz_start_{a['role']}", rgb, rough=0.6, alpha=0.9)
            for s, (i0, i1) in enumerate(a["segments"]):
                if self.cfg["start_markers"]:  # hollow ground ring under each (re)spawn point, appears with it
                    ob = ring(f"{a['name']}_start_{s}", (0, 0), 0.07 * max(1, self.dscale * 0.6), 0.02 * max(1, self.dscale * 0.6), sm, c, z=0.0)
                    ob.location = (a["pos"][i0, 0], a["pos"][i0, 1], 0.008)
                    key_bool_visibility(ob, self.frames, np.arange(self.S["T"]) >= i0)
                captured = (a["name"], i1 + 1) in caps  # segment ended by an interception: the capture mark shows it
                if self.cfg["end_markers"] and not captured:
                    ob = uv_sphere(f"{a['name']}_end_{s}", 0.02 * max(1, self.dscale * 0.6), sm, c, seg=12, rings=6)
                    ob.location = a["pos"][i1]
                    key_bool_visibility(ob, self.frames, np.arange(self.S["T"]) >= i1)

    # ------------------------------------------------------------------ events
    def events(self):
        caps = [e for e in self.S["events"] if e["kind"] == "capture"]
        if not caps and not self.cfg["proximity_lines"]:
            return
        c = bpy.data.collections.new("Events"); coll(ROOT).children.link(c)
        flash = emission_material("viz_capture_flash", (1.0, 0.92, 0.6), 4.0, alpha=0.85, fade_prop="fade")
        mark = emission_material("viz_capture_mark", CAPTURE_RGB, 1.2)
        cr = float(self.S["meta"].get("capture_radius") or 0.4)
        dur = 0.5  # [s sim] flash duration
        for n, e in enumerate(caps):
            p = np.asarray(e["pos"], float)
            f0 = self.frames[e["step"]]
            f1 = f0 + dur * self.fps / self.speed
            if self.cfg["capture_bursts"]:  # expanding flash at the recorded capture point, radius -> capture radius
                ob = uv_sphere(f"Capture_{n:02d}_flash", 1.0, flash, c)
                no_shadow(ob)
                ob.location = p
                set_keys(ob, "scale", [f0 - 0.01, f0, f1], np.array([[0.01] * 3, [cr * 0.3] * 3, [cr * 1.2] * 3]))
                key_prop(ob, "fade", [f0, f1], [1.0, 0.0], interp="LINEAR")
                key_bool_visibility(ob, [1, f0, f1], [False, True, False])
            if self.cfg["capture_marks"]:  # persistent ground cross where the interception happened
                s = 0.06 * max(1, self.dscale * 0.6)
                w = s * 0.22
                v, f = [], []
                for ang in (math.pi / 4, -math.pi / 4):
                    ca, sa = math.cos(ang), math.sin(ang)
                    b = len(v)
                    v += [(-s * ca - w * sa, -s * sa + w * ca, 0), (s * ca - w * sa, s * sa + w * ca, 0),
                          (s * ca + w * sa, s * sa - w * ca, 0), (-s * ca + w * sa, -s * sa - w * ca, 0)]
                    f.append((b, b + 1, b + 2, b + 3))
                ob = mesh_object(f"Capture_{n:02d}_mark", v, f, mark, c)
                ob.location = (p[0], p[1], 0.009)
                key_bool_visibility(ob, [1, f0], [False, True])
        if self.cfg["proximity_lines"]:
            self._proximity(c)

    def _proximity(self, c):
        D = [a for a in self.S["agents"] if a["role"] == "defender"]
        A = [a for a in self.S["agents"] if a["role"] == "attacker"]
        m = emission_material("viz_proximity", (1.0, 0.85, 0.2), 1.5, alpha=0.8)
        for d in D:
            for a in A:
                live = a["visible"] & ~a["downed"]
                close = (np.linalg.norm(d["pos"] - a["pos"], axis=-1) < self.cfg["proximity_radius"]) & live
                if not close.any():
                    continue
                cu = bpy.data.curves.new(f"viz_prox_{d['name']}_{a['name']}", "CURVE")
                cu.dimensions = "3D"; cu.bevel_depth = self.tw * 0.5
                sp = cu.splines.new("POLY"); sp.points.add(1)
                cu.materials.append(m)
                ob = link(bpy.data.objects.new(f"Proximity_{d['name']}_{a['name']}", cu), c)
                no_shadow(ob)
                for k, tgt in enumerate((self.ob[d["name"]], self.ob[a["name"]])):
                    h = ob.modifiers.new(f"hook{k}", "HOOK")
                    h.object = tgt
                    h.vertex_indices_set([k])
                key_bool_visibility(ob, self.frames, close)

    # ------------------------------------------------------------------ cameras + HUD
    def _fit_distance(self, cam_ob, direction, target, pts, margin=0.86):
        """Smallest distance along -direction from `target` such that every point projects inside the frame."""
        from bpy_extras.object_utils import world_to_camera_view
        sc = bpy.context.scene
        lo_d, hi_d = 0.5, 500.0
        for _ in range(40):
            d = (lo_d + hi_d) / 2
            cam_ob.location = Vector(target) - Vector(direction) * d
            bpy.context.view_layer.update()
            ok = True
            for p in pts:
                v = world_to_camera_view(sc, cam_ob, Vector(p))
                if v.z <= 0 or abs(v.x - 0.5) > margin / 2 or abs(v.y - 0.5) > margin / 2:
                    ok = False
                    break
            lo_d, hi_d = (d, hi_d) if not ok else (lo_d, d)
        cam_ob.location = Vector(target) - Vector(direction) * hi_d
        return hi_d

    def _box_points(self, ground=None):
        """Points that must stay in frame: every visible recorded position (subsampled), their ground shadows, regions."""
        P = [a["pos"][a["visible"]] for a in self.S["agents"]] + [t["pos"] for t in self.S["targets"]]
        P = np.concatenate([p.reshape(-1, 3) for p in P])
        P = P[np.isfinite(P).all(-1)]
        P = P[:: max(1, len(P) // 400)]
        pad = 0.07 * (self.cfg["drone_scale"] or 1.0)  # keep whole drone models, not only their centres, in frame
        P = np.concatenate([P] + [P + np.array(o) * pad for o in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))])
        pts = [P]
        if self.cfg["fit_ground"] if ground is None else ground:
            G = P.copy(); G[:, 2] = 0
            pts.append(G)
        for r in self.S["regions"]:
            if r["kind"] == "ring":
                continue
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            ring = np.stack([r["center"][0] + r["radius"] * np.cos(a), r["center"][1] + r["radius"] * np.sin(a), np.zeros_like(a)], -1)
            pts.append(ring)
            if r.get("height"):
                pts.append(ring + [0, 0, r["height"]])
        return [tuple(p) for p in np.concatenate(pts)]

    def cameras(self, res):
        sc = bpy.context.scene
        sc.render.resolution_x, sc.render.resolution_y = res
        c = bpy.data.collections.new("Cameras"); coll(ROOT).children.link(c)
        pts = self._box_points()
        plo, phi = np.min(pts, 0), np.max(pts, 0)
        ctr = Vector(((plo + phi) / 2).tolist())
        if self.cfg["fit_ground"]:
            ctr.z = max(0.3, phi[2] * 0.4)
        cams = {}

        def make(name, lens=35.0, ortho=False):
            cd = bpy.data.cameras.new(f"viz_cam_{name}")
            cd.lens = lens
            cd.clip_start, cd.clip_end = 0.05, 2000
            if ortho:
                cd.type = "ORTHO"
            ob = link(bpy.data.objects.new(name, cd), c)
            cams[name] = ob
            return ob

        def aim(ob, direction):
            ob.rotation_mode = "QUATERNION"
            ob.rotation_quaternion = Vector(direction).to_track_quat("-Z", "Y")

        # 1. Overview (default): static elevated 3/4 view that keeps the whole recorded volume in frame
        ov = make("Overview", 32)
        az, el = self.cfg["overview_azim"], self.cfg["overview_elev"]
        dvec = Vector((-math.cos(math.radians(el)) * math.cos(math.radians(az)), -math.cos(math.radians(el)) * math.sin(math.radians(az)),
                       -math.sin(math.radians(el))))
        aim(ov, dvec)
        self._fit_distance(ov, dvec, ctr, pts)

        # 2. TopDown: orthographic, +x right / +y up on screen (sim map view)
        td = make("TopDown", ortho=True)
        aspect = res[0] / res[1]
        w, h = phi[0] - plo[0], phi[1] - plo[1]
        td.data.ortho_scale = max(w, h * aspect, 0.5) * 1.15
        td.location = (ctr.x, ctr.y, self.hi[2] + 20)
        td.rotation_mode = "XYZ"
        td.rotation_euler = (0, 0, 0)

        # 3. Cinematic: lower, longer lens, slow orbit; looks at a smoothed centroid of the live drones
        cn = make("Cinematic", 45)
        tgt = link(bpy.data.objects.new("Cinematic_target", None), c)
        tgt.empty_display_size = 0.2
        live = []
        for i in range(self.S["T"]):
            ps = [a["pos"][i] for a in self.S["agents"] if a["visible"][i] and not a["downed"][i]]
            live.append(np.mean(ps, 0) if ps else np.asarray(ctr))
        live = np.asarray(live)
        k = max(1, int(1.5 / self.S["dt"]))
        ker = np.ones(k) / k
        sm = np.stack([np.convolve(np.pad(live[:, j], (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid") for j in range(3)], -1)
        fw = float(self.cfg["cinematic_follow"])
        aimp = fw * sm + (1 - fw) * np.asarray(ctr)  # mostly scene centre: the whole interaction stays in view
        step = max(1, int(round(0.25 / self.S["dt"])))
        idx = np.r_[np.arange(0, self.S["T"], step), self.S["T"] - 1]
        set_keys(tgt, "location", self.frames[idx], aimp[idx])
        el = math.radians(self.cfg["cinematic_elev"])
        az0 = math.radians(self.cfg["overview_azim"] + 25)
        orbit = math.radians(self.cfg["cinematic_orbit"])
        dist = 0.0
        for az in (az0, az0 + orbit / 2, az0 + orbit):  # whole recorded volume stays in frame along the orbit
            dv = Vector((-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)))
            aim(cn, dv)
            dist = max(dist, self._fit_distance(cn, dv, ctr, pts, margin=0.92 - 0.25 * fw))
        con = cn.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
        fr = np.linspace(1, self.frame_end, 24)
        locs = []
        for f in fr:
            s = (f - 1) / max(1, self.frame_end - 1)
            s = s * s * (3 - 2 * s)
            az = az0 + orbit * s
            locs.append([ctr.x + dist * math.cos(el) * math.cos(az), ctr.y + dist * math.cos(el) * math.sin(az), ctr.z + dist * math.sin(el)])
        set_keys(cn, "location", fr, np.asarray(locs), interp="BEZIER")
        sc.camera = ov
        self.cams = cams
        if self.cfg["hud"]:
            for name, cam in cams.items():
                self.hud(name, cam, res)
        return cams

    def hud(self, name, cam, res):
        """Camera-locked overlay: title, provenance, legend, sim clock, per-step counters (all from the recording)."""
        c = bpy.data.collections.new(f"HUD_{name}"); bpy.data.collections["Cameras"].children.link(c)
        cd = cam.data
        dist = 1.0
        if cd.type == "ORTHO":
            half_w = cd.ortho_scale / 2
        else:
            half_w = dist * (cd.sensor_width / 2) / cd.lens
        half_h = half_w * res[1] / res[0]
        u = half_w / 50.0  # 1 HUD unit = 1% of the frame width

        def text(tname, body, x, y, size, rgb=HUD_TEXT, align="LEFT", bold=False):
            cu = bpy.data.curves.new(f"viz_hud_{name}_{tname}", "FONT")
            cu.body = body
            cu.size = size * u
            cu.align_x = align
            m = bpy.data.materials.get(f"viz_hud_{rgb}") or emission_material(f"viz_hud_{rgb}", rgb, 1.0)
            cu.materials.append(m)
            ob = link(bpy.data.objects.new(f"HUD_{name}_{tname}", cu), c)
            ob.parent = cam
            ob.location = (x * u - half_w, y * u - half_h, -dist if cd.type != "ORTHO" else -1.0)
            no_shadow(ob)
            if hasattr(ob, "visible_diffuse"):
                ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = False
            return ob

        H = half_h * 2 / u
        title = self.cfg["title"] or self.S["title"]
        text("title", title, 2.2, H - 4.0, 2.2)
        text("source", self.S["source"], 2.2, H - 6.6, 1.15, rgb=HUD_DIM)
        clock = text("clock", "t = 0.00 s", 97.8, H - 4.0, 2.0, align="RIGHT")
        counter = text("counter", "", 97.8, H - 6.6, 1.4, align="RIGHT")
        roles = sorted({a["role"] for a in self.S["agents"]})
        legend = []
        names = {"attacker": "attacker (red)", "defender": "defender (green)", "agent": "RL-controlled drone"}
        for r in roles:
            legend.append((names[r], ROLE_RGB[r]))
        kinds = {t["kind"] for t in self.S["targets"]}
        for k, lbl in (("goal", "goal (recorded)"), ("gate", "gate"), ("pad", "landing pad")):
            if k in kinds:
                legend.append((lbl, TARGET_RGB[k]))
        if any(e["kind"] == "capture" for e in self.S["events"]):
            legend.append(("interception (recorded capture)", CAPTURE_RGB))
        y = 3.0
        for lbl, rgb in reversed(legend):
            sq = 0.55 * u * 1.6
            sw = mesh_object(f"HUD_{name}_sw_{lbl}", [(-sq, -sq, 0), (sq, -sq, 0), (sq, sq, 0), (-sq, sq, 0)], [(0, 1, 2, 3)],
                             bpy.data.materials.get(f"viz_hud_{rgb}") or emission_material(f"viz_hud_{rgb}", rgb, 1.0), c)
            sw.parent = cam
            sw.location = (3.0 * u - half_w, (y + 0.45) * u - half_h, -dist if cd.type != "ORTHO" else -1.0)
            text(f"lg_{lbl}", lbl, 4.6, y, 1.3)
            y += 2.4
        notes = [f"1 grid cell = {self.cfg['grid']:g} m"] if self.cfg["grid"] else []
        if abs(self.dscale - 1) > 1e-3:
            notes.append(f"drone models drawn x{self.dscale:.1f} for visibility (positions exact)")
        if any(a["quat"] is None for a in self.S["agents"]):
            notes.append("no orientation recorded for some drones: shown level" + (", heading from velocity" if self.cfg["visual_heading"] else ""))
        if self.speed != 1:
            notes.append(f"playback x{self.speed:g}")
        text("notes", "   ".join(notes), 97.8, 3.0, 1.1, rgb=HUD_DIM, align="RIGHT")
        self.hud_objs = getattr(self, "hud_objs", {})
        self.hud_objs[name] = c

    def install_hud_driver(self):
        """Per-frame clock/counter text from the recorded per-step strings. Stored in the scene so a saved .blend
        keeps working (the text block is registered as a module; enable 'Auto Run Python Scripts' to scrub with it)."""
        sc = bpy.context.scene
        sc["viz_hud"] = json.dumps(dict(dt=self.S["dt"], fps=self.fps, speed=self.speed, T=self.S["T"], counter=self.S["hud"]))
        code = HUD_DRIVER
        txt = bpy.data.texts.get("viz_hud_driver.py") or bpy.data.texts.new("viz_hud_driver.py")
        txt.clear(); txt.write(code); txt.use_module = True
        ns = {}
        exec(compile(code, "viz_hud_driver.py", "exec"), ns)
        ns["register"]()
        ns["viz_hud_update"](sc)

    # ------------------------------------------------------------------ all
    def build(self, res):
        sc = bpy.context.scene
        coll(ROOT)
        self.environment()
        self.regions()
        self.drones()
        self.targets()
        self.trajectories()
        self.markers()
        self.events()
        self.cameras(res)
        if self.cfg["hud"]:
            self.install_hud_driver()
        sc.render.fps = self.fps
        sc.frame_start, sc.frame_end = 1, self.frame_end
        sc.frame_set(1)
        sc["viz_source_npz"] = self.S["path"]
        sc["viz_frame_of_sample0"] = 1.0
        sc["viz_frames_per_sim_second"] = self.fps / self.speed
        return self


HUD_DRIVER = r'''
import json
import bpy


def viz_hud_update(scene, depsgraph=None):
    raw = scene.get("viz_hud")
    if not raw:
        return
    h = json.loads(raw)
    f = scene.frame_current + scene.frame_subframe
    t = max(0.0, (f - 1.0) * h["speed"] / h["fps"])
    i = min(int(round(t / h["dt"])), h["T"] - 1)
    t = min(t, (h["T"] - 1) * h["dt"])
    for ob in scene.objects:
        if ob.type == "FONT" and ob.name.startswith("HUD_"):
            if ob.name.endswith("_clock"):
                ob.data.body = f"t = {t:5.2f} s"
            elif ob.name.endswith("_counter"):
                ob.data.body = h["counter"][i] if h["counter"] else ""


def register():
    for fn in list(bpy.app.handlers.frame_change_pre):
        if getattr(fn, "__name__", "") == "viz_hud_update":
            bpy.app.handlers.frame_change_pre.remove(fn)
    bpy.app.handlers.frame_change_pre.append(viz_hud_update)


if __name__ == "__main__":
    register()
'''
