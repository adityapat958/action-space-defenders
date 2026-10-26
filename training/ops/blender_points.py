"""Blender (4.x): render any trajectory .npz (ops/traj_io.py) as glowing point objects with trailing tails.

    blender -b -P ops/blender_points.py -- run.npz --env 0 --out /abs/path/out_ --mp4 [--still /abs/still.png]

Look: dark background + ground with a subtle grid (1 cell = --grid m), AgX view transform, Cycles (GPU OptiX/CUDA if
available, else CPU) with denoising, compositor fog-glow so the points read as small lights.
Content: blue = our drones/defenders, red = goal/attackers (hidden from the step they are downed, with a short white
"pop" at the kill point; reappear on respawn), green disc = protected site, dashed grey ring = attacker spawn radius.
Each point drags a tapered tail of the last --tail s (split at resets/respawns); for scenes with <= 2 points the full
past path is also drawn very faintly. A thin drop line to the ground gives a height cue.
Camera: 3/4 view (elevation --elev) that is solved so the whole bounds box stays in frame over a slow --orbit deg orbit.
Overlay: title (task + policy from the scene title, or --title), legend, sim clock and playback speed.
Clips are capped at --max-seconds of video; longer episodes are played faster (speed shown on screen), positions are
interpolated between samples so playback stays smooth at --fps.
Writes PNG frames <out>0001.png ...; --mp4 stitches <out>.mp4 (strip trailing _) with ffmpeg (yuv420p).
"""
import argparse, math, os, re, shutil, subprocess, sys, time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_io import load  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument("traj")
p.add_argument("--env", type=int, default=0)
p.add_argument("--out", required=True, help="Frame prefix, e.g. /abs/out/clip_")
p.add_argument("--fps", type=int, default=20)
p.add_argument("--max-seconds", type=float, default=15.0, help="Max video length; longer episodes play faster.")
p.add_argument("--seconds", type=float, default=None, help="Only use the first N sim seconds.")
p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
p.add_argument("--device", default="AUTO", help="AUTO|OPTIX|CUDA|CPU (Cycles)")
p.add_argument("--samples", type=int, default=24)
p.add_argument("--res", type=int, nargs=2, default=[1280, 720])
p.add_argument("--radius", type=float, default=0.08, help="Point sphere radius [m] (Tello ~0.1 m).")
p.add_argument("--tail", type=float, default=2.0, help="Tail length [sim s].")
p.add_argument("--ghost", type=int, default=-1, help="Faint full past path: 1 on, 0 off, -1 auto (<= 2 points).")
p.add_argument("--orbit", type=float, default=35.0, help="Camera orbit over the clip [deg].")
p.add_argument("--azim", type=float, default=-60.0, help="Start azimuth [deg] (0 = +x).")
p.add_argument("--elev", type=float, default=30.0)
p.add_argument("--grid", type=float, default=0.0, help="Grid cell [m]; 0 = auto (0.5 or 1).")
p.add_argument("--title", default=None)
p.add_argument("--still", default=None, help="Also copy one representative frame here.")
p.add_argument("--still-at", type=float, default=0.66, help="Fraction of the clip for --still.")
p.add_argument("--mp4", action="store_true")
p.add_argument("--frames", type=int, default=0, help="Debug: render only N evenly spaced frames.")
a = p.parse_args(argv)

S = load(a.traj, a.env)
dt = S["dt"]
T = S["T"] if not a.seconds else min(S["T"], int(round(a.seconds / dt)))
sim_len = (T - 1) * dt
speed = max(1.0, sim_len / a.max_seconds)
nfr = min(int(math.floor(sim_len / speed * a.fps)) + 1, int(a.max_seconds * a.fps))
tidx = np.minimum(np.arange(nfr) * speed / a.fps / dt, T - 1)  # fractional sample index per video frame
JUMP = 0.3  # [m] between samples => reset / respawn


# ---------------------------------------------------------------- titles
def nice_title(raw):
    if a.title:
        return a.title, ""
    task, _, label = raw.partition(" · ")
    tmap = {"AS-Tello-Hover-v0": "T1 Hover", "AS-Tello-Waypoint-v0": "T2 Waypoint", "AS-Tello-Approach-v0": "T4 Approach"}
    if task.startswith("Defend"):
        m = re.search(r"curriculum_stage=(\d+)", raw)
        lab = label.split(" ")[0]
        return "Defend" + (f" (stage {m.group(1)})" if m else ""), f"scripted gate · {lab}"
    t = tmap.get(task.strip(), task.strip())
    if label.startswith("baseline:"):
        return t, f"scripted {label.split(':', 1)[1].upper()} baseline"
    m = re.search(r"ppo_torch_([^/]+)/checkpoints/([^/]+)\.pt", label)
    if m:
        run = m.group(1).replace("tello_waypoint_ppo", "").replace("tello_hover_ppo", "").strip("_")
        ts = re.search(r"(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)", label)
        run = run or (f"run {ts.group(1)} {ts.group(2)}:{ts.group(3)}" if ts else "")
        return t, " · ".join(x for x in ("PPO policy", run, m.group(2)) if x)
    m = re.search(r"(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)[^/]*/([^/]+)\.(zip|pt)$", label)
    if m:
        return t, f"RL policy · run {m.group(1)} {m.group(2)}:{m.group(3)} · {m.group(4)}"
    return t, label


title, subtitle = nice_title(S["title"])
print("title:", title, "|", subtitle, f"| T={T} dt={dt:.4f} speed={speed:.2f}x frames={nfr}")

# ---------------------------------------------------------------- scene / render settings
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = a.engine
dev_used = "n/a"
if a.engine == "CYCLES":
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.device = "CPU"
    dev_used = "CPU"
    if a.device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ([a.device] if a.device != "AUTO" else ["OPTIX", "CUDA"]):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == kind]
                if devs:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"
                    dev_used = f"{kind} ({devs[0].name})"
                    break
            except Exception as e:  # noqa: BLE001
                print("device", kind, "unavailable:", e)
    try:
        sc.cycles.denoiser = "OPTIX" if dev_used.startswith("OPTIX") else "OPENIMAGEDENOISE"
    except Exception:
        pass
print("device:", dev_used)
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.resolution_percentage = 100
sc.render.fps = a.fps
sc.render.use_persistent_data = True
sc.render.image_settings.file_format = "PNG"
for vt in ("AgX", "Filmic"):
    try:
        sc.view_settings.view_transform = vt
        break
    except TypeError:
        pass
for lk in ("AgX - Medium High Contrast", "AgX - Base Contrast", "None"):
    try:
        sc.view_settings.look = lk
        break
    except TypeError:
        pass
sc.view_settings.exposure = 0.0
print("view:", sc.view_settings.view_transform, "/", sc.view_settings.look)

# compositor: soft glow on bright (emissive) pixels only
sc.use_nodes = True
nt = sc.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl, comp = nt.nodes.new("CompositorNodeRLayers"), nt.nodes.new("CompositorNodeComposite")
gl = nt.nodes.new("CompositorNodeGlare")
gl.glare_type = "FOG_GLOW"
for k, v in (("Threshold", 1.2), ("Strength", 0.6), ("Size", 0.35)):  # 4.5: inputs; older: properties
    if k in gl.inputs:
        gl.inputs[k].default_value = v
try:
    gl.threshold, gl.size, gl.mix = 1.2, 7, -0.4
except Exception:
    pass
nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
nt.links.new(gl.outputs["Image"], comp.inputs["Image"])


# ---------------------------------------------------------------- materials
def m_emit(name, rgb, strength, alpha=1.0, diffuse=None):
    """Emission (optionally over a diffuse base) mixed with transparency."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial")
    em = N.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*rgb, 1); em.inputs["Strength"].default_value = strength
    sh = em.outputs[0]
    if diffuse is not None:
        bs = N.new("ShaderNodeBsdfPrincipled"); bs.inputs["Base Color"].default_value = (*diffuse, 1)
        bs.inputs["Roughness"].default_value = 0.35
        ad = N.new("ShaderNodeAddShader"); L.new(bs.outputs[0], ad.inputs[0]); L.new(em.outputs[0], ad.inputs[1]); sh = ad.outputs[0]
    if alpha < 1.0:
        tr = N.new("ShaderNodeBsdfTransparent"); mx = N.new("ShaderNodeMixShader")
        mx.inputs[0].default_value = alpha
        L.new(tr.outputs[0], mx.inputs[1]); L.new(sh, mx.inputs[2]); sh = mx.outputs[0]
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            m.blend_method = "BLEND"
    L.new(sh, out.inputs["Surface"])
    m.diffuse_color = (*rgb, alpha)
    return m


def m_diffuse(name, rgb, rough=0.9):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = rough
    for k in ("Specular IOR Level", "Specular"):
        if k in b.inputs:
            b.inputs[k].default_value = 0.2
            break
    return m


def no_shadow(ob):
    for k in ("visible_shadow", "visible_glossy"):
        if hasattr(ob, k):
            setattr(ob, k, False)


def link(ob):
    sc.collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------- world, ground, grid, discs
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.012, 0.014, 0.02, 1); bg.inputs[1].default_value = 1.0

lo, hi = [np.array(v, float) for v in S["bounds"]]
lo[2] = 0.0; hi[2] = max(hi[2], 1.0)
ctr = (lo + hi) / 2
span = float(max((hi - lo)[:2].max(), 1.5))
cell = a.grid or (0.5 if span <= 6 else 1.0)
gpad = max(1.0, 0.35 * span)
glo = np.floor((lo[:2] - gpad) / cell) * cell; ghi = np.ceil((hi[:2] + gpad) / cell) * cell

bpy.ops.mesh.primitive_plane_add(size=1, location=(ctr[0], ctr[1], 0))
ground = bpy.context.object; ground.scale = (max(span * 12, 80.0), max(span * 12, 80.0), 1)
ground.data.materials.append(m_diffuse("ground", (0.035, 0.037, 0.045)))

gcu = bpy.data.curves.new("grid", "CURVE"); gcu.dimensions = "3D"; gcu.bevel_depth = 0.004 * max(1.0, span / 4); gcu.bevel_resolution = 0
for x in np.arange(glo[0], ghi[0] + 1e-6, cell):
    sp = gcu.splines.new("POLY"); sp.points.add(1); sp.points[0].co = (x, glo[1], 0.002, 1); sp.points[1].co = (x, ghi[1], 0.002, 1)
for y in np.arange(glo[1], ghi[1] + 1e-6, cell):
    sp = gcu.splines.new("POLY"); sp.points.add(1); sp.points[0].co = (glo[0], y, 0.002, 1); sp.points[1].co = (ghi[0], y, 0.002, 1)
grid = link(bpy.data.objects.new("grid", gcu)); grid.data.materials.append(m_emit("grid", (0.55, 0.6, 0.7), 0.16, alpha=0.4)); no_shadow(grid)

for i, dsc in enumerate(S["discs"]):
    cx, cy = float(dsc["center"][0]), float(dsc["center"][1])
    if dsc.get("ring"):  # dashed ring on the ground
        cu = bpy.data.curves.new(f"ring{i}", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.012 * max(1, span / 6); cu.bevel_resolution = 0
        nd = 48
        for k in range(nd):
            a0, a1 = 2 * np.pi * k / nd, 2 * np.pi * (k + 0.55) / nd
            sp = cu.splines.new("POLY"); sp.points.add(3)
            for q, ang in zip(sp.points, np.linspace(a0, a1, 4)):
                q.co = (cx + dsc["radius"] * np.cos(ang), cy + dsc["radius"] * np.sin(ang), 0.004, 1)
        ob = link(bpy.data.objects.new(cu.name, cu)); ob.data.materials.append(m_emit(f"ring{i}", dsc["color"], 0.6, alpha=0.7))
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=dsc["radius"], depth=0.004, location=(cx, cy, 0.003 + 0.001 * i))
        ob = bpy.context.object; ob.data.materials.append(m_emit(f"disc{i}", dsc["color"], 0.5, alpha=0.35))
        cu = bpy.data.curves.new(f"edge{i}", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.01 * max(1, span / 6); cu.bevel_resolution = 0
        sp = cu.splines.new("POLY"); sp.points.add(95); sp.use_cyclic_u = True
        for k, q in enumerate(sp.points):
            ang = 2 * np.pi * k / 96
            q.co = (cx + dsc["radius"] * np.cos(ang), cy + dsc["radius"] * np.sin(ang), 0.006, 1)
        e = link(bpy.data.objects.new(cu.name, cu)); e.data.materials.append(m_emit(f"edge{i}", dsc["color"], 1.2, alpha=0.9))
        no_shadow(e)
    no_shadow(ob)

# key light (soft sun -> small contact shadows under the points) + dim fill
sun = link(bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")))
sun.data.energy = 1.6; sun.data.angle = math.radians(6); sun.rotation_euler = (math.radians(35), 0, math.radians(-30))
fill = link(bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN")))
fill.data.energy = 0.25; fill.rotation_euler = (math.radians(70), 0, math.radians(150)); fill.data.use_shadow = False


# ---------------------------------------------------------------- per-point objects
def sample(P, alive, ti):
    """Position at fractional index ti (linear interp unless a reset/respawn sits in between). None = hidden."""
    i0 = int(math.floor(ti)); i1 = min(i0 + 1, len(P) - 1); f = ti - i0
    if alive is not None and not (alive[i0] and (f < 1e-6 or alive[i1])):
        return None
    x0, x1 = P[i0], P[i1]
    if not np.isfinite(x0).all():
        return None
    if f < 1e-6 or not np.isfinite(x1).all() or np.linalg.norm(x1 - x0) > JUMP:
        return x0
    return x0 * (1 - f) + x1 * f


def segments(P, alive, i_from, i_to):
    """Index runs in [i_from, i_to] that are finite, alive and without jumps."""
    runs, cur = [], []
    for k in range(i_from, i_to + 1):
        ok = np.isfinite(P[k]).all() and (alive is None or alive[k])
        if ok and cur and np.linalg.norm(P[k] - P[cur[-1]]) > JUMP:
            runs.append(cur); cur = []
        if ok:
            cur.append(k)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs


npts = sum(g["pos"].shape[1] for g in S["groups"] if g["trail"] > 0)
ghost_on = a.ghost == 1 or (a.ghost == -1 and npts <= 2)
tail_w = max(0.006, a.radius * 0.28) * (1 + 0.6 * (span > 6))
stem_m = m_emit("stem", (0.6, 0.65, 0.75), 0.15, alpha=0.35)
pop_m = m_emit("pop", (1.0, 0.95, 0.85), 6.0)
items = []
for g in S["groups"]:
    rgb = tuple(g["color"])
    m_pt = m_emit(g["name"], rgb, 2.2, diffuse=rgb)
    m_tail = m_emit(g["name"] + "_tail", rgb, 1.6, alpha=0.75)
    m_ghost = m_emit(g["name"] + "_ghost", rgb, 0.5, alpha=0.18)
    R = a.radius * g["size"]
    for j in range(g["pos"].shape[1]):
        P = g["pos"][:T, j].astype(float)
        al = None if g["alive"] is None else g["alive"][:T, j].astype(bool)
        if al is not None and not al.any():
            continue  # unused slot (parked forever)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=24, ring_count=12)
        ob = bpy.context.object; ob.name = f"{g['name']}_{j}"; ob.data.materials.append(m_pt)
        bpy.ops.object.shade_smooth()
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.004 * max(1, span / 4), depth=1.0)
        stem = bpy.context.object; stem.data.materials.append(stem_m); no_shadow(stem)
        has_tail = g["trail"] > 0
        tc = bpy.data.curves.new(f"tail_{ob.name}", "CURVE"); tc.dimensions = "3D"; tc.bevel_depth = tail_w * g["size"]
        tc.bevel_resolution = 2; tc.use_fill_caps = True
        tail = link(bpy.data.objects.new(tc.name, tc)); tail.data.materials.append(m_tail); no_shadow(tail)
        ghost = None
        if ghost_on and has_tail:
            gc = bpy.data.curves.new(f"ghost_{ob.name}", "CURVE"); gc.dimensions = "3D"; gc.bevel_depth = tail_w * 0.35; gc.bevel_resolution = 0
            ghost = link(bpy.data.objects.new(gc.name, gc)); ghost.data.materials.append(m_ghost); no_shadow(ghost)
        pop = None
        deaths = []
        if al is not None:
            deaths = [k + 1 for k in np.flatnonzero(al[:-1] & ~al[1:])]  # first dead sample index
            bpy.ops.mesh.primitive_uv_sphere_add(radius=R, segments=16, ring_count=8)
            pop = bpy.context.object; pop.data.materials.append(pop_m); no_shadow(pop)
        items.append(dict(g=g, P=P, al=al, ob=ob, stem=stem, tail=tail, ghost=ghost, pop=pop, deaths=deaths,
                          trail=0.0 if not has_tail else (a.tail if g["name"] != "goal" else min(a.tail, 1.0))))


def set_curve(cu, runs, P, taper, ti=None, head=None):
    cu.splines.clear()
    for run in runs:
        pts = [P[k] for k in run]
        if head is not None and run[-1] == int(math.floor(ti)):
            pts.append(head)
        if len(pts) < 2:
            continue
        sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
        n = len(pts)
        for q, (k, v) in zip(sp.points, enumerate(pts)):
            q.co = (float(v[0]), float(v[1]), float(v[2]), 1)
            if taper:
                q.radius = 0.15 + 0.85 * (k / (n - 1)) ** 1.3


# ---------------------------------------------------------------- camera (3/4, orbit solved to fit bounds)
cam_d = bpy.data.cameras.new("cam"); cam_d.lens = 32; cam_d.sensor_fit = "HORIZONTAL"; cam_d.sensor_width = 36
cam_d.clip_start = 0.05; cam_d.clip_end = 500
cam = link(bpy.data.objects.new("cam", cam_d)); sc.camera = cam
aspect = a.res[0] / a.res[1]
tx = 18 / cam_d.lens; ty = tx / aspect
tgt = np.array([ctr[0], ctr[1], 0.42 * hi[2]])
corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (0.0, hi[2])])
for dsc in S["discs"]:
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    corners = np.vstack([corners, np.c_[dsc["center"][0] + dsc["radius"] * np.cos(th), dsc["center"][1] + dsc["radius"] * np.sin(th), 0 * th]])


def cam_basis(az, el, d):
    C = tgt + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    f = (tgt - C) / np.linalg.norm(tgt - C); r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return C, f, r, u


def fits(az, el, d):
    C, f, r, u = cam_basis(az, el, d)
    v = corners - C; z = v @ f
    if (z <= 0.1).any():
        return False
    x, y = (v @ r) / z / tx, (v @ u) / z / ty
    return (np.abs(x) <= 0.9).all() and (y <= 0.72).all() and (y >= -0.9).all()  # leave room at the top for the title


def ease(s):
    return s * s * (3 - 2 * s)


def cam_az(i):
    return math.radians(a.azim + a.orbit * ease(i / max(nfr - 1, 1)))


el = math.radians(a.elev)
lo_d, hi_d = 0.5, 200.0
for _ in range(40):
    mid = (lo_d + hi_d) / 2
    ok = all(fits(cam_az(i), el, mid) for i in np.linspace(0, nfr - 1, 9).astype(int))
    lo_d, hi_d = (lo_d, mid) if ok else (mid, hi_d)
cam_dist = hi_d
print(f"camera distance {cam_dist:.2f} m, target {tgt.round(2)}")


def place_cam(i):
    C, f, r, u = cam_basis(cam_az(i), el, cam_dist)
    M = np.eye(3); M[:, 0], M[:, 1], M[:, 2] = r, u, -f
    cam.matrix_world = _mat4(M, C)


def _mat4(M, C):
    from mathutils import Matrix
    return Matrix(((M[0, 0], M[0, 1], M[0, 2], C[0]), (M[1, 0], M[1, 1], M[1, 2], C[1]),
                   (M[2, 0], M[2, 1], M[2, 2], C[2]), (0, 0, 0, 1)))


# ---------------------------------------------------------------- overlay text (child of camera, emission only)
DZ = 1.0  # text plane distance in front of the camera
HW, HH = tx * DZ, ty * DZ  # half extents of the view at DZ


def text(body, x, y, size, rgb=(0.9, 0.92, 0.95), strength=1.0, align="LEFT"):
    td = bpy.data.curves.new("txt", "FONT"); td.body = body; td.size = size * 2 * HH; td.align_x = align
    ob = link(bpy.data.objects.new("txt", td)); ob.parent = cam
    ob.location = (x * HW, y * HH, -DZ)
    ob.data.materials.append(m_emit(f"txt{len(bpy.data.materials)}", rgb, strength))
    for k in ("visible_shadow", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"):
        if hasattr(ob, k):
            setattr(ob, k, False)
    return ob


text(title, -0.95, 0.86, 0.055, strength=1.1)
if subtitle:
    text(subtitle, -0.95, 0.775, 0.034, rgb=(0.65, 0.68, 0.75), strength=0.9)
leg_y = -0.92
xs = -0.95
for g in S["groups"]:
    lab = {"drone": "drone", "goal": "goal", "defenders": "defenders", "attackers": "attackers", "agents": "drones"}.get(g["name"], g["name"])
    bpy.ops.mesh.primitive_circle_add(vertices=24, radius=0.012 * HH * 2, fill_type="NGON")
    dot = bpy.context.object; dot.parent = cam; dot.location = (xs * HW + 0.012 * HH * 2, (leg_y + 0.018) * HH, -DZ)
    dot.data.materials.append(m_emit("dot_" + g["name"], tuple(g["color"]), 2.5)); no_shadow(dot)
    text(lab, xs + 0.045, leg_y, 0.03, rgb=(0.75, 0.78, 0.85))
    xs += 0.06 + 0.019 * len(lab)
for dsc in S["discs"]:
    if not dsc.get("ring"):
        bpy.ops.mesh.primitive_circle_add(vertices=24, radius=0.012 * HH * 2, fill_type="NGON")
        dot = bpy.context.object; dot.parent = cam; dot.location = (xs * HW + 0.012 * HH * 2, (leg_y + 0.018) * HH, -DZ)
        dot.data.materials.append(m_emit("dot_site", tuple(dsc["color"]), 2.0)); no_shadow(dot)
        text("protected site", xs + 0.045, leg_y, 0.03, rgb=(0.75, 0.78, 0.85))
        xs += 0.06 + 0.019 * 14
text(f"grid {cell:g} m", xs + 0.02, leg_y, 0.03, rgb=(0.5, 0.53, 0.6))
clock = text("", 0.95, 0.86, 0.04, rgb=(0.85, 0.87, 0.9), align="RIGHT")
info = text("", 0.95, 0.775, 0.032, rgb=(0.65, 0.68, 0.75), align="RIGHT")

# ---------------------------------------------------------------- render loop
order = list(range(nfr))
if a.frames:
    order = sorted(set(np.linspace(0, nfr - 1, a.frames).astype(int).tolist()))
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
t_start = time.time()
POP_S = 0.35  # [sim s] kill flash
for n, i in enumerate(order):
    ti = float(tidx[i]); k_now = int(math.floor(ti)); t_sim = ti * dt
    downed = 0
    for it in items:
        P, al = it["P"], it["al"]
        x = sample(P, al, ti)
        vis = x is not None
        if vis:
            it["ob"].location = [float(v) for v in x]
            h = max(float(x[2]), 1e-3)
            it["stem"].location = (float(x[0]), float(x[1]), h / 2); it["stem"].scale = (1, 1, h)
        for o in (it["ob"], it["stem"]):
            o.hide_render = not vis
        it["ob"].scale = (1, 1, 1) if vis else (0, 0, 0)
        k0 = max(0, k_now - int(round(it["trail"] / dt)))
        runs = segments(P, al, k0, k_now) if it["trail"] > 0 else []
        set_curve(it["tail"].data, runs, P, taper=True, ti=ti, head=x if vis else None)
        if it["ghost"] is not None:
            set_curve(it["ghost"].data, segments(P, al, 0, k_now), P, taper=False)
        if it["pop"] is not None:
            past = [d for d in it["deaths"] if d <= ti + 1e-6]
            downed += len(past)
            age = (ti - past[-1]) * dt if past else 1e9
            if age < POP_S:
                s = 1.0 - age / POP_S
                it["pop"].hide_render = False
                it["pop"].location = [float(v) for v in P[past[-1] - 1]]
                it["pop"].scale = (2.2 * s,) * 3
            else:
                it["pop"].hide_render = True
    clock.data.body = f"t = {t_sim:4.1f} s"
    extra = f"{speed:.2g}x speed" if speed > 1.01 else "real time"
    if any(it["al"] is not None for it in items):
        extra = f"attackers downed: {downed}   ·   " + extra
    info.data.body = extra
    place_cam(i)
    sc.frame_set(i + 1)
    sc.render.filepath = f"{a.out}{i + 1:04d}"
    bpy.ops.render.render(write_still=True)
    if n % 25 == 0:
        el_t = time.time() - t_start
        print(f"frame {n + 1}/{len(order)}  {el_t:.0f}s elapsed, eta {el_t / (n + 1) * (len(order) - n - 1):.0f}s", flush=True)
print(f"RENDERED {len(order)} frames in {time.time() - t_start:.1f}s ({dev_used}) -> {a.out}")

if a.still:
    k = order[min(len(order) - 1, int(a.still_at * len(order)))]
    shutil.copy(f"{a.out}{k + 1:04d}.png", a.still)
    print("still", a.still)
if a.mp4 and not a.frames:
    mp4 = a.out.rstrip("_") + ".mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", a.out + "%04d.png",
                    "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
                   check=False)
    print("wrote", mp4)
