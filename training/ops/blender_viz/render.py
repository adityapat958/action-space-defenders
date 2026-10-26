"""Blender entry point: recorded sim episode -> scene -> .blend / stills / mp4 clips.

    blender -b -P ops/blender_viz/render.py -- --config ops/blender_viz/configs/t3_defend.json \
        [--quality draft|preview|final] [--camera Overview|Cinematic|TopDown|all] [--still 0.6] [--anim] \
        [--frames 1 120] [--save-blend] [--validate] [--out DIR] [--set key=value ...]

Config (JSON) keys: npz (repo-relative or absolute), env, name, out_dir, plus any scene_build.DEFAULTS key
(fps, speed, trails, drone_scale, proximity_lines, roles, title, ...). --set overrides single keys (JSON values).
Re-running rebuilds only the DroneViz collection, so it is idempotent inside one .blend.
Outputs: <out>/<name>.blend, <out>/<name>_<Camera>.mp4 (H.264), <out>/<name>_<Camera>.png, <out>/<name>_validate.json
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viz_data  # noqa: E402
import scene_build  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
QUALITY = {
    "draft": dict(res=(960, 540), engine="EEVEE", samples=16),
    "preview": dict(res=(1280, 720), engine="EEVEE", samples=32),
    "final": dict(res=(1920, 1080), engine="EEVEE", samples=64),
    "cycles": dict(res=(1920, 1080), engine="CYCLES", samples=96),
}


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--npz", default=None, help="Override the config's trajectory file.")
    p.add_argument("--quality", default="preview", choices=list(QUALITY))
    p.add_argument("--res", type=int, nargs=2, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--engine", default=None, choices=["EEVEE", "CYCLES"])
    p.add_argument("--camera", default="Overview", help="Overview | Cinematic | TopDown | all | comma list")
    p.add_argument("--still", type=float, nargs="*", default=None, help="Still(s) at these clip fractions (default 0.6).")
    p.add_argument("--anim", action="store_true", help="Render the animation to mp4.")
    p.add_argument("--frames", type=int, nargs=2, default=None, help="Frame range for --anim.")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--set", nargs="*", default=[], help="key=json_value overrides of the config")
    return p.parse_args(argv)


def setup_render(q, args):
    sc = bpy.context.scene
    eng = args.engine or q["engine"]
    samples = args.samples or q["samples"]
    if eng == "EEVEE":
        sc.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version < (5, 0, 0) else "BLENDER_EEVEE"
        ee = sc.eevee
        ee.taa_render_samples = samples
        for k, v in (("use_shadows", True), ("use_raytracing", True), ("use_gtao", True), ("shadow_ray_count", 2),
                     ("shadow_step_count", 8)):
            if hasattr(ee, k):
                setattr(ee, k, v)
    else:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dev
                    prefs.get_devices()
                    if any(d.type == dev for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == dev
                        sc.cycles.device = "GPU"
                        break
                except TypeError:
                    continue
        except Exception:
            pass
    sc.render.film_transparent = False
    sc.render.use_motion_blur = False  # crisp positions: each frame shows the interpolated recorded state
    vs = sc.view_settings
    vs.view_transform = "Standard"  # keeps the team red / green at their sRGB values (AgX desaturates them)
    vs.look = "None"
    vs.exposure = 0.0
    sc.render.resolution_percentage = 100


def render_still(path, frame):
    sc = bpy.context.scene
    sc.frame_set(int(frame))
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_anim(path, f0, f1):
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = f0, f1
    r = sc.render
    r.image_settings.file_format = "FFMPEG"
    r.ffmpeg.format = "MPEG4"
    r.ffmpeg.codec = "H264"
    r.ffmpeg.constant_rate_factor = "HIGH"
    r.ffmpeg.ffmpeg_preset = "GOOD"
    r.ffmpeg.gopsize = sc.render.fps
    r.filepath = path
    r.use_file_extension = False
    r.use_lock_interface = True  # the HUD handler edits text data per frame
    bpy.ops.render.render(animation=True)


def select_camera(B, name):
    sc = bpy.context.scene
    sc.camera = B.cams[name]
    for n, c in getattr(B, "hud_objs", {}).items():
        c.hide_render = n != name
        c.hide_viewport = n != name


def validate(B, S, out_json):
    """Evaluate the built scene at every sample time and compare with the recording (no Blender rendering needed)."""
    sc = bpy.context.scene
    rep = dict(npz=S["path"], samples=S["T"], dt=S["dt"], fps=B.fps, speed=B.speed, agents={}, checks={})
    worst = 0.0
    worst_rot = 0.0
    idx = np.arange(S["T"])  # every recorded sample
    for a in S["agents"]:
        ob = B.ob[a["name"]]
        err, rot_err, vis_err = [], [], 0
        for i in idx:
            f = B.frames[i]
            sc.frame_set(int(math.floor(f)), subframe=float(f - math.floor(f)))
            hidden = bool(ob.hide_render)
            vis_err += int(hidden == bool(a["visible"][i]))
            if not a["visible"][i]:
                continue  # hidden objects are not evaluated by the depsgraph; nothing is drawn there
            p = np.array(ob.matrix_world.translation)
            err.append(np.linalg.norm(p - a["pos"][i]))
            if a["quat"] is not None and a["visible"][i]:
                q = np.array(ob.matrix_world.to_quaternion())
                rot_err.append(math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q, a["quat"][i] / np.linalg.norm(a["quat"][i]))))))))
        body = bpy.data.objects[a["name"] + "_model"]
        mats = [s.material.name for s in body.material_slots if s.material and s.link == "OBJECT"]
        rep["agents"][a["name"]] = dict(role=a["role"], team_material=mats[0] if mats else None,
                                        max_pos_err_m=float(max(err)), max_rot_err_deg=float(max(rot_err)) if rot_err else None,
                                        visibility_mismatches=vis_err, orientation=a["quat_source"])
        worst = max(worst, max(err))
        worst_rot = max([worst_rot] + rot_err)
    rgb = {r: tuple(round(x, 2) for x in v) for r, v in scene_build.ROLE_RGB.items()}
    rep["checks"] = dict(
        max_pos_err_m=worst, max_rot_err_deg=worst_rot,
        all_attackers_red=all(v["team_material"] == "viz_team_attacker" for v in rep["agents"].values() if v["role"] == "attacker"),
        all_defenders_green=all(v["team_material"] == "viz_team_defender" for v in rep["agents"].values() if v["role"] == "defender"),
        role_rgb=rgb, frame_of_last_sample=float(B.frames[-1]), frame_end=B.frame_end,
        sim_seconds=(S["T"] - 1) * S["dt"], video_seconds=(B.frames[-1] - 1) / B.fps,
        visibility_ok=all(v["visibility_mismatches"] == 0 for v in rep["agents"].values()),
        regions=[dict(kind=r["kind"], center=np.asarray(r["center"]).round(3).tolist(), radius=r["radius"]) for r in S["regions"]],
        events={k: sum(e["kind"] == k for e in S["events"]) for k in {e["kind"] for e in S["events"]}},
    )
    rep["pass"] = bool(worst < 1e-4 and worst_rot < 0.1 and rep["checks"]["visibility_ok"]
                       and rep["checks"]["all_attackers_red"] and rep["checks"]["all_defenders_green"])
    with open(out_json, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[validate] pass={rep['pass']} max_pos_err={worst:.2e} m max_rot_err={worst_rot:.3f} deg -> {out_json}", flush=True)
    sc.frame_set(1)
    return rep


def main():
    args = parse()
    cfg = json.load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    npz = args.npz or cfg["npz"]
    npz = npz if os.path.isabs(npz) else os.path.join(REPO, npz)
    name = cfg.get("name", os.path.splitext(os.path.basename(npz))[0])
    out = args.out or cfg.get("out_dir", "docs/results/blender_viz")
    out = out if os.path.isabs(out) else os.path.join(REPO, out)
    os.makedirs(out, exist_ok=True)
    q = dict(QUALITY[args.quality])
    res = tuple(args.res or q["res"])

    t0 = time.time()
    S = viz_data.load(npz, env=int(cfg.get("env", 0)), first_episode=bool(cfg.get("first_episode", True)))
    print(viz_data.summary(S), flush=True)
    scene_build.clear()
    for ob in list(bpy.data.objects):  # factory startup cube / light / camera
        if ob.name in ("Cube", "Light", "Camera") and not ob.users_collection[0].name.startswith(scene_build.ROOT):
            bpy.data.objects.remove(ob, do_unlink=True)
    scene_cfg = {k: v for k, v in cfg.items() if k in scene_build.DEFAULTS}
    B = scene_build.Builder(S, scene_cfg).build(res)
    setup_render(q, args)
    print(f"[build] {time.time() - t0:.1f} s, frames 1..{B.frame_end} at {B.fps} fps (speed x{B.speed})", flush=True)

    if args.validate:
        validate(B, S, os.path.join(out, f"{name}_validate.json"))
    cams = list(B.cams) if args.camera == "all" else [c.strip() for c in args.camera.split(",")]
    select_camera(B, cams[0])
    if args.save_blend:
        path = os.path.join(out, f"{name}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
        print(f"[blend] {path}", flush=True)
    for cam in cams:
        select_camera(B, cam)
        if args.still is not None:
            for k, frac in enumerate(args.still or [0.6]):
                f = 1 + round(frac * (B.frames[-1] - 1))
                suffix = "" if len(args.still or [0.6]) == 1 else f"_{k}"
                path = os.path.join(out, f"{name}_{cam}{suffix}.png")
                t1 = time.time()
                render_still(path, f)
                print(f"[still] {path} frame {f} ({time.time() - t1:.1f} s)", flush=True)
        if args.anim:
            f0, f1 = args.frames or (1, B.frame_end)
            path = os.path.join(out, f"{name}_{cam}.mp4")
            t1 = time.time()
            render_anim(path, f0, f1)
            print(f"[anim] {path} frames {f0}..{f1} ({time.time() - t1:.1f} s)", flush=True)
    select_camera(B, cams[0])
    print(f"[done] {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
