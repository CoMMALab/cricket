#!/usr/bin/env python3
"""Show dumps from lerp_kernel_eval in viser: a straight-line (lerp) curve through 20 keyframes, and a projection onto
the point P(t) of it with t held fixed.

    python3 tests/view_lerp.py lerp_r3.json                       # one dump, then open http://localhost:8080
    python3 tests/view_lerp.py lerp_r3.json lerp_so3r3.json       # several: a "Problem" dropdown picks one

Made with  lerp_kernel_eval curve=r3 dump=lerp_r3.json  or  curve=so3r3 dump=lerp_so3r3.json.

Drawn:
  curve             the polyline through the keyframes, coloured by t (blue at t = 0, red at t = 1)
  keyframes         the points the curve is made of (white), with their numbers; for an so3r3 curve a frame at each, the
                    rotation the slerp starts or ends a segment at
  segment           the segment t is on, highlighted (yellow); the kernel picks it from t by conditionals
  robot             the Panda (meshes from --urdf) at the selected iteration
  end-effector      orange sphere; the curve point P(t) it is being matched to is the green sphere, with the error between
                    them in red. For so3r3 a frame at each shows the end-effector's rotation and the slerped target
                    rotation, so the rotation error can be seen
  trail             where the end-effector has been over the iterations so far

The slider steps through the projection iterations. t does not move (it is held fixed), so P(t) and the segment stay; the
arm walks to them.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf


def t_colors(t):
    """Blue (t = 0) to red (t = 1), uint8 RGB."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return (np.stack([t, 0.25 + 0.0 * t, 1.0 - t], axis=-1) * 255).astype(np.uint8)


def segments(points):
    """(n, 3) polyline -> (n - 1, 2, 3) segments."""
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def collect(arguments):
    """The dump files named by the arguments: files, directories (their *.json) and globs."""
    paths = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif any(c in argument for c in "*?["):
            paths.extend(Path(p) for p in sorted(glob.glob(argument)))
        else:
            paths.append(path)

    return paths


def load_dumps(paths):
    dumps = []
    for path in paths:
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as error:
            print(f"skipping {path}: {error}")
            continue

        if not all(key in d for key in ("keyframes", "taus", "curve", "iterations")):
            print(f"skipping {path}: not a lerp dump")
            continue

        d["_file"] = str(path)
        dumps.append(d)

    return dumps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dumps", nargs="+", help="dump files, directories of them, or globs")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "resources/panda/panda.urdf",
        help="URDF (with meshes) of the robot the dumps were made for; its actuated joints must match q",
    )
    parser.add_argument("--once", action="store_true", help="load every dump once and exit (for testing)")
    args = parser.parse_args()

    dumps = load_dumps(collect(args.dumps))
    if not dumps:
        raise SystemExit("no lerp dumps found")

    print(f"{len(dumps)} dump(s)")

    server = viser.ViserServer(port=args.port)
    scene = server.scene

    # The robot itself, from the URDF with meshes. update_cfg() moves it to each step.
    urdf_dir = args.urdf.resolve().parent

    def resolve_mesh(fname):
        # the URDFs here say package://meshes/visual/x.obj: the meshes directory next to the URDF file
        # (yourdfpy would look for it relative to the current directory)
        return str(urdf_dir / fname.removeprefix("package://"))

    urdf = yourdfpy.URDF.load(str(args.urdf.resolve()), filename_handler=resolve_mesh)
    robot = ViserUrdf(
        server, urdf, load_meshes=True, load_collision_meshes=False, root_node_name="/robot", mesh_color_override=(0.55, 0.65, 0.9)
    )
    scene.add_grid("/grid", width=2.0, height=2.0, plane="xy", position=(0.0, 0.0, 0.0), cell_size=0.1)
    scene.add_frame("/world", axes_length=0.2, axes_radius=0.004)

    # ---- the GUI (the fields are filled by load() and show()) ----
    gui = server.gui
    labels = [f"{i:03d}  {d.get('label') or Path(d['_file']).stem}" for i, d in enumerate(dumps)]
    problem = gui.add_dropdown("Problem", options=tuple(labels), initial_value=labels[0])
    slider = gui.add_slider("Step", min=0, max=1, step=1, initial_value=0)
    play = gui.add_checkbox("Play", False)
    speed = gui.add_slider("Steps per second", min=1, max=20, step=1, initial_value=4)

    info_t = gui.add_text("t (held fixed)", initial_value="", disabled=True)
    info_segment = gui.add_text("segment, s along it", initial_value="", disabled=True)
    info_error = gui.add_text("|err|", initial_value="", disabled=True)
    info_position = gui.add_text("position error (m)", initial_value="", disabled=True)
    info_rotation = gui.add_text("rotation error (weighted)", initial_value="", disabled=True)
    info_q = gui.add_text("q", initial_value="", disabled=True)
    compare = gui.add_text("traced vs non-traced", initial_value="", disabled=True)

    layers = {
        "Keyframes (white)": [],
        "Keyframe numbers": [],
        "Keyframe frames (so3r3)": [],
    }
    boxes = {}
    with gui.add_folder("Show"):
        for name in layers:
            box = gui.add_checkbox(name, name != "Keyframe numbers")

            def on_toggle(_, name=name, box=box):
                for handle in layers[name]:
                    handle.visible = box.value

            box.on_update(on_toggle)
            boxes[name] = box

    # ---- the scene of the loaded dump ----
    nodes = []  # scene handles of the loaded dump, removed when another one is loaded
    current = {"dump": None, "iterations": [], "ee_trail": []}

    def keep(handle, layer=None):
        nodes.append(handle)
        if layer is not None:
            layers[layer].append(handle)
        return handle

    def load(k):
        d = dumps[k]
        for handle in nodes:
            handle.remove()

        nodes.clear()
        for handles in layers.values():
            handles.clear()

        keyframes = np.array(d["keyframes"])
        curve = np.array(d["curve"])
        curve_t = np.array(d["curve_t"])
        pose = d["kind"] == "so3r3"

        keep(
            scene.add_line_segments(
                "/curve",
                points=segments(curve),
                colors=np.repeat(t_colors(curve_t[:-1])[:, None, :], 2, axis=1),
                thickness=0.008,
            )
        )
        keep(
            scene.add_point_cloud(
                "/keyframes", points=keyframes, colors=(255, 255, 255), point_size=0.014, point_shape="circle"
            ),
            "Keyframes (white)",
        )
        for i, point in enumerate(keyframes):
            keep(scene.add_label(f"/keyframe_numbers/{i}", text=f"{i}", position=tuple(point + np.array([0.0, 0.0, 0.02]))), "Keyframe numbers")

        if pose:
            for i, (point, quaternion) in enumerate(zip(keyframes, d["keyframe_quaternions"])):
                keep(
                    scene.add_frame(
                        f"/keyframe_frames/{i}",
                        wxyz=tuple(quaternion),
                        position=tuple(point),
                        axes_length=0.05,
                        axes_radius=0.002,
                    ),
                    "Keyframe frames (so3r3)",
                )

        for name, handles in layers.items():
            for handle in handles:
                handle.visible = boxes[name].value

        # the segment t is on (it is the same at every step: t is held fixed)
        segment = d["iterations"][0]["segment"]
        keep(
            scene.add_line_segments(
                "/segment",
                points=np.stack([keyframes[segment], keyframes[segment + 1]])[None],
                colors=(250, 220, 60),
                thickness=0.016,
            )
        )

        current["dump"] = d
        current["iterations"] = d["iterations"]
        current["ee_trail"] = [np.array(it["arm"][-1]) for it in d["iterations"]]

        if "reference_output" in d and "traced_output" in d:
            difference = np.max(np.abs(np.array(d["reference_output"]) - np.array(d["traced_output"])))
            compare.value = f"max |difference| = {difference:.2e}"
        else:
            compare.value = ""

        slider.value = 0
        slider.max = max(len(d["iterations"]) - 1, 1)
        show(0)

    def show(step):
        iterations = current["iterations"]
        if not iterations:
            return

        d = current["dump"]
        step = min(step, len(iterations) - 1)
        it = iterations[step]
        arm = np.array(it["arm"])
        point = np.array(it["curve_point"])
        ee = arm[-1]

        cfg = np.zeros(urdf.num_actuated_joints)
        q = np.array(it["q"])
        cfg[: len(q)] = q  # any extra actuated joints (a gripper) stay at zero
        robot.update_cfg(cfg)

        scene.add_icosphere("/ee", radius=0.02, color=(255, 140, 0), position=tuple(ee))
        scene.add_icosphere("/curve_point", radius=0.016, color=(60, 220, 90), position=tuple(point))
        scene.add_line_segments("/error", points=np.stack([ee, point])[None], colors=(255, 50, 50), thickness=0.006)

        if d["kind"] == "so3r3":
            scene.add_frame("/ee_frame", wxyz=tuple(it["eef_quaternion"]), position=tuple(ee), axes_length=0.12, axes_radius=0.004)
            scene.add_frame("/target_frame", wxyz=tuple(it["curve_quaternion"]), position=tuple(point), axes_length=0.12, axes_radius=0.004, origin_radius=0.0)

        # the trail up to this step; a line needs two points, so at step 0 it is a zero-length segment
        ee_points = current["ee_trail"][: step + 1]
        if len(ee_points) < 2:
            ee_points = [ee, ee]

        scene.add_line_segments("/ee_trail", points=segments(ee_points), colors=(255, 140, 0), thickness=0.004)

        info_t.value = f"{it['t']:.5f}"
        info_segment.value = f"{it['segment']} (keyframes {it['segment']} to {it['segment'] + 1}), s = {it['s']:.4f}"
        info_error.value = f"{it['error_norm']:.3e}"
        info_position.value = f"{it['position_error']:.3e}"
        info_rotation.value = f"{it['rotation_error']:.3e}" if "rotation_error" in it else "-"
        info_q.value = ", ".join(f"{x:.3f}" for x in it["q"])

    problem.on_update(lambda _: load(labels.index(problem.value)))
    slider.on_update(lambda _: show(int(slider.value)))
    load(0)

    if args.once:
        for k in range(len(dumps)):
            load(k)
            show(len(current["iterations"]) - 1)  # also exercise the trail

        server.stop()
        return

    print(f"viser at http://localhost:{args.port}")
    last = time.time()
    while True:
        time.sleep(0.02)
        count = len(current["iterations"])
        if play.value and count > 0 and time.time() - last > 1.0 / speed.value:
            last = time.time()
            slider.value = (int(slider.value) + 1) % count


if __name__ == "__main__":
    main()
