#!/usr/bin/env python3
"""Show dumps from spline_kernel_eval, the vamp spline tests and examples, in viser.

    python3 tests/view_spline.py spline.json                    # one dump, then open http://localhost:8080
    python3 tests/view_spline.py runs/                          # every *.json in a directory
    python3 tests/view_spline.py "spline_mbm_planner_*.json"    # a glob (quote it), e.g. one dump per MBM problem

With several dumps there is a "Problem" dropdown: selecting one redraws its curve, environment and path, and the
step slider takes that dump's number of steps. If a dump has a "stats" object (the MBM planner example writes one),
its fields are shown in text boxes: the planning time in milliseconds, whether the planner solved, and so on.

Drawn:
  curve             the spline, coloured by t (blue at t = 0, red at t = 1)
  waypoints         the points the spline was fitted through (white)
  control points    where the curve is pulled (green), with the control polygon; most are NOT on the curve
  knots             where one cubic piece hands over to the next (yellow diamonds)
  obstacles         the environment, if the dump has one (boxes, cylinders, spheres), and a reference path (grey)
  robot             the Panda (meshes from --urdf) at the selected step
  end-effector      orange sphere; the curve point P(t) it is being matched to is the green sphere,
                    with the error between them in red and the tangent dP/dt as a short green line
  trails            where the end-effector and P(t) have been over the steps so far
  segment           for a curve of segments (dumps whose steps have "segment", its two end points, with "segment_index"
                    and "s", the fraction along it): the segment t is on, highlighted in yellow
  frames            for a pose curve (dumps whose steps have "eef_quaternion" and "target_quaternion", w x y z): the
                    end-effector's frame and the curve's frame at P(t), with the rotation and position errors

The slider steps through the projection or the planned path, so both the arm and t move.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import trimesh
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

        if not all(key in d for key in ("waypoints", "control_points", "knots", "curve", "iterations")):
            print(f"skipping {path}: not a spline dump")
            continue

        d["_file"] = str(path)
        dumps.append(d)

    return dumps


def describe(index, d):
    """The dropdown entry of a dump."""
    name = d.get("label") or Path(d["_file"]).stem
    status = d.get("stats", {}).get("status")
    suffix = f"  [{status}]" if status and status != "solved" else ""
    return f"{index:03d}  {name}{suffix}"


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
        raise SystemExit("no spline dumps found")

    solved = sum(1 for d in dumps if d.get("stats", {}).get("status", "solved") == "solved")
    print(f"{len(dumps)} dump(s), {solved} solved")

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
    labels = [describe(i, d) for i, d in enumerate(dumps)]
    problem = gui.add_dropdown("Problem", options=tuple(labels), initial_value=labels[0])
    slider = gui.add_slider("Step", min=0, max=1, step=1, initial_value=0)
    play = gui.add_checkbox("Play", False)
    speed = gui.add_slider("Steps per second", min=1, max=20, step=1, initial_value=4)

    info_t = gui.add_text("t", initial_value="", disabled=True)
    info_error = gui.add_text("|FK(q) - P(t)|", initial_value="", disabled=True)
    info_rotation = gui.add_text("rotation / position error", initial_value="", disabled=True)
    info_segment = gui.add_text("segment, s", initial_value="", disabled=True)
    info_q = gui.add_text("q", initial_value="", disabled=True)

    with gui.add_folder("Problem"):
        stat_boxes = {
            "status": gui.add_text("status", initial_value="", disabled=True),
            "planning_ms": gui.add_text("planning time (ms)", initial_value="", disabled=True),
            "reference_ms": gui.add_text("reference RRTC time (ms)", initial_value="", disabled=True),
            "iterations": gui.add_text("iterations", initial_value="", disabled=True),
            "path": gui.add_text("path (states, joint length)", initial_value="", disabled=True),
            "deviation_mm": gui.add_text("reference vs spline (mm)", initial_value="", disabled=True),
            "worst_mm": gui.add_text("worst distance to curve (mm)", initial_value="", disabled=True),
        }
        run_label = gui.add_text("run", initial_value="", disabled=True)
        compare = gui.add_text("traced vs non-traced", initial_value="", disabled=True)

    layers = {
        "Waypoints (white)": [],
        "Control points (green)": [],
        "Control polygon": [],
        "Knots (yellow)": [],
    }
    boxes = {}
    with gui.add_folder("Show"):
        for name in layers:
            box = gui.add_checkbox(name, True)

            def on_toggle(_, name=name, box=box):
                for handle in layers[name]:
                    handle.visible = box.value

            box.on_update(on_toggle)
            boxes[name] = box

    # ---- the scene of the loaded dump ----
    nodes = []  # scene handles of the loaded dump, removed when another one is loaded
    current = {"iterations": [], "ee_trail": [], "curve_trail": []}
    frames = []  # the end-effector's and the curve's frames of a pose curve
    segment_handles = []  # the highlighted segment t is on

    def keep(handle, layer=None):
        nodes.append(handle)
        if layer is not None:
            layers[layer].append(handle)
        return handle

    def format_stat(key, stats):
        if key not in stats:
            return ""
        value = stats[key]
        if key in ("planning_ms", "reference_ms"):
            return f"{value:.1f}"
        if key == "path":
            return f"{value['states']}, {value['joint_length']:.3f} rad"
        if key in ("deviation_mm", "worst_mm"):
            return f"{value:.2f}"
        return str(value)

    def load(k):
        d = dumps[k]
        for handle in nodes:
            handle.remove()

        nodes.clear()
        for handles in layers.values():
            handles.clear()

        for handle in frames + segment_handles:
            handle.remove()

        frames.clear()
        segment_handles.clear()

        waypoints = np.array(d["waypoints"])
        control_points = np.array(d["control_points"])
        curve = np.array(d["curve"])
        curve_t = np.array(d["curve_t"])
        knot_points = np.array(d["knot_points"])

        # the environment, and a reference path (end-effector positions) to compare with
        obstacles = d.get("obstacles", {})
        for i, box in enumerate(obstacles.get("boxes", [])):
            x, y, z, w = box["quaternion_xyzw"]
            keep(
                scene.add_box(
                    f"/obstacles/box_{i}",
                    color=(170, 150, 120),
                    dimensions=tuple(2.0 * np.asarray(box["half_extents"])),
                    position=tuple(box["position"]),
                    wxyz=(w, x, y, z),
                    opacity=0.6,
                )
            )

        for i, cylinder in enumerate(obstacles.get("cylinders", [])):
            x, y, z, w = cylinder["quaternion_xyzw"]
            # trimesh's cylinder has its axis along z, which is the axis of the MBM cylinders' frame
            mesh = trimesh.creation.cylinder(radius=cylinder["radius"], height=cylinder["length"])
            mesh.visual.face_colors = (200, 90, 90, 255)
            keep(
                scene.add_mesh_trimesh(
                    f"/obstacles/cylinder_{i}", mesh, wxyz=(w, x, y, z), position=tuple(cylinder["position"])
                )
            )

        for i, sphere in enumerate(obstacles.get("spheres", [])):
            keep(
                scene.add_icosphere(
                    f"/obstacles/sphere_{i}", radius=sphere["radius"], color=(200, 90, 90), position=tuple(sphere["position"])
                )
            )

        if "reference_path" in d:
            keep(
                scene.add_line_segments(
                    "/reference_path",
                    points=segments(np.array(d["reference_path"])),
                    colors=(180, 180, 180),
                    thickness=0.004,
                )
            )

        # the spline, the points it was built from, and its pieces
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
                "/waypoints", points=waypoints, colors=(255, 255, 255), point_size=0.014, point_shape="circle"
            ),
            "Waypoints (white)",
        )
        keep(
            scene.add_point_cloud(
                "/control_points", points=control_points, colors=(80, 200, 120), point_size=0.014, point_shape="square"
            ),
            "Control points (green)",
        )
        keep(
            scene.add_line_segments(
                "/control_polygon", points=segments(control_points), colors=(80, 200, 120), thickness=0.002
            ),
            "Control polygon",
        )
        keep(
            scene.add_point_cloud(
                "/knots", points=knot_points, colors=(250, 220, 60), point_size=0.018, point_shape="diamond"
            ),
            "Knots (yellow)",
        )

        for name, handles in layers.items():
            for handle in handles:
                handle.visible = boxes[name].value

        # the steps of this dump
        current["iterations"] = d["iterations"]
        current["ee_trail"] = [np.array(it["arm"][-1]) for it in d["iterations"]]
        current["curve_trail"] = [np.array(it["curve_point"]) for it in d["iterations"]]

        # the text boxes
        stats = d.get("stats", {})
        for key, box in stat_boxes.items():
            box.value = format_stat(key, stats)

        run_label.value = str(d.get("label", ""))
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

        step = min(step, len(iterations) - 1)
        it = iterations[step]
        arm = np.array(it["arm"])
        point = np.array(it["curve_point"])
        tangent = np.array(it["tangent"])
        ee = arm[-1]

        cfg = np.zeros(urdf.num_actuated_joints)
        q = np.array(it["q"])
        cfg[: len(q)] = q  # any extra actuated joints (a gripper) stay at zero
        robot.update_cfg(cfg)

        scene.add_icosphere("/ee", radius=0.02, color=(255, 140, 0), position=tuple(ee))
        scene.add_icosphere("/curve_point", radius=0.016, color=(60, 220, 90), position=tuple(point))
        scene.add_line_segments("/error", points=np.stack([ee, point])[None], colors=(255, 50, 50), thickness=0.006)

        length = np.linalg.norm(tangent)
        direction = tangent / length if length > 0 else tangent
        scene.add_line_segments(
            "/tangent", points=np.stack([point, point + 0.15 * direction])[None], colors=(60, 220, 90), thickness=0.006
        )

        # trails up to this step; a line needs two points, so at step 0 it is a zero-length segment
        ee_points = current["ee_trail"][: step + 1]
        curve_points = current["curve_trail"][: step + 1]
        if len(ee_points) < 2:
            ee_points, curve_points = [ee, ee], [point, point]

        scene.add_line_segments("/ee_trail", points=segments(ee_points), colors=(255, 140, 0), thickness=0.004)
        scene.add_line_segments("/curve_trail", points=segments(curve_points), colors=(60, 220, 90), thickness=0.004)

        if "eef_quaternion" in it and "target_quaternion" in it:
            # a pose curve: the frames of the end-effector and of the curve's pose at t
            for handle in frames:
                handle.remove()

            frames.clear()
            for name, quaternion, position in (
                ("/ee_frame", it["eef_quaternion"], ee),
                ("/target_frame", it["target_quaternion"], point),
            ):
                frames.append(
                    scene.add_frame(
                        name, wxyz=tuple(quaternion), position=tuple(position), axes_length=0.09, axes_radius=0.004
                    )
                )

            info_rotation.value = (
                f"{np.degrees(it['rotation_error']):.3f} deg, {1000.0 * it['position_error']:.3f} mm"
            )
        else:
            info_rotation.value = ""

        for handle in segment_handles:
            handle.remove()

        segment_handles.clear()
        if "segment" in it:
            # the segment of the curve that t is on
            segment_handles.append(
                scene.add_line_segments(
                    "/segment", points=np.array(it["segment"], dtype=float)[None], colors=(255, 230, 0), thickness=0.018
                )
            )
            info_segment.value = f"segment {it['segment_index']}, s = {it['s']:.4f}"
        else:
            info_segment.value = ""

        info_t.value = f"{it['t']:.5f}"
        info_error.value = f"{it['error_norm']:.6f}"
        info_q.value = ", ".join(f"{x:.3f}" for x in it["q"])

    problem.on_update(lambda _: load(labels.index(problem.value)))
    slider.on_update(lambda _: show(int(slider.value)))
    load(0)

    if args.once:
        for k in range(len(dumps)):
            load(k)
            show(len(current["iterations"]) - 1)  # also exercise the trails

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
