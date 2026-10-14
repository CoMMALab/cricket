#!/usr/bin/env python3
"""Show a dump from spline_kernel_eval in viser.

    spline_kernel_eval dump=spline.json t=0.4 iterations=15
    python3 tests/view_spline.py spline.json            # then open http://localhost:8080

Drawn:
  curve             the spline, coloured by t (blue at t = 0, red at t = 1)
  waypoints         the points the spline was fitted through (white)
  control points    where the curve is pulled (green), with the control polygon; most are NOT on the curve
  knots             where one cubic piece hands over to the next (yellow diamonds)
  robot             the Panda (meshes from --urdf) at the selected projection step
  end-effector      orange sphere; the curve point P(t) it is being projected onto is the green sphere,
                    with the error between them in red and the tangent dP/dt as a short green line
  trails            where the end-effector and P(t) have been over the projection steps so far

The slider steps through the projection: Gauss-Newton on (q, t), so both the arm and t move.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="JSON written by spline_kernel_eval dump=...")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "resources/panda/panda.urdf",
        help="URDF (with meshes) of the robot the dump was made for; its actuated joints must match q",
    )
    parser.add_argument("--once", action="store_true", help="build the scene once and exit (for testing)")
    args = parser.parse_args()

    with open(args.dump) as f:
        d = json.load(f)

    waypoints = np.array(d["waypoints"])
    control_points = np.array(d["control_points"])
    knots = np.array(d["knots"])
    curve = np.array(d["curve"])
    curve_t = np.array(d["curve_t"])
    knot_points = np.array(d["knot_points"])
    iterations = d["iterations"]
    n_steps = len(iterations)

    reference = np.array(d["reference_output"])
    traced = np.array(d["traced_output"])

    server = viser.ViserServer(port=args.port)
    scene = server.scene

    # The robot itself, from the URDF with meshes. update_cfg() moves it to each projection step.
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

    # The spline, the points it was built from, and its pieces
    scene.add_line_segments(
        "/curve", points=segments(curve), colors=np.repeat(t_colors(curve_t[:-1])[:, None, :], 2, axis=1), thickness=0.008
    )
    waypoint_cloud = scene.add_point_cloud(
        "/waypoints", points=waypoints, colors=(255, 255, 255), point_size=0.014, point_shape="circle"
    )
    control_cloud = scene.add_point_cloud(
        "/control_points", points=control_points, colors=(80, 200, 120), point_size=0.014, point_shape="square"
    )
    control_polygon = scene.add_line_segments(
        "/control_polygon", points=segments(control_points), colors=(80, 200, 120), thickness=0.002
    )
    knot_cloud = scene.add_point_cloud(
        "/knots", points=knot_points, colors=(250, 220, 60), point_size=0.018, point_shape="diamond"
    )

    # Per-step objects, replaced when the slider moves
    ee_trail = [np.array(it["arm"][-1]) for it in iterations]
    curve_trail = [np.array(it["curve_point"]) for it in iterations]

    def show(step):
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
        scene.add_line_segments(
            "/error", points=np.stack([ee, point])[None], colors=(255, 50, 50), thickness=0.006
        )

        length = np.linalg.norm(tangent)
        direction = tangent / length if length > 0 else tangent
        scene.add_line_segments(
            "/tangent", points=np.stack([point, point + 0.15 * direction])[None], colors=(60, 220, 90), thickness=0.006
        )

        if step > 0:
            scene.add_line_segments("/ee_trail", points=segments(ee_trail[: step + 1]), colors=(255, 140, 0), thickness=0.004)
            scene.add_line_segments(
                "/curve_trail", points=segments(curve_trail[: step + 1]), colors=(60, 220, 90), thickness=0.004
            )

        info_t.value = f"{it['t']:.5f}"
        info_error.value = f"{it['error_norm']:.6f}"
        info_q.value = ", ".join(f"{x:.3f}" for x in it["q"])

    gui = server.gui
    slider = gui.add_slider("Projection step", min=0, max=max(n_steps - 1, 1), step=1, initial_value=0)
    play = gui.add_checkbox("Play", False)
    speed = gui.add_slider("Steps per second", min=1, max=20, step=1, initial_value=4)
    info_t = gui.add_text("t", initial_value="", disabled=True)
    info_error = gui.add_text("|FK(q) - P(t)|", initial_value="", disabled=True)
    info_q = gui.add_text("q", initial_value="", disabled=True)
    gui.add_text(
        "traced vs non-traced",
        initial_value=f"max |difference| = {np.max(np.abs(reference - traced)):.2e}",
        disabled=True,
    )

    with gui.add_folder("Show"):
        for label, handle in [
            ("Waypoints (white)", waypoint_cloud),
            ("Control points (green)", control_cloud),
            ("Control polygon", control_polygon),
            ("Knots (yellow)", knot_cloud),
        ]:
            box = gui.add_checkbox(label, True)

            def on_toggle(_, box=box, handle=handle):
                handle.visible = box.value

            box.on_update(on_toggle)

    slider.on_update(lambda _: show(int(slider.value)))
    show(0)

    print(f"{n_steps} projection steps; knots: {np.round(knots, 3).tolist()}")
    if args.once:
        show(n_steps - 1)  # also exercise the trails
        server.stop()
        return

    print(f"viser at http://localhost:{args.port}")
    last = time.time()
    while True:
        time.sleep(0.02)
        if play.value and time.time() - last > 1.0 / speed.value:
            last = time.time()
            slider.value = (int(slider.value) + 1) % n_steps


if __name__ == "__main__":
    main()
