"""Inspect and illustrate the actual robot model, without running a controller.

Joint sweeps below use MuJoCo forward kinematics with a fixed root pose. They
illustrate geometry only; a displayed configuration need not be balanced.
"""

import base64
import html
import io
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

COLORS = {"Left": "#25847a", "Right": "#d78535", "Head": "#6376ad", "Neck": "#6376ad"}


def lesson_font(size, bold=False):
    """Use Matplotlib's bundled font, independent of Colab's system fonts."""
    from matplotlib import get_data_path
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf" / name), size)


def model_state(env):
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
    mujoco.mj_forward(model, data)
    return model, data


def joint_inventory(env):
    """Return names, local axes, limits, and nearest articulated parent from XML."""
    model, _ = model_state(env)
    joints = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(joint).name.split("/")[-1]
        body = model.jnt_bodyid[joint]
        ancestor = model.body_parentid[body]
        parent_name = "Floating base"
        while ancestor:
            if model.body_jntnum[ancestor]:
                parent_joint = model.body_jntadr[ancestor]
                if model.jnt_type[parent_joint] != mujoco.mjtJoint.mjJNT_FREE:
                    parent_name = model.joint(parent_joint).name.split("/")[-1]
                break
            ancestor = model.body_parentid[ancestor]
        joints.append(
            {
                "name": name,
                "parent": parent_name,
                "axis_local": model.jnt_axis[joint].tolist(),
                "limits_deg": np.rad2deg(model.jnt_range[joint]).round(1).tolist(),
                "qpos_address": int(model.jnt_qposadr[joint]),
            }
        )
    return joints


def print_joint_table(env):
    from prettytable import PrettyTable

    table = PrettyTable(["Joint", "Parent", "Local axis", "Limits (degrees)"])
    for row in joint_inventory(env):
        table.add_row(
            [row["name"], row["parent"], row["axis_local"], row["limits_deg"]]
        )
    print(table)


def save_tree(env, destination):
    """Expand the inspected left leg and collapse the other measured branches."""
    rows = joint_inventory(env)
    left = [row for row in rows if row["name"].startswith("Left_")]
    right_count = sum(row["name"].startswith("Right_") for row in rows)
    head_count = len(rows) - len(left) - right_count
    coordinates = {"Floating base": (290, 58)}
    for index, row in enumerate(left):
        coordinates[row["name"]] = (180, 155 + index * 79)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="660" viewBox="0 0 600 660">',
        '<rect width="600" height="660" rx="16" fill="#faf7f0"/>',
    ]
    for row in left:
        x1, y1 = coordinates[row["parent"]]
        x2, y2 = coordinates[row["name"]]
        parts.append(
            f'<path d="M{x1},{y1 + 24} V{y2 - 35} H{x2} V{y2 - 24}" fill="none" stroke="#a6b7ad" stroke-width="3"/>'
        )
    for index, (name, (x, y)) in enumerate(coordinates.items()):
        color = (
            "#263b3b"
            if not index
            else ["#006c67", "#2355b9", "#a13e8e", "#cf6517", "#b0303e"][index - 1]
        )
        label = name if not index else f"{index}  {name}"
        parts.append(
            f'<rect x="{x - 145}" y="{y - 24}" width="290" height="48" rx="10" fill="{color}"/><text x="{x}" y="{y + 6}" text-anchor="middle" fill="white" font-family="sans-serif" font-size="19">{label}</text>'
        )
    for y, label, color in [
        (170, f"Right leg · {right_count} joints", "#a8753a"),
        (285, f"Head/neck · {head_count} joints", "#6376ad"),
    ]:
        branch_path = (
            f"M435,58 H476 V{y - 22}" if y == 170 else f"M435,58 H595 V{y + 11} H589"
        )
        parts.append(
            f'<path d="{branch_path}" fill="none" stroke="#a6b7ad" stroke-width="3"/><rect x="363" y="{y - 22}" width="226" height="66" rx="10" fill="{color}"/><text x="476" y="{y + 3}" text-anchor="middle" fill="white" font-family="sans-serif" font-size="18">{label}</text><text x="476" y="{y + 27}" text-anchor="middle" fill="white" font-family="sans-serif" font-size="14">branch collapsed</text>'
        )
    x, y = coordinates[left[-1]["name"]]
    parts.append(
        f'<path d="M{x},{y + 24} V{y + 65}" stroke="#a6b7ad" stroke-width="3"/><rect x="55" y="{y + 65}" width="250" height="40" rx="10" fill="#718c83"/><text x="180" y="{y + 91}" text-anchor="middle" fill="white" font-family="sans-serif" font-size="17">left_foot · measurement site</text>'
    )
    parts.append(
        '<text x="300" y="620" text-anchor="middle" fill="#516766" font-family="sans-serif" font-size="16">Connections read from the model. Fixed links collapsed.</text></svg>'
    )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("".join(parts))


def save_pose(env, destination):
    """Save the current full simulation view for the environment walkthrough."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(destination)


def sweep_joint(env, joint_name, angles_deg, destination):
    """Write a self-contained slider with genuine rendered FK configurations.

    The other joint coordinates and floating root stay fixed. The graph reports
    selected-leg foot position in the torso frame, never an asserted safe workspace.
    """
    model, data = model_state(env)
    row = next(row for row in joint_inventory(env) if row["name"] == joint_name)
    side = joint_name.split("_", 1)[0]
    if side not in {"Left", "Right"}:
        raise ValueError(
            "Choose a left- or right-leg joint for this foot-position study."
        )
    angles = np.asarray(angles_deg, dtype=float)
    low, high = row["limits_deg"]
    if (
        not angles.size
        or not np.all(np.isfinite(angles))
        or angles.min() < low
        or angles.max() > high
    ):
        raise ValueError(f"Requested angles exceed {joint_name}'s model limits.")
    site_name = side.lower() + "_foot"
    site_id = next(
        i for i in range(model.nsite) if model.site(i).name.endswith(site_name)
    )
    root_joint = next(
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    root_body = int(model.jnt_bodyid[root_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[root_body] + np.array([0, 0, 0.04])
    camera.distance = 1.10
    camera.azimuth = 65 if side == "Right" else 245
    camera.elevation = -12
    joint_id = model.joint("robot/" + joint_name).id
    hip_id = model.joint("robot/" + side + "_Hip_Yaw").id
    other = "Left" if side == "Right" else "Right"
    other_hip = model.joint("robot/" + other + "_Hip_Yaw").id
    frames, positions, projected_joints = [], [], []
    from PIL import ImageDraw

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for angle in angles:
            data.qpos[row["qpos_address"]] = np.deg2rad(angle)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            eye = renderer.scene.camera
            origin = (eye[0].pos + eye[1].pos) / 2
            forward = eye[0].forward / np.linalg.norm(eye[0].forward)
            up = eye[0].up / np.linalg.norm(eye[0].up)
            right = np.cross(forward, up)
            near_depth = float((data.xanchor[hip_id] - origin) @ forward)
            far_depth = float((data.xanchor[other_hip] - origin) @ forward)
            if near_depth >= far_depth:
                raise RuntimeError("The moving leg must be on the camera-facing side.")
            delta = data.xanchor[joint_id] - origin
            scale = (
                float(eye[0].frustum_near)
                * 480
                / (
                    float(eye[0].frustum_top - eye[0].frustum_bottom)
                    * float(delta @ forward)
                )
            )
            x, y = 320 + float(delta @ right) * scale, 240 - float(delta @ up) * scale
            projected_joints.append([x, y])
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="white", width=5)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#d78535", width=3)
            draw.rounded_rectangle((12, 12, 365, 43), radius=7, fill="#faf7f0")
            draw.text(
                (23, 18),
                f"{joint_name}  |  {angle:+.0f}°  |  visible leg",
                font=lesson_font(16),
                fill="#263b3b",
            )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=83)
            frames.append(base64.b64encode(buffer.getvalue()).decode())
            rotation = data.xmat[root_body].reshape(3, 3)
            positions.append(
                (rotation.T @ (data.site_xpos[site_id] - data.xpos[root_body])).tolist()
            )
    payload = json.dumps(
        {"angles": angles.tolist(), "frames": frames, "positions": positions}
    )
    title = html.escape(joint_name)
    initial = int(np.argmin(np.abs(angles)))
    document = f'''<!doctype html><meta charset="utf-8"><style>
body{{font-family:system-ui;margin:0;background:#faf7f0;color:#263b3b;padding:20px}}
main{{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}}img{{width:min(100%,640px);border-radius:12px}}
input{{width:100%}}aside{{flex:1;min-width:220px}}h2{{margin-top:0}}pre{{font-size:16px}}
</style><h2>{title}: forward kinematics</h2><main><img id="pose" src="data:image/jpeg;base64,{frames[initial]}" alt="Actual MuJoCo joint configuration"><aside>
<p>Move the visible {side.lower()}-leg joint marked by the orange ring. The root and other joints stay fixed.</p><input id="angle" aria-label="{title} angle" type="range" min="0" max="{len(angles) - 1}" value="{initial}" step="1" disabled><p id="value">{angles[initial]:.1f} degrees</p>
<h3>{side} foot in torso frame</h3><pre id="position">{html.escape(str(positions[initial]))} m</pre>
<p>These are geometric configurations, not a balance test or the full reachable workspace.</p>
<svg id="trace" viewBox="0 0 280 210" style="width:100%;max-width:350px"></svg></aside></main>
<script>const data={payload};const slider=document.getElementById('angle');
const xs=data.positions.map(p=>p[0]),zs=data.positions.map(p=>p[2]);
const scale=(v,a,b,lo,hi)=>lo+(v-a)/(b-a||1)*(hi-lo);
const point=p=>[scale(p[0],Math.min(...xs)-.01,Math.max(...xs)+.01,30,255),scale(p[2],Math.min(...zs)-.01,Math.max(...zs)+.01,170,20)];
function update(){{const i=+slider.value;document.getElementById('pose').src='data:image/jpeg;base64,'+data.frames[i];
document.getElementById('value').textContent=data.angles[i].toFixed(1)+' degrees';
document.getElementById('position').textContent=data.positions[i].map((v,j)=>['x','y','z'][j]+': '+v.toFixed(3)+' m').join('\u005cn');
const q=point(data.positions[i]);document.getElementById('trace').innerHTML='<polyline fill="none" stroke="#25847a" stroke-width="3" points="'+data.positions.map(point).map(p=>p.join(',')).join(' ')+'"/><circle cx="'+q[0]+'" cy="'+q[1]+'" r="6" fill="#d78535"/><text x="90" y="203" font-size="13">Foot x–z path (m)</text>';}}
slider.disabled=false;slider.oninput=update;update();</script>'''
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    destination.with_suffix(".frames.json").write_text(payload)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "joint": joint_name,
                "foot_site": site_name,
                "angles_deg": angles.tolist(),
                "foot_positions_torso_m": positions,
                "projected_joint_pixels": projected_joints,
                "camera": {"azimuth": camera.azimuth, "elevation": camera.elevation},
                "visible_leg_verified": True,
            },
            indent=2,
        )
    )
    return positions


def save_annotated_pose(env, destination):
    """Show a measured joint-center skeleton through translucent model covers.

    Markers are joint anchors, not guesses at points on the outer shell. The
    annotation records projected centers and camera settings for verification.
    """
    from PIL import ImageDraw

    model, data = model_state(env)
    names = [
        "Left_Hip_Yaw",
        "Left_Hip_Roll",
        "Left_Hip_Pitch",
        "Left_Knee",
        "Left_Ankle",
    ]
    ids = [
        next(i for i in range(model.njnt) if model.joint(i).name.split("/")[-1] == name)
        for name in names
    ]
    foot = next(
        i for i in range(model.nsite) if model.site(i).name.endswith("left_foot")
    )
    anchors = [data.xanchor[i].copy() for i in ids] + [data.site_xpos[foot].copy()]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.mean(anchors, axis=0) + np.array([0, 0, 0.075])
    camera.distance, camera.azimuth, camera.elevation = 0.90, 255, -10
    width, height = 600, 660
    colors = ["#006c67", "#2355b9", "#a13e8e", "#cf6517", "#b0303e"]
    original = model.geom_rgba.copy()
    original_height = model.vis.global_.offheight
    left_root = int(model.jnt_bodyid[ids[0]])
    left_bodies = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == left_root:
                left_bodies.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    try:
        model.vis.global_.offheight = max(original_height, height)
        for geom in range(model.ngeom):
            if model.geom_group[geom] == 2:
                selected = int(model.geom_bodyid[geom]) in left_bodies
                model.geom_rgba[geom, :3] = (
                    [0.25, 0.72, 0.67] if selected else [0.65, 0.68, 0.69]
                )
                model.geom_rgba[geom, 3] = 0.80 if selected else 0.15
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render()).convert("RGBA")
            eyes = renderer.scene.camera
            origin = (eyes[0].pos + eyes[1].pos) / 2
            forward = eyes[0].forward / np.linalg.norm(eyes[0].forward)
            up = eyes[0].up / np.linalg.norm(eyes[0].up)
            right = np.cross(forward, up)
            near = float(eyes[0].frustum_near)
            vertical = float(eyes[0].frustum_top - eyes[0].frustum_bottom)

            def project(point):
                delta = point - origin
                depth = float(delta @ forward)
                scale = near * height / (vertical * depth)
                return (
                    width / 2 + float(delta @ right) * scale,
                    height / 2 - float(delta @ up) * scale,
                )

            points = [project(point) for point in anchors]
            # Project collars in each actual hinge plane, not screen-aligned circles.
            collars = []
            for index, joint in enumerate(ids):
                axis = data.xaxis[joint]
                basis = np.eye(3)[np.argmin(np.abs(axis))]
                u = np.cross(axis, basis)
                u /= np.linalg.norm(u)
                v = np.cross(axis, u)
                radius = 0.011 if index < 3 else 0.016
                angles = np.linspace(0, 2 * np.pi, 49)
                center = anchors[index]
                ring = np.array(
                    [radius * (np.cos(t) * u + np.sin(t) * v) for t in angles]
                )
                front = [project(center + offset + axis * 0.006) for offset in ring]
                back = [project(center + offset - axis * 0.006) for offset in ring]
                collars.append((front, back))
            right_hip = next(
                j
                for j in range(model.njnt)
                if model.joint(j).name.endswith("Right_Hip_Yaw")
            )
            left_depth = float((anchors[0] - origin) @ forward)
            right_depth = float((data.xanchor[right_hip] - origin) @ forward)
            assert left_depth < right_depth, "The annotated leg must face the camera."
    finally:
        model.geom_rgba[:] = original
        model.vis.global_.offheight = original_height
    # Preserve recognizable links while quieter branches recede.
    frame = Image.alpha_composite(
        frame, Image.new("RGBA", frame.size, (250, 247, 240, 20))
    )
    draw = ImageDraw.Draw(frame)
    font = lesson_font(19)
    small = lesson_font(15)
    bold = lesson_font(17, bold=True)
    draw.rounded_rectangle((12, 10, 588, 65), radius=12, fill="#faf7f0")
    draw.text(
        (26, 19),
        "Left leg • links and revolute joints",
        font=font,
        fill="#263b3b",
    )
    draw.text(
        (26, 43),
        "Highlighted links connect the numbered joint pivots.",
        font=small,
        fill="#516766",
    )
    for a, b in zip(points, points[1:]):
        draw.line([a, b], fill="white", width=7)
        draw.line([a, b], fill="#263b3b", width=3)
    for (front, back), color in zip(collars, colors):
        draw.polygon(back, fill="#263b3b", outline="white")
        for k in (0, 12, 24, 36):
            draw.line([back[k], front[k]], fill=color, width=4)
        draw.polygon(front, fill="#faf7f0", outline=color, width=3)
        # A partial rim and arrowhead communicate depth and hinge rotation.
        draw.line(front[5:29], fill=color, width=4)
        tip = np.array(front[28])
        tangent = tip - np.array(front[25])
        tangent /= max(np.linalg.norm(tangent), 1e-6)
        side = np.array([-tangent[1], tangent[0]])
        draw.polygon(
            [
                tuple(tip),
                tuple(tip - 8 * tangent + 4 * side),
                tuple(tip - 8 * tangent - 4 * side),
            ],
            fill=color,
        )
    # Place the three close hip labels separately; endpoints stay at exact centers.
    label_positions = [(40, 130), (40, 205), (380, 265), (40, 390), (380, 490)]
    for index, (name, point, label, color) in enumerate(
        zip(names, points, label_positions, colors)
    ):
        x, y = point
        lx, ly = label
        draw.line([(x, y), (lx + 80, ly + 16)], fill=color, width=2)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        draw.rounded_rectangle(
            (lx - 5, ly - 5, lx + 174, ly + 44), radius=8, fill="#faf7f0"
        )
        draw.text(
            (lx, ly),
            f"{index + 1}  {name.replace('Left_', '').replace('_', ' ')}",
            font=bold,
            fill=color,
        )
        draw.text(
            (lx, ly + 23),
            f"axis {model.jnt_axis[ids[index]].round(1).tolist()}",
            font=small,
            fill="#516766",
        )
    x, y = points[-1]
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#263b3b", outline="white", width=2)
    draw.rounded_rectangle((12, 612, 588, 651), radius=10, fill="#faf7f0")
    draw.text(
        (26, 622),
        "Collars are schematic overlays at measured hinge centers.",
        font=small,
        fill="#263b3b",
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(destination)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "description": "Actual MuJoCo joint anchors projected through the rendered camera; not shell landmarks.",
                "camera": {"azimuth": 255, "elevation": -10, "distance": 0.90},
                "left_hip_depth": left_depth,
                "right_hip_depth": right_depth,
                "highlighted_bodies": [model.body(b).name for b in sorted(left_bodies)],
                "joints": [
                    {
                        "name": name,
                        "world_anchor": anchor.tolist(),
                        "pixel": list(pixel),
                        "local_axis": model.jnt_axis[joint].tolist(),
                    }
                    for name, joint, anchor, pixel in zip(names, ids, anchors, points)
                ],
            },
            indent=2,
        )
        + "\n"
    )
