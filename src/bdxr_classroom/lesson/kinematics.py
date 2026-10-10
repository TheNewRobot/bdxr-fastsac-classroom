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
    """Collapse rigid links, but derive every joint connection from the model."""
    rows = joint_inventory(env)
    coordinates = {"Floating base": (470, 52)}
    branches = {"Left": (180, 0), "Right": (470, 0), "Head": (760, 0)}
    for row in rows:
        branch = row["name"].split("_")[0]
        branch = branch if branch in ("Left", "Right") else "Head"
        x, index = branches[branch]
        coordinates[row["name"]] = (x, 157 + index * 83)
        branches[branch] = (x, index + 1)
    model, _ = model_state(env)
    for side in ("left", "right"):
        site = next(
            i for i in range(model.nsite) if model.site(i).name.endswith(f"{side}_foot")
        )
        body = int(model.site_bodyid[site])
        while body and not model.body_jntnum[body]:
            body = int(model.body_parentid[body])
        parent_joint = int(model.body_jntadr[body])
        parent = model.joint(parent_joint).name.split("/")[-1]
        name = f"{side}_foot (site)"
        rows.append({"name": name, "parent": parent})
        coordinates[name] = (coordinates[parent][0], coordinates[parent][1] + 83)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="690" viewBox="0 0 940 690">',
        '<rect width="940" height="690" rx="18" fill="#faf7f0"/>',
    ]
    for row in rows:
        x1, y1 = coordinates[row["parent"]]
        x2, y2 = coordinates[row["name"]]
        parts.append(
            f'<path d="M{x1},{y1 + 22} V{y2 - 34} H{x2} V{y2 - 22}" fill="none" stroke="#a6b7ad" stroke-width="3"/>'
        )
    for name, (x, y) in coordinates.items():
        color = COLORS.get(name.split("_")[0], "#263b3b")
        if name.endswith("(site)"):
            color = "#718c83"
        parts.extend(
            [
                f'<rect x="{x - 124}" y="{y - 22}" width="248" height="44" rx="10" fill="{color}"/>',
                f'<text x="{x}" y="{y + 6}" font-family="sans-serif" font-size="17" text-anchor="middle" fill="white">{html.escape(name)}</text>',
            ]
        )
    parts.append(
        '<text x="470" y="658" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#516766">Fixed links collapsed • foot sites are measurement points, not motors</text></svg>'
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
    left-foot position in the torso frame, never an asserted safe workspace.
    """
    model, data = model_state(env)
    row = next(row for row in joint_inventory(env) if row["name"] == joint_name)
    angles = np.asarray(angles_deg, dtype=float)
    low, high = row["limits_deg"]
    if not np.all(np.isfinite(angles)) or angles.min() < low or angles.max() > high:
        raise ValueError(f"Requested angles exceed {joint_name}'s model limits.")
    site_id = next(
        i for i in range(model.nsite) if model.site(i).name.endswith("left_foot")
    )
    root_joint = next(
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    root_body = int(model.jnt_bodyid[root_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[root_body] + np.array([0, 0, 0.04])
    camera.distance, camera.azimuth, camera.elevation = 1.20, 65, -12
    frames, positions = [], []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for angle in angles:
            data.qpos[row["qpos_address"]] = np.deg2rad(angle)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
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
    document = f'''<!doctype html><meta charset="utf-8"><style>
body{{font-family:system-ui;margin:0;background:#faf7f0;color:#263b3b;padding:20px}}
main{{display:flex;gap:24px;flex-wrap:wrap}}img{{width:min(100%,640px);border-radius:12px}}
input{{width:100%}}aside{{flex:1;min-width:220px}}h2{{margin-top:0}}pre{{font-size:16px}}
</style><h2>{title}: forward kinematics</h2><main><img id="pose" alt="Actual MuJoCo joint configuration"><aside>
<p>Move one joint. The root and other joints stay fixed.</p><input id="angle" aria-label="Joint angle" type="range" min="0" max="{len(angles) - 1}" value="0" step="1"><p id="value"></p>
<h3>Left foot in torso frame</h3><pre id="position"></pre>
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
slider.oninput=update;update();</script>'''
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "joint": joint_name,
                "angles_deg": angles.tolist(),
                "foot_positions_torso_m": positions,
            },
            indent=2,
        )
    )
    return positions


def save_annotated_pose(env, destination):
    """Label left-leg joint anchors projected from the actual MuJoCo camera."""
    from PIL import ImageDraw, ImageFont

    model, data = model_state(env)
    root_joint = next(
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    )
    root_body = int(model.jnt_bodyid[root_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[root_body] + np.array([0, 0, 0.035])
    camera.distance, camera.azimuth, camera.elevation = 1.15, 65, -12
    width, height = 640, 620
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        frame = Image.fromarray(renderer.render())
        eyes = renderer.scene.camera
        origin = (eyes[0].pos + eyes[1].pos) / 2
        forward = eyes[0].forward / np.linalg.norm(eyes[0].forward)
        up = eyes[0].up / np.linalg.norm(eyes[0].up)
        right = np.cross(forward, up)
        vertical = float(eyes[0].frustum_top - eyes[0].frustum_bottom)
        near = float(eyes[0].frustum_near)

        def project(point):
            delta = point - origin
            depth = float(delta @ forward)
            scale = near * height / (vertical * depth)
            return (
                width / 2 + float(delta @ right) * scale,
                height / 2 - float(delta @ up) * scale,
            )

        annotated = Image.new("RGB", (1000, height), "#faf7f0")
        annotated.paste(frame, (0, 0))
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
            small = ImageFont.truetype("DejaVuSans.ttf", 15)
        except OSError:
            font = small = ImageFont.load_default()
        draw.text((665, 30), "Follow the left leg", font=font, fill="#263b3b")
        names = [
            row["name"]
            for row in joint_inventory(env)
            if row["name"].startswith("Left_")
        ]
        for index, name in enumerate(names):
            joint = next(
                i
                for i in range(model.njnt)
                if model.joint(i).name.split("/")[-1] == name
            )
            x, y = project(data.xanchor[joint])
            label_y = 115 + index * 88
            draw.line(
                [(x, y), (625, label_y + 12), (658, label_y + 12)],
                fill="#58d5bc",
                width=2,
            )
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5), fill="#25847a", outline="white", width=2
            )
            draw.text((665, label_y), name, font=font, fill="#263b3b")
            axis = model.jnt_axis[joint].round(1).tolist()
            draw.text(
                (665, label_y + 28), f"local axis {axis}", font=small, fill="#516766"
            )
        draw.text(
            (665, 580), "Anchors and axes read from model", font=small, fill="#516766"
        )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(destination)
