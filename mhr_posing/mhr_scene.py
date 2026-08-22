"""Show solved MHR bodies in a 3D scene that matches what the camera saw.

`analytic_ik.py` turns 3D keypoints into an MHR pose, but leaves the body wherever
the dataset's world happens to be. This puts it somewhere you can look at, without
changing anything about it:

* the pose stays exactly as solved,
* each person keeps the way they were turned relative to the camera,
* each person keeps their real distance from the camera,
* and with several people, their distances from each other.

The trick is to move the *scene* rather than the body: shift and turn everything so
the camera lands on the origin. Sliding and turning a scene cannot change any
distance inside it, so all four survive by themselves.

The frame we land in
--------------------
    origin      the camera
    +x          the camera's right
    +y          up
    -z          the direction the camera is looking

So a body sits at negative z, exactly as far away as it really was.

Three ways to mirror a body by accident, all of which I managed
--------------------------------------------------------------
1. Camera axes have y pointing **down**; ours has y up. Getting there means
   flipping y *and* z together. Flipping y alone is a mirror.
2. A plotly 3D scene draws +z upwards, so it is tempting to plot the body as
   (x, z, y) -- but *swapping* two axes is a reflection, determinant -1. That
   mirrors the whole scene: left and right change places while towards and away
   stay put. Use (x, -z, y) instead, which is a proper rotation.
3. Looking from the wrong side. The camera is at the origin and the body out at
   negative z, so the camera is in the +z direction from it -- which becomes
   *negative* y once plotted as (x, -z, y).

The first is arithmetic, the other two are only about drawing, but all three look
the same in the end: a body facing the wrong way. `handedness()` checks the one
that is easy to get silently wrong.
"""

import base64
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

CAMERA_TO_VIEWER = np.diag([1.0, -1.0, -1.0])
ROOT_ROTATION = ("root_rx", "root_ry", "root_rz")
ROOT_TRANSLATION = ("root_tx", "root_ty", "root_tz")
CENTIMETRES_PER_UNIT = 10.0          # what root_tx/ty/tz are measured in
BODY_FACES = np.array([0.0, 0.0, 1.0])       # a body at rest faces +z


# ---------------------------------------------------------------------------
# Part 1: putting a body where the camera saw it
# ---------------------------------------------------------------------------

def camera_from_aspset(extrinsic_matrix):
    """An ASPset camera, in the centimetres and y-up the solved body uses."""
    flip = np.diag([1.0, -1.0, -1.0])
    rotation = np.asarray(extrinsic_matrix)[:3, :3] @ flip
    translation = np.asarray(extrinsic_matrix)[:3, 3] / 10.0
    return rotation, translation


def place(rig, pose, camera_rotation, camera_translation):
    """The same pose, moved so the camera is at the origin. Only the root changes.

    Two measured facts make this simple: turning the root does not move the root,
    and root_tx/ty/tz slide the body along the world axes at 10 cm per unit
    whatever the rotation. So the place and the heading are written separately and
    every joint angle is left alone.
    """
    positions, rotations = rig.forward_kinematics(pose)
    root = rig.joint_index["root"]

    seen_position = camera_rotation @ positions[root] + camera_translation
    seen_rotation = camera_rotation @ rotations[root]

    wanted_position = CAMERA_TO_VIEWER @ seen_position
    wanted_rotation = CAMERA_TO_VIEWER @ seen_rotation

    moved = np.array(pose, dtype=float)
    rig.set_rotation(moved, "root", ROOT_ROTATION, wanted_rotation, rotations)
    for name, amount in zip(ROOT_TRANSLATION, wanted_position - positions[root]):
        moved[rig.parameter_index[name]] += amount / CENTIMETRES_PER_UNIT
    return moved


def camera_view():
    """Rotation and translation for `mhr_kit.render.Camera` in this frame."""
    return CAMERA_TO_VIEWER, np.zeros(3)


# ---------------------------------------------------------------------------
# Part 2: reading the result back
# ---------------------------------------------------------------------------

def distance_to_camera(rig, pose):
    """Centimetres from the camera to the body's hips."""
    positions, _ = rig.forward_kinematics(pose)
    return float(np.linalg.norm(positions[rig.joint_index["root"]]))


def facing(rig, pose):
    """The unit vector the hips point along."""
    _, rotations = rig.forward_kinematics(pose)
    direction = rotations[rig.joint_index["root"]] @ BODY_FACES
    return direction / np.linalg.norm(direction)


def turn_from_camera(rig, pose):
    """Degrees the body is turned away from the camera: 0 faces it, 180 faces away.

    Positive means turned towards the camera's right. Only the turn on the level
    counts, so leaning forward is not mistaken for turning away.
    """
    positions, _ = rig.forward_kinematics(pose)
    up = np.array([0.0, 1.0, 0.0])

    look = facing(rig, pose)
    towards = -positions[rig.joint_index["root"]]        # the camera is the origin
    look = look - np.dot(look, up) * up
    towards = towards - np.dot(towards, up) * up
    look /= np.linalg.norm(look)
    towards /= np.linalg.norm(towards)

    return float(np.degrees(np.arctan2(np.dot(np.cross(look, towards), up),
                                       np.dot(look, towards))))


def describe(rig, pose):
    """One line: how far away, and which way round."""
    turn = turn_from_camera(rig, pose)
    size = abs(turn)
    side = "right" if turn > 0 else "left"
    if size < 25:
        words = "facing the camera"
    elif size > 155:
        words = "back to the camera"
    elif size < 115:
        words = f"turned {side}"
    else:
        words = f"turned {side}, mostly away"
    return f"{distance_to_camera(rig, pose)/100:.1f} m, {words} ({turn:+.0f}°)"


# ---------------------------------------------------------------------------
# Part 3: drawing it
# ---------------------------------------------------------------------------

COLOURS = ["lightsteelblue", "lightsalmon", "darkseagreen", "plum", "khaki", "lightpink"]


# How body coordinates are laid out in a plotly scene. A scene draws +z upwards,
# so the body's up axis has to end up there -- but it must be done with a proper
# rotation, not by swapping two axes, which would mirror everything.
#
#     plotted x =  body x      (the camera's right)
#     plotted y = -body z      (distance away from the camera, so positive)
#     plotted z =  body y      (up)
#
PLOT = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0]])


def handedness():
    """+1 if the plotting layout is a rotation, -1 if it secretly mirrors the scene.

    Worth keeping around: a mirrored body looks completely plausible until you
    notice it is facing the wrong way, and no amount of staring at the numbers
    will show it up.
    """
    return float(np.linalg.det(PLOT))


def plotted(points):
    """Body coordinates, laid out for a plotly scene."""
    return np.asarray(points) @ PLOT.T


def _body(mesh, triangles, colour, name, legend=True):
    """One person's mesh."""
    drawn = plotted(mesh)
    return go.Mesh3d(x=drawn[:, 0], y=drawn[:, 1], z=drawn[:, 2],
                     i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                     color=colour, flatshading=False,
                     lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                     lightposition=dict(x=0, y=-3000, z=2000),
                     name=name, showlegend=legend, hoverinfo="skip")


def hand_markers(rig, pose, legend=True):
    """A dot on each wrist, red for the left hand and blue for the right.

    This is the mirror check you can actually see: if the red dot is on the wrong
    side compared with the photograph, the scene is flipped.
    """
    positions, _ = rig.forward_kinematics(pose)
    traces = []
    for joint, colour, label in (("l_wrist", "crimson", "left hand"),
                                 ("r_wrist", "royalblue", "right hand")):
        spot = plotted(positions[rig.joint_index[joint]])
        traces.append(go.Scatter3d(x=[spot[0]], y=[spot[1]], z=[spot[2]], mode="markers",
                                   marker=dict(size=5, color=colour), name=label,
                                   showlegend=legend, hoverinfo="skip"))
    return traces


def _picture(rgb, quality=80):
    """A picture as a compressed image the browser can show.

    Sending raw pixels would mean a few hundred thousand numbers per frame; a
    JPEG is a few tens of kilobytes.
    """
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, "JPEG", quality=quality)
    return go.Image(source="data:image/jpeg;base64,"
                           + base64.b64encode(buffer.getvalue()).decode("ascii"))


def _scene(centre, reach):
    """Axis ranges, and an eye on the same side as the real camera.

    The camera is at the origin and the bodies are out at negative body-z, which
    is *positive* plotted y. So the camera is at smaller plotted y than they are,
    and the eye offset has to be negative.
    """
    middle = plotted(centre)
    return dict(
        xaxis=dict(range=[middle[0] - reach, middle[0] + reach], title="x (cm) right"),
        yaxis=dict(range=[middle[1] - reach, middle[1] + reach], title="distance away (cm)"),
        zaxis=dict(range=[middle[2] - reach, middle[2] + reach], title="y (cm) up"),
        aspectmode="cube",
        camera=dict(eye=dict(x=0.0, y=-2.0, z=0.35), up=dict(x=0, y=0, z=1)))


def _framing(frames_of_bodies, padding=1.25):
    """A box big enough for the bodies, and where to centre it on each frame.

    One size for the whole clip, so the view never zooms in and out; but centred
    per frame, so someone crossing a field stays large instead of shrinking to a
    speck in a box the size of their run.
    """
    reach = 0.0
    centres = []
    for bodies in frames_of_bodies:
        points = np.concatenate(bodies)
        low, high = points.min(axis=0), points.max(axis=0)
        centres.append(0.5 * (low + high))
        reach = max(reach, 0.5 * float((high - low).max()) * padding)
    return centres, reach


def scene_figure(bodies, triangles, names=None, title="", height=620, rig=None, poses=None):
    """An interactive 3D view of one frame: one entry in `bodies` per person.

    Pass `rig` and `poses` as well and each person gets a red dot on their left
    hand and a blue one on their right, so a mirrored scene is obvious at a glance.
    """
    names = names or [f"person {i + 1}" for i in range(len(bodies))]
    centres, reach = _framing([bodies])
    traces = [_body(mesh, triangles, COLOURS[i % len(COLOURS)], names[i])
              for i, mesh in enumerate(bodies)]
    if rig is not None and poses is not None:
        for i, pose in enumerate(poses):
            traces += hand_markers(rig, pose, legend=(i == 0))
    figure = go.Figure(traces)
    figure.update_layout(height=height, margin=dict(l=0, r=0, t=50, b=0),
                         title=dict(text=title, x=0.02, font=dict(size=12)),
                         scene=_scene(centres[0], reach))
    return figure


def playback_figure(pictures, frames_of_bodies, triangles, captions=None, names=None,
                    fps=10, height=560):
    """Footage beside the bodies, with one play button driving both.

    `frames_of_bodies[n]` holds one mesh per person on frame `n`, and every frame
    must hold the same number of people -- an animation needs a fixed set of
    traces, so a changing count would quietly pair bodies with the wrong person.
    """
    counts = {len(bodies) for bodies in frames_of_bodies}
    if len(counts) != 1:
        raise ValueError(f"every frame needs the same number of people, got {sorted(counts)}")
    if len(pictures) != len(frames_of_bodies):
        raise ValueError(f"{len(pictures)} pictures but {len(frames_of_bodies)} frames of bodies")

    people = counts.pop()
    names = names or [f"person {i + 1}" for i in range(people)]
    captions = captions or [""] * len(pictures)
    centres, reach = _framing(frames_of_bodies)

    def traces(n):
        return [_picture(pictures[n])] + [
            _body(frames_of_bodies[n][p], triangles, COLOURS[p % len(COLOURS)],
                  names[p], legend=(n == 0)) for p in range(people)]

    figure = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                           specs=[[{"type": "xy"}, {"type": "scene"}]],
                           subplot_titles=("the footage", "the same body, in its own frame"))
    for position, trace in enumerate(traces(0)):
        figure.add_trace(trace, row=1, col=1 if position == 0 else 2)

    figure.frames = [go.Frame(name=str(n), traces=list(range(people + 1)), data=traces(n),
                              layout=go.Layout(title=dict(text=captions[n]),
                                               scene=_scene(centres[n], reach)))
                     for n in range(len(pictures))]

    figure.update_layout(
        height=height, margin=dict(l=0, r=0, t=90, b=0),
        title=dict(text=captions[0], x=0.02, font=dict(size=12)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        scene=_scene(centres[0], reach),
        legend=dict(orientation="h", y=-0.02, x=0.55),
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=-0.06, xanchor="left",
                          buttons=[
                              dict(label="play", method="animate",
                                   args=[None, dict(frame=dict(duration=int(1000 / fps),
                                                               redraw=True),
                                                    transition=dict(duration=0),
                                                    fromcurrent=True)]),
                              dict(label="pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode="immediate")])])],
        sliders=[dict(x=0.16, y=-0.04, len=0.82, currentvalue=dict(prefix="frame "),
                      steps=[dict(method="animate", label=str(n),
                                  args=[[str(n)], dict(frame=dict(duration=0, redraw=True),
                                                       mode="immediate")])
                             for n in range(len(pictures))])])
    return figure
