"""Scenes for the Go2 sim: a YAML file per world, turned into SDF here.

A scene (scenes/<name>.yaml) lists:
  static:  boxes, cylinders and walls that never move
  movers:  boxes/cylinders that scene_runner moves along a path (kinematic)
  people:  animated walking actors on a path; the one named "leader" is the
           person the Go2 follows

Paths are lists of [x, y] in the world frame. The Go2 spawns at the origin
facing +x, so the world frame is also the robot's odom frame.
Path options: speed (m/s), mode ("loop", the default for people: walk back
to the first point and repeat; "pingpong", the default for movers: walk the
path forwards then backwards), start_delay (s of sim time standing at the
first point before walking).

timeline() is the one definition of where a path entity is when: to_sdf()
writes it as actor waypoints and scene_runner interpolates it for the
ground truth and the movers, so both agree.
"""

import math
import os
from xml.sax.saxutils import escape

import yaml
from ament_index_python.packages import get_package_share_directory

WORLD_NAME = 'go2_world'

# Walking person mesh (skin + animation), vendored from Gazebo classic: models/README.md
ACTOR_MESH = os.path.join(get_package_share_directory('go2_scenarios'), 'models', 'walk.dae')
# Height of the actor mesh origin (hips) above its feet.
ACTOR_Z = 1.0

# Gazebo interpolates actor waypoints with a Catmull-Rom spline, which is
# exactly linear only on evenly spaced points along a straight line. Splitting
# path segments this finely keeps the actor within a few cm of the straight
# path (only corners get rounded), so the linear ground truth stays accurate.
WAYPOINT_SPACING = 0.25

# Nothing may stand within this distance of the spawn point (origin).
SPAWN_CLEARANCE = 1.0
# Free space kept between a person's centre and static objects, and between a
# mover's bounding circle (already generous for boxes) and static objects.
PATH_CLEARANCE = {'people': 0.3, 'movers': 0.1}

DEFAULT_COLOR = [0.6, 0.6, 0.6]


def scenes_dir():
    return os.path.join(get_package_share_directory('go2_scenarios'), 'scenes')


def available():
    return sorted(f[:-5] for f in os.listdir(scenes_dir()) if f.endswith('.yaml'))


def load(name):
    path = os.path.join(scenes_dir(), f'{name}.yaml')
    if not os.path.isfile(path):
        raise ValueError(f"Unknown world '{name}', available: {', '.join(available())}")
    with open(path) as f:
        scene = yaml.safe_load(f) or {}
    for key in ('static', 'movers', 'people'):
        scene.setdefault(key, [])
    for mover in scene['movers']:
        mover.setdefault('mode', 'pingpong')
    scene['static'] = [_as_box(o) if o['type'] == 'wall' else o for o in scene['static']]
    check(scene)
    return scene


def check(scene):
    """Raise ValueError if names repeat, the spawn point is blocked or a path hits an object."""
    names = [e['name'] for key in ('static', 'movers', 'people') for e in scene[key]]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f'Duplicate names in scene: {sorted(duplicates)}')
    for obj in scene['static']:
        if obj['type'] not in ('box', 'cylinder'):
            raise ValueError(f"'{obj['name']}': unknown type '{obj['type']}'")
        if distance(obj, 0.0, 0.0) < SPAWN_CLEARANCE:
            raise ValueError(f"'{obj['name']}' is within {SPAWN_CLEARANCE} m of the spawn point")
    for kind in ('movers', 'people'):
        for entity in scene[kind]:
            reach = _radius(entity) if kind == 'movers' else 0.0
            clearance = PATH_CLEARANCE[kind]
            for _, x, y, _ in timeline(entity):
                if kind == 'movers' and math.hypot(x, y) - reach < SPAWN_CLEARANCE:
                    raise ValueError(f"Mover '{entity['name']}' passes within "
                                     f'{SPAWN_CLEARANCE} m of the spawn point')
                for obj in scene['static']:
                    if distance(obj, x, y) - reach < clearance:
                        raise ValueError(
                            f"'{entity['name']}' passes within {clearance} m of "
                            f"'{obj['name']}' at ({x:.2f}, {y:.2f})")


# ---- Paths ----

def _points(entity):
    """Path corners in walking order, closed so that the last equals the first."""
    path = [tuple(map(float, p)) for p in entity['path']]
    if len(path) < 2:
        raise ValueError(f"'{entity['name']}' needs a path of at least 2 points")
    mode = entity.get('mode', 'loop')
    if mode == 'pingpong':
        return path + path[-2::-1]
    if mode == 'loop':
        return path if path[0] == path[-1] else path + [path[0]]
    raise ValueError(f"'{entity['name']}': unknown mode '{mode}'")


def timeline(entity):
    """[(t, x, y, yaw), ...] for one lap, t from 0, evenly spaced along each segment.

    Each point faces the segment leaving it, so the person turns over the last
    WAYPOINT_SPACING before a corner. The lap is closed: the last point equals
    the first, yaw included.
    """
    corners = _points(entity)
    speed = float(entity.get('speed', 1.0))
    keys = []
    t = 0.0
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            raise ValueError(f"'{entity['name']}': repeated point {x0, y0} in path")
        yaw = math.atan2(y1 - y0, x1 - x0)
        pieces = max(1, math.ceil(length / WAYPOINT_SPACING))
        for i in range(pieces):
            f = i / pieces
            keys.append((t + f * length / speed, x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw))
        t += length / speed
    keys.append((t, *corners[-1], keys[0][3]))
    # Unwrap yaw so consecutive points never differ by more than pi.
    unwrapped = [keys[0]]
    for k in keys[1:]:
        prev = unwrapped[-1][3]
        unwrapped.append(k[:3] + (prev + _wrap(k[3] - prev),))
    return unwrapped


def pose_at(keys, t):
    """(x, y, yaw) at t seconds after the entity started walking; laps repeat."""
    if t <= 0.0:
        return keys[0][1:]
    t = math.fmod(t, keys[-1][0])
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(keys, keys[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0), _wrap(a0 + f * _wrap(a1 - a0))
    return keys[-1][1:]


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _radius(obj):
    """Radius of the circle around an object's centre that contains it, in xy."""
    if obj['type'] == 'cylinder':
        return float(obj['radius'])
    sx, sy = obj['size'][:2]
    return math.hypot(sx, sy) / 2


def _pose2d(obj):
    return (list(map(float, obj['pose'])) + [0.0])[:3]


def distance(obj, x, y):
    """Distance in xy from (x, y) to the nearest part of a static box or cylinder."""
    cx, cy, yaw = _pose2d(obj)
    if obj['type'] == 'cylinder':
        return math.hypot(x - cx, y - cy) - float(obj['radius'])
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
    sx, sy = obj['size'][:2]
    return math.hypot(max(abs(lx) - sx / 2, 0.0), max(abs(ly) - sy / 2, 0.0))


def _as_box(wall):
    """A wall from/to/height/thickness as the equivalent box."""
    (x0, y0), (x1, y1) = wall['from'], wall['to']
    return {
        'type': 'box',
        'name': wall['name'],
        'pose': [(x0 + x1) / 2, (y0 + y1) / 2, math.atan2(y1 - y0, x1 - x0)],
        'size': [math.hypot(x1 - x0, y1 - y0), float(wall.get('thickness', 0.15)),
                 float(wall.get('height', 2.0))],
        'color': wall.get('color', [0.85, 0.85, 0.8]),
    }


# ---- SDF ----

def to_sdf(scene):
    parts = [_WORLD_HEADER.format(world=WORLD_NAME)]
    parts += [_static(o) for o in scene['static']]
    parts += [_mover(m) for m in scene['movers']]
    parts += [_actor(p) for p in scene['people']]
    parts.append(_WORLD_FOOTER)
    return '\n'.join(parts)


def _fmt(*values):
    return ' '.join(f'{v:.4f}'.rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
                    for v in values)


def _geometry(obj):
    if obj['type'] == 'cylinder':
        return (f'<cylinder><radius>{_fmt(float(obj["radius"]))}</radius>'
                f'<length>{_fmt(float(obj["length"]))}</length></cylinder>')
    return f'<box><size>{_fmt(*map(float, obj["size"]))}</size></box>'


def height(obj):
    return float(obj['length'] if obj['type'] == 'cylinder' else obj['size'][2])


def _material(obj):
    r, g, b = map(float, obj.get('color', DEFAULT_COLOR))
    return (f'<material><ambient>{_fmt(r, g, b)} 1</ambient>'
            f'<diffuse>{_fmt(r, g, b)} 1</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular></material>')


def _link_body(obj):
    geometry = _geometry(obj)
    return (f'<collision name="collision"><geometry>{geometry}</geometry></collision>\n'
            f'        <visual name="visual"><geometry>{geometry}</geometry>'
            f'{_material(obj)}</visual>')


def _static(obj):
    x, y, yaw = _pose2d(obj)
    return f'''    <model name="{escape(obj['name'])}">
      <static>true</static>
      <pose>{_fmt(x, y, height(obj) / 2, 0.0, 0.0, yaw)}</pose>
      <link name="link">
        {_link_body(obj)}
      </link>
    </model>'''


def _mover(obj):
    # Kinematic: no gravity, lifted off the ground, heavy so the robot can't
    # push it. scene_runner sets its pose every step.
    _, x, y, _ = timeline(obj)[0]
    return f'''    <model name="{escape(obj['name'])}">
      <pose>{_fmt(x, y, height(obj) / 2 + 0.01, 0.0, 0.0, float(obj.get('yaw', 0.0)))}</pose>
      <link name="link">
        <gravity>false</gravity>
        <inertial><mass>1000</mass>
          <inertia><ixx>1000</ixx><iyy>1000</iyy><izz>1000</izz></inertia>
        </inertial>
        {_link_body(obj)}
      </link>
    </model>'''


def _actor(person):
    keys = timeline(person)
    waypoints = '\n'.join(
        f'          <waypoint><time>{t:.3f}</time>'
        f'<pose>{_fmt(x, y, ACTOR_Z, 0.0, 0.0, yaw)}</pose></waypoint>'
        for t, x, y, yaw in keys)
    # No actor <pose>: Gazebo adds it to every waypoint pose.
    return f'''    <actor name="{escape(person['name'])}">
      <skin><filename>{ACTOR_MESH}</filename><scale>1.0</scale></skin>
      <animation name="walk">
        <filename>{ACTOR_MESH}</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <delay_start>{float(person.get('start_delay', 0.0)):.3f}</delay_start>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walk">
{waypoints}
        </trajectory>
      </script>
    </actor>'''


_WORLD_HEADER = '''<?xml version="1.0" ?>
<!-- Generated by go2_scenarios/scene.py; edit the scene YAML instead. -->
<sdf version="1.9">
  <world name="{world}">

    <!-- Core simulation systems -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <!-- Physics: ODE, 1ms step, real-time factor 1 -->
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <gravity>0 0 -9.81</gravity>

    <!-- Sunlight -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>
'''

_WORLD_FOOTER = '''
  </world>
</sdf>
'''
