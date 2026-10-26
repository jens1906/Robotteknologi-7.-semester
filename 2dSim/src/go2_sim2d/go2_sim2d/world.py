"""A go2_scenarios scene as 2D shapes.

The scenes are the same YAML files the Gazebo sim uses (go2_scenarios/scenes),
loaded and checked by go2_scenarios.scene, so both sims run the same worlds
with the same paths and timing. t is seconds since the sim started; a path
entity starts walking start_delay seconds after that.
"""

from go2_scenarios import scene

from go2_sim2d.geometry import Circle, from_scene

# Footprint of a person seen from above.
PERSON_RADIUS = 0.25


class World:

    def __init__(self, name):
        loaded = scene.load(name)
        self.name = name
        self.static = loaded['static']
        self.movers = loaded['movers']
        self.people = loaded['people']
        self.static_shapes = [from_scene(o) for o in self.static]
        self._mover_paths = [(scene.timeline(m), float(m.get('start_delay', 0.0)))
                             for m in self.movers]
        self._people_paths = [(scene.timeline(p), float(p.get('start_delay', 0.0)))
                              for p in self.people]
        names = [p['name'] for p in self.people]
        self.leader_index = names.index('leader') if 'leader' in names else None

    # Entity ids: statics, then movers, then people, in scene order, from 1.
    def static_id(self, i):
        return 1 + i

    def mover_id(self, i):
        return 1 + len(self.static) + i

    def person_id(self, i):
        return 1 + len(self.static) + len(self.movers) + i

    def mover_poses(self, t):
        """(x, y, yaw) of every mover. Movers keep their yaw, like in Gazebo."""
        return [scene.pose_at(keys, t - delay)[:2] + (float(m.get('yaw', 0.0)),)
                for m, (keys, delay) in zip(self.movers, self._mover_paths)]

    def people_poses(self, t):
        return [scene.pose_at(keys, t - delay) for keys, delay in self._people_paths]

    def mover_shapes(self, poses):
        return [from_scene(m, x, y, yaw) for m, (x, y, yaw) in zip(self.movers, poses)]

    @staticmethod
    def people_shapes(poses):
        return [Circle(x, y, PERSON_RADIUS) for x, y, *_ in poses]
