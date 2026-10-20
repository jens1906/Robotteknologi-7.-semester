"""One tracker, one or more leader matchers (an embedder + a Leader each) that share its people."""
import math


class Pipeline:
    def __init__(self, tracker, matchers):
        """tracker: PersonTracker. matchers: {name: (embed, Leader)}, see reid.make_embedder."""
        self.tracker, self.matchers, self.people = tracker, matchers, []

    def process(self, frame_bgr, now):
        """Next frame in video time `now` (s). Returns the people and {name: leader.Result}."""
        self.people = self.tracker(frame_bgr)
        by_id = {p.id: p for p in self.people}
        results = {}
        for name, (embed, leader) in self.matchers.items():
            results[name] = leader.update(
                self.people, lambda ids, embed=embed: embed(frame_bgr, [by_id[i] for i in ids],
                                                            self.people), now)
        return self.people, results

    def person_at(self, x, y):
        """The person under an image point (the smallest box if boxes overlap), or None."""
        hit = [p for p in self.people if p.box[0] <= x <= p.box[2] and p.box[1] <= y <= p.box[3]]
        return min(hit, key=lambda p: p.area, default=None)

    def enroll_at(self, x, y, now):
        """Make the person at an image point the leader of every matcher. Returns them, or None."""
        person = self.person_at(x, y)
        if person is not None:
            for _, leader in self.matchers.values():
                leader.enroll(person.id, now)
        return person

    def enroll_centre(self, width, height, now):
        """Make the person nearest the image centre the leader (no click needed). Returns them, or None."""
        centre = lambda p: math.hypot((p.box[0] + p.box[2]) / 2 - width / 2, (p.box[1] + p.box[3]) / 2 - height / 2)  # noqa: E731
        person = min(self.people, key=centre, default=None)
        if person is not None:
            for _, leader in self.matchers.values():
                leader.enroll(person.id, now)
        return person

    def new_scene(self):
        """The video restarted: new tracker, and every leader searches again with the views it has."""
        self.tracker.reset()
        self.people = []
        for _, leader in self.matchers.values():
            leader.new_scene()

    def reset(self):
        for _, leader in self.matchers.values():
            leader.reset()
