"""Who is the leader: enrolment, the stored views, finding them again, lost / last seen. Plain python, no ROS.

To include:
- States: IDLE, ENROLLING, FOLLOWING, SEARCHING, LOST (the values in PerceptionLeaderState).
- Enrolment (calibrate service): collect enroll_views views of the chosen person, enroll_gap apart,
  only clean ones (nobody overlapping, enough of the body visible). These are the anchors and are
  never dropped. Enrolment survives a tracker id change at the same place.
- While FOLLOWING: locked to the tracker id; check it every verify_period; learn new views into a
  rolling buffer (rolling_views) only when clean and new (not a near-duplicate); drop the lock when the
  person looks different (reid_drop) or someone crosses them (OSNet); grace_time for flicker.
- While SEARCHING: compare search_candidates people every search_period (promising ones first, then
  the ones compared longest ago); accept only below reid_accept, with reid_margin over every other
  visible person, confirmed reid_confirm times in the last reid_confirm_window comparisons.
- Precision first: "not identified" is fine, the wrong person is the worst failure.
- Keep last-seen time and position; SEARCHING becomes LOST after lost_time.
- Save the anchors to leader_file (numpy, no pickle, tagged with the reid model) and load them at
  start, so the leader survives a restart.
- Embeddings come in through a callable, so this file needs no model and can be tested on its own.
"""
