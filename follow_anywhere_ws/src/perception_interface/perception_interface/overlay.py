"""Draws perception results onto an image. Plain python (OpenCV), no ROS, so it can be tested on its own.

To include:
- Boxes of all tracked people with their tracker id; the leader in green, others in grey.
- Under each box (show_distances): reid distance to the leader (if compared this frame) and the
  distance in metres (from the person's position, when position_valid).
- Leader state banner: IDLE / ENROLLING (with progress) / FOLLOWING / SEARCHING / LOST, last seen
  "x s ago" while searching or lost.
- Header line with mode (exact / live), fps and image age.
- Click feedback: a ring where the user clicked and the calibration service's answer for a few seconds.
- A person-under-the-mouse lookup (box containing the click point, the smallest if several overlap)
  used by perception_viewer to turn a click into a track id.
"""
