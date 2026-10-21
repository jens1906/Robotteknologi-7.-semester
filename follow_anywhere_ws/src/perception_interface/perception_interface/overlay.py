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


import colorsys
import math

import cv2

# Colors (BGR)
LEADER_COLOR = (0, 220, 0)
TEXT_COLOR = (0, 0, 0)
HEADER_COLOR = (255, 255, 255)
NOTICE_COLOR = (0, 165, 255)
CLICK_MARKER_COLOR = (0, 220, 255)
STATE_COLORS = ((255, 255, 255), (0, 220, 255), (0, 220, 0), (0, 165, 255), (60, 60, 255))

# Leader states (PerceptionLeaderState)
STATE_NAMES = ('IDLE', 'ENROLLING', 'FOLLOWING', 'SEARCHING', 'LOST')
FOLLOWING_STATE = 2
SEARCHING_STATE = 3
LOST_STATE = 4

# Text and drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
GOLDEN_RATIO = 0.61803398875
HEADER_BOTTOM_PIXELS = 30
BANNER_BOTTOM_PIXELS = 54
CLICK_MARKER_RADIUS_PIXELS = 22


def PickColorForTrackId(trackId):
    hueDegrees = (trackId * GOLDEN_RATIO % 1.0) * 240
    if hueDegrees > 60:
        hueDegrees += 120
    red, green, blue = colorsys.hsv_to_rgb(hueDegrees / 360, 1.0, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def DrawLabel(image, text, left, bottom, backgroundColor, textScale=0.5, textThickness=1):
    (textWidth, textHeight), baseline = cv2.getTextSize(text, FONT, textScale, textThickness)
    top = max(0, bottom - textHeight - baseline - 4)
    cv2.rectangle(image, (left, top), (left + textWidth + 6, top + textHeight + baseline + 4), backgroundColor, -1)
    cv2.putText(image, text, (left + 3, top + textHeight + 1), FONT, textScale, TEXT_COLOR, textThickness)


def DrawTrackedPeople(image, trackedPeople, showDistances):
    for trackedPerson in trackedPeople:
        left, top = int(trackedPerson.x_min), int(trackedPerson.y_min)
        right, bottom = int(trackedPerson.x_max), int(trackedPerson.y_max)
        boxColor = LEADER_COLOR if trackedPerson.is_leader else PickColorForTrackId(trackedPerson.id)
        cv2.rectangle(image, (left, top), (right, bottom), boxColor, 4 if trackedPerson.is_leader else 2)
        labelText = f'LEADER #{trackedPerson.id}' if trackedPerson.is_leader else f'#{trackedPerson.id}'
        if showDistances and not math.isnan(trackedPerson.reid_distance):
            labelText += f' {trackedPerson.reid_distance:.2f}'
        if showDistances and trackedPerson.position_valid:
            labelText += f' {math.hypot(trackedPerson.position.x, trackedPerson.position.y):.1f}m'
        DrawLabel(image, labelText, left, top, boxColor)


def DrawLeaderBanner(image, leaderState):
    if leaderState is None:
        DrawLabel(image, 'LEADER: no leader_state yet', 0, BANNER_BOTTOM_PIXELS, HEADER_COLOR, 0.6)
        return
    bannerText = f'LEADER: {STATE_NAMES[leaderState.state]}'
    if leaderState.state == FOLLOWING_STATE:
        bannerText += f' #{leaderState.track_id}  {leaderState.distance:.1f} m'
    if leaderState.state in (SEARCHING_STATE, LOST_STATE) and leaderState.last_seen_time.sec > 0:
        secondsSinceSeen = ConvertStampToSeconds(leaderState.header.stamp) - ConvertStampToSeconds(
            leaderState.last_seen_time)
        bannerText += f'  last seen {secondsSinceSeen:.0f} s ago'
    DrawLabel(image, bannerText, 0, BANNER_BOTTOM_PIXELS, STATE_COLORS[leaderState.state], 0.6)


def ConvertStampToSeconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def DrawHeader(image, text):
    DrawLabel(image, text, 0, HEADER_BOTTOM_PIXELS, HEADER_COLOR, 0.7, 2)


def DrawNotice(image, text, color):
    DrawLabel(image, text, 0, image.shape[0] - 8, color, 0.7, 2)


def DrawClickMarker(image, markerX, markerY):
    center = (int(markerX), int(markerY))
    cv2.circle(image, center, CLICK_MARKER_RADIUS_PIXELS, TEXT_COLOR, 5)
    cv2.circle(image, center, CLICK_MARKER_RADIUS_PIXELS, CLICK_MARKER_COLOR, 2)


def FindPersonAtPoint(trackedPeople, pointX, pointY):
    peopleUnderPoint = [trackedPerson for trackedPerson in trackedPeople
                        if trackedPerson.x_min <= pointX <= trackedPerson.x_max
                        and trackedPerson.y_min <= pointY <= trackedPerson.y_max]
    return min(peopleUnderPoint, key=CalculateBoxArea, default=None)


def CalculateBoxArea(trackedPerson):
    return (trackedPerson.x_max - trackedPerson.x_min) * (trackedPerson.y_max - trackedPerson.y_min)
