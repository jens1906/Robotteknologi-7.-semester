"""Draws perception results onto an image. Plain python (OpenCV), no ROS, so it can be tested on its own.

To include:
- Boxes of all tracked people with their tracker id; the leader in green, others in grey.
- Under each box (show_distances): reid distance to the leader (if compared this frame) and the
  distance in metres (from the person's position, when position_valid).
- Leader state banner: IDLE / ENROLLING (with "turn around slowly" and the views per direction) /
  FOLLOWING #id (with the stored views per group), and "NOT IN FRAME: last seen x s ago" (with the
  last position when there is one) while the leader is not identified in the image.
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
ENROLLING_STATE = 1
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
        bannerText, bannerColor = 'LEADER: no leader_state yet', HEADER_COLOR
    elif leaderState.state in (FOLLOWING_STATE, SEARCHING_STATE, LOST_STATE) and not leaderState.leader_visible:
        bannerText, bannerColor = DescribeLeaderOutOfFrame(leaderState), NOTICE_COLOR
    elif leaderState.state == FOLLOWING_STATE:
        bannerText, bannerColor = DescribeFollowedLeader(leaderState), LEADER_COLOR
    elif leaderState.state == ENROLLING_STATE:
        bannerText = f'LEADER: ENROLLING, turn around slowly  {DescribeStoredViews(leaderState)}'
        bannerColor = STATE_COLORS[ENROLLING_STATE]
    else:
        bannerText, bannerColor = f'LEADER: {STATE_NAMES[leaderState.state]}', STATE_COLORS[leaderState.state]
    DrawLabel(image, bannerText, 0, BANNER_BOTTOM_PIXELS, bannerColor, 0.6)


def DescribeLeaderOutOfFrame(leaderState):
    if math.isnan(leaderState.seconds_since_seen):
        return f'LEADER: {STATE_NAMES[leaderState.state]}, looking for the saved leader'
    statusText = 'LEADER LOST' if leaderState.state == LOST_STATE else 'LEADER NOT IN FRAME'
    description = f'{statusText}: last seen {leaderState.seconds_since_seen:.0f} s ago'
    if not math.isnan(leaderState.last_seen_position.x):
        description += f' at x {leaderState.last_seen_position.x:.1f} y {leaderState.last_seen_position.y:.1f} m'
    return description


def DescribeFollowedLeader(leaderState):
    description = f'LEADER: FOLLOWING #{leaderState.track_id}'
    if not math.isnan(leaderState.distance):
        description += f'  {leaderState.distance:.1f} m'
    return f'{description}  views {DescribeStoredViews(leaderState)}'


def DescribeStoredViews(leaderState):
    return (f'front {leaderState.front_views} back {leaderState.back_views} left {leaderState.side_left_views} '
            f'right {leaderState.side_right_views} hidden {leaderState.partly_hidden_views}')


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
