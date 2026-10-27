"""Draws perception results onto an image. Plain python (OpenCV), no ROS, so it can be tested on its own.

To include:
- Boxes of all tracked people with their tracker id; the leader in green, others in grey.
- Under each box (show_distances): reid distance to the leader (if compared this frame) and the
  distance in metres (from the person's position, when position_valid).
- Leader state banner: IDLE / ENROLLING (with "turn around slowly" and the views per direction) /
  FOLLOWING #id (with the stored views per group), and "NOT IN FRAME: last seen x s ago" (with the
  last position when there is one) while the leader is not identified in the image.
- Top-down map in the bottom right corner (as seen from the ceiling): the robot at the bottom middle,
  forward up, left left, a half ring every metre, every person with a valid position as a dot with its id and an
  arrow to where they will be in MAP_ARROW_SECONDS at their velocity (no arrow when standing still).
- Header line with mode (exact / live), fps and image age.
- Click feedback: a ring where the user clicked and the calibration service's answer for a few seconds.
- A person-under-the-mouse lookup (box containing the click point, the smallest if several overlap)
  used by perception_viewer to turn a click into a track id.
"""


import colorsys
import math

import cv2
import numpy as np

# Colors (BGR)
LEADER_COLOR = (0, 220, 0)
TEXT_COLOR = (0, 0, 0)
HEADER_COLOR = (255, 255, 255)
NOTICE_COLOR = (0, 165, 255)
CLICK_MARKER_COLOR = (0, 220, 255)
STATE_COLORS = ((255, 255, 255), (0, 220, 255), (0, 220, 0), (0, 165, 255), (60, 60, 255))

# Leader states (PerceptionDebugFrame.leader_state)
STATE_NAMES = ('IDLE', 'ENROLLING', 'FOLLOWING', 'SEARCHING', 'LOST')
ENROLLING_STATE = 1
FOLLOWING_STATE = 2
SEARCHING_STATE = 3
LOST_STATE = 4
NO_TRACK_ID = -1

# Text and drawing
FONT = cv2.FONT_HERSHEY_SIMPLEX
GOLDEN_RATIO = 0.61803398875
HEADER_BOTTOM_PIXELS = 30
BANNER_BOTTOM_PIXELS = 54
CLICK_MARKER_RADIUS_PIXELS = 22

# Top-down map
MAP_SIZE_PIXELS = 200
MAP_MARGIN_PIXELS = 10
MAP_ROBOT_MARGIN_PIXELS = 12
MAP_DOT_RADIUS_PIXELS = 6
MAP_ARROW_SECONDS = 1.0
MAP_OPACITY = 0.85
MAP_BACKGROUND_COLOR = (40, 40, 40)
MAP_RING_COLOR = (110, 110, 110)
MAP_ROBOT_COLOR = (255, 255, 255)


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


def DrawTopDownMap(image, trackedPeople, mapRangeMeters):
    mapImage = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), MAP_BACKGROUND_COLOR, np.uint8)
    robotX = MAP_SIZE_PIXELS // 2
    robotY = MAP_SIZE_PIXELS - MAP_ROBOT_MARGIN_PIXELS
    pixelsPerMeter = (MAP_SIZE_PIXELS - MAP_ROBOT_MARGIN_PIXELS) / mapRangeMeters
    for ringMeters in range(1, int(mapRangeMeters) + 1):
        ringRadius = int(ringMeters * pixelsPerMeter)
        cv2.ellipse(mapImage, (robotX, robotY), (ringRadius, ringRadius), 0, 180, 360, MAP_RING_COLOR, 1)
        cv2.putText(mapImage, f'{ringMeters}', (robotX + 3, robotY - ringRadius + 12), FONT, 0.35, MAP_RING_COLOR, 1)
    cv2.drawMarker(mapImage, (robotX, robotY), MAP_ROBOT_COLOR, cv2.MARKER_TRIANGLE_UP, 14, 2)
    for trackedPerson in trackedPeople:
        if not trackedPerson.position_valid:
            continue
        dotX = int(robotX - trackedPerson.position.y * pixelsPerMeter)
        dotY = int(robotY - trackedPerson.position.x * pixelsPerMeter)
        dotColor = LEADER_COLOR if trackedPerson.is_leader else PickColorForTrackId(trackedPerson.id)
        cv2.circle(mapImage, (dotX, dotY), MAP_DOT_RADIUS_PIXELS, dotColor, -1)
        if trackedPerson.velocity.x != 0.0 or trackedPerson.velocity.y != 0.0:
            arrowX = int(dotX - trackedPerson.velocity.y * MAP_ARROW_SECONDS * pixelsPerMeter)
            arrowY = int(dotY - trackedPerson.velocity.x * MAP_ARROW_SECONDS * pixelsPerMeter)
            cv2.arrowedLine(mapImage, (dotX, dotY), (arrowX, arrowY), dotColor, 2, tipLength=0.3)
        cv2.putText(mapImage, f'{trackedPerson.id}', (dotX + 8, dotY + 4), FONT, 0.4, dotColor, 1)
    imageHeight, imageWidth = image.shape[:2]
    mapLeft = imageWidth - MAP_SIZE_PIXELS - MAP_MARGIN_PIXELS
    mapTop = imageHeight - MAP_SIZE_PIXELS - MAP_MARGIN_PIXELS
    mapArea = image[mapTop:mapTop + MAP_SIZE_PIXELS, mapLeft:mapLeft + MAP_SIZE_PIXELS]
    mapArea[:] = cv2.addWeighted(mapImage, MAP_OPACITY, mapArea, 1.0 - MAP_OPACITY, 0)


def DrawLeaderBanner(image, debugFrame):
    if debugFrame is None:
        bannerText, bannerColor = 'LEADER: no debug_frame yet', HEADER_COLOR
    elif (debugFrame.leader_state in (FOLLOWING_STATE, SEARCHING_STATE, LOST_STATE)
          and debugFrame.leader_track_id == NO_TRACK_ID):
        bannerText, bannerColor = DescribeLeaderOutOfFrame(debugFrame), NOTICE_COLOR
    elif debugFrame.leader_state == FOLLOWING_STATE:
        bannerText, bannerColor = DescribeFollowedLeader(debugFrame), LEADER_COLOR
    elif debugFrame.leader_state == ENROLLING_STATE:
        bannerText = f'LEADER: ENROLLING, turn around slowly  {DescribeStoredViews(debugFrame)}'
        bannerColor = STATE_COLORS[ENROLLING_STATE]
    else:
        bannerText = f'LEADER: {STATE_NAMES[debugFrame.leader_state]}'
        bannerColor = STATE_COLORS[debugFrame.leader_state]
    DrawLabel(image, bannerText, 0, BANNER_BOTTOM_PIXELS, bannerColor, 0.6)


def DescribeLeaderOutOfFrame(debugFrame):
    if math.isnan(debugFrame.seconds_since_seen):
        return f'LEADER: {STATE_NAMES[debugFrame.leader_state]}, looking for the saved leader'
    statusText = 'LEADER LOST' if debugFrame.leader_state == LOST_STATE else 'LEADER NOT IN FRAME'
    description = f'{statusText}: last seen {debugFrame.seconds_since_seen:.0f} s ago'
    if not math.isnan(debugFrame.last_seen_position.x):
        description += f' at x {debugFrame.last_seen_position.x:.1f} y {debugFrame.last_seen_position.y:.1f} m'
    return description


def DescribeFollowedLeader(debugFrame):
    description = f'LEADER: FOLLOWING #{debugFrame.leader_track_id}'
    if not math.isnan(debugFrame.leader_distance):
        description += f'  {debugFrame.leader_distance:.1f} m'
    return f'{description}  views {DescribeStoredViews(debugFrame)}'


def DescribeStoredViews(debugFrame):
    return (f'f{debugFrame.front_views} b{debugFrame.back_views} l{debugFrame.side_left_views} '
            f'r{debugFrame.side_right_views} h{debugFrame.partly_hidden_views}')


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
