"""How a person is seen in a photo, from the YOLO-pose keypoints. Plain python, no ROS, no model.

- IsCleanView: the photo is good enough to compare (not cut at the left/right image edge, tall enough,
  some body part visible). A box at the edge is still clean when shoulders and hips are visible inside
  the image (only an arm is cut). A partly hidden person is still clean.
- FindViewGroup: which group of stored leader views the photo belongs to: front, back, side left,
  side right (whole torso visible) or partly hidden (only the upper or only the lower body visible).
"""

# Parameter defaults (set in config/perception_params.yaml)
BODY_VIEW_SETTING_DEFAULTS = {
    'keypointConfidence': 0.5,
    'sideRatio': 0.35,
    'edgeMarginPixels': 8,
    'minimumCropHeightPixels': 96,
}

# Body keypoints (COCO order of the YOLO pose model)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14

# Visible body parts
FULL_BODY = 'full'
UPPER_BODY = 'upper'
LOWER_BODY = 'lower'
NO_BODY = 'none'

# View groups (the same order as the view counts in PerceptionLeaderState)
FRONT_GROUP = 0
BACK_GROUP = 1
SIDE_LEFT_GROUP = 2
SIDE_RIGHT_GROUP = 3
PARTLY_HIDDEN_GROUP = 4
UNKNOWN_GROUP = -1
FACING_GROUPS = (FRONT_GROUP, BACK_GROUP, SIDE_LEFT_GROUP, SIDE_RIGHT_GROUP)
VIEW_GROUPS = FACING_GROUPS + (PARTLY_HIDDEN_GROUP,)


def IsCleanView(person, imageWidth, settings):
    boxInsideImage = person.left > settings.edgeMarginPixels and person.right < imageWidth - settings.edgeMarginPixels
    tallEnough = person.bottom - person.top >= settings.minimumCropHeightPixels
    visibleBodyPart = FindVisibleBodyPart(person, settings)
    if not boxInsideImage and not (visibleBodyPart == FULL_BODY and IsTorsoInsideImage(person, imageWidth, settings)):
        return False
    return tallEnough and visibleBodyPart != NO_BODY


def IsTorsoInsideImage(person, imageWidth, settings):
    torsoX = person.bodyKeypoints[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], 0]
    return torsoX.min() > settings.edgeMarginPixels and torsoX.max() < imageWidth - settings.edgeMarginPixels


def FindViewGroup(person, settings):
    visibleBodyPart = FindVisibleBodyPart(person, settings)
    if visibleBodyPart == FULL_BODY:
        return FindFacingGroup(person, settings)
    if visibleBodyPart in (UPPER_BODY, LOWER_BODY):
        return PARTLY_HIDDEN_GROUP
    return UNKNOWN_GROUP


def FindVisibleBodyPart(person, settings):
    shouldersVisible = ArePointsVisible(person, (LEFT_SHOULDER, RIGHT_SHOULDER), settings)
    hipsVisible = ArePointsVisible(person, (LEFT_HIP, RIGHT_HIP), settings)
    kneesVisible = ArePointsVisible(person, (LEFT_KNEE, RIGHT_KNEE), settings)
    if shouldersVisible and hipsVisible:
        return FULL_BODY
    if shouldersVisible:
        return UPPER_BODY
    if hipsVisible or kneesVisible:
        return LOWER_BODY
    return NO_BODY


def FindFacingGroup(person, settings):
    keypoints = person.bodyKeypoints
    faceVisible = max(keypoints[NOSE, 2], keypoints[LEFT_EYE, 2], keypoints[RIGHT_EYE, 2]) >= settings.keypointConfidence
    shoulderOffset = keypoints[LEFT_SHOULDER, 0] - keypoints[RIGHT_SHOULDER, 0]
    torsoHeight = (keypoints[LEFT_HIP, 1] + keypoints[RIGHT_HIP, 1] - keypoints[LEFT_SHOULDER, 1] - keypoints[RIGHT_SHOULDER, 1]) / 2
    if abs(shoulderOffset) < settings.sideRatio * torsoHeight:
        if keypoints[NOSE, 2] < settings.keypointConfidence:
            return UNKNOWN_GROUP
        shoulderCenterX = (keypoints[LEFT_SHOULDER, 0] + keypoints[RIGHT_SHOULDER, 0]) / 2
        return SIDE_LEFT_GROUP if keypoints[NOSE, 0] < shoulderCenterX else SIDE_RIGHT_GROUP
    if shoulderOffset > 0 and faceVisible:
        return FRONT_GROUP
    if shoulderOffset < 0 and not faceVisible:
        return BACK_GROUP
    return UNKNOWN_GROUP


def ArePointsVisible(person, keypointIndices, settings):
    return all(person.bodyKeypoints[keypointIndex, 2] >= settings.keypointConfidence for keypointIndex in keypointIndices)
