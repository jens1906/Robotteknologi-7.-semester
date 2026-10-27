"""Pixel + depth -> 3D position in base_link. Uses ROS libraries (image_geometry, tf2_geometry_msgs, cv_bridge),
creates no topics.

To include:
- Camera model from camera_info (image_geometry.PinholeCameraModel): pixel (u, v), lens distortion removed,
  + depth z -> point in the camera optical frame.
- Camera optical frame -> base_link with the mounting transform (TransformStamped looked up by
  perception_node, applied with tf2_geometry_msgs).
- A person's depth: median of small patches around the visible torso keypoints (shoulders, hips), not the
  box centre (that often hits the background). Fallback: the middle strip of the box (partly hidden
  person). Ignore 0 / out-of-range depth (depthMin..depthMax); no usable depth, or less than
  minimumValidDepthShare of the sampled pixels usable (person closer than depthMin) -> position_valid = false.
- Size (width, height) from the box and the depth; uncertainty from the spread of the depth samples.
- Horizontal distance of the leader for the debug frame.
- Later: depth image -> points for obstacle_mapper's fallback when there is no lidar.
"""

import math
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Point, PointStamped
from image_geometry import PinholeCameraModel
from tf2_geometry_msgs import do_transform_point

from perception.body_view import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

# Parameter defaults (set in config/perception_params.yaml)
COORDINATE_SETTING_DEFAULTS = {
    'depthMin': 0.3,
    'depthMax': 8.0,
    'depthPatchPixels': 3,
    'minimumDepthSamples': 5,
    'minimumValidDepthShare': 0.3,
}

# Depth image
DEPTH_SCALES = {'16UC1': 0.001, 'mono16': 0.001, '32FC1': 1.0}
TORSO_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# Fallback strip of the box (shares of its width and height)
STRIP_LEFT_SHARE = 1.0 / 3.0
STRIP_RIGHT_SHARE = 2.0 / 3.0
STRIP_TOP_SHARE = 0.25
STRIP_BOTTOM_SHARE = 0.6

# Median absolute deviation -> one standard deviation
MEDIAN_DEVIATION_TO_SIGMA = 1.4826


@dataclass
class PersonPosition:
    position: Point
    width: float
    height: float
    uncertainty: float


def ConvertDepthImage(depthImageMessage, imageBridge):
    depthImage = imageBridge.imgmsg_to_cv2(depthImageMessage, 'passthrough')
    return depthImage.astype(np.float32) * DEPTH_SCALES[depthImageMessage.encoding]


def BuildCameraModel(cameraInfoMessage):
    cameraModel = PinholeCameraModel()
    cameraModel.from_camera_info(cameraInfoMessage)
    return cameraModel


def PlacePerson(person, depthMeters, cameraModel, opticalToBaseTransform, settings):
    measuredDepth = MeasurePersonDepth(person, depthMeters, settings)
    if measuredDepth is None:
        return None
    pixelX, pixelY, depth, spread = measuredDepth
    rayX, rayY, rayZ = cameraModel.project_pixel_to_3d_ray(cameraModel.rectify_point((pixelX, pixelY)))
    opticalPoint = PointStamped()
    opticalPoint.header.frame_id = opticalToBaseTransform.child_frame_id
    opticalPoint.point.x = rayX / rayZ * depth
    opticalPoint.point.y = rayY / rayZ * depth
    opticalPoint.point.z = depth
    basePoint = do_transform_point(opticalPoint, opticalToBaseTransform)
    width = (person.right - person.left) * depth / cameraModel.fx()
    height = (person.bottom - person.top) * depth / cameraModel.fy()
    return PersonPosition(basePoint.point, width, height, spread)


def MeasurePersonDepth(person, depthMeters, settings):
    keypointDepths = []
    keypointXs = []
    keypointYs = []
    sampledPixelCount = 0
    for keypointIndex in TORSO_KEYPOINTS:
        keypointX, keypointY, keypointConfidence = person.bodyKeypoints[keypointIndex]
        if keypointConfidence < settings.keypointConfidence:
            continue
        patch = CutPatch(depthMeters, keypointX, keypointY, settings.depthPatchPixels)
        sampledPixelCount += patch.size
        patchDepths = KeepValidDepths(patch, settings)
        if patchDepths.size:
            keypointDepths.append(patchDepths)
            keypointXs.append(keypointX)
            keypointYs.append(keypointY)
    depthSamples = np.concatenate(keypointDepths) if keypointDepths else np.zeros(0, np.float32)
    if depthSamples.size >= settings.minimumDepthSamples:
        pixelX, pixelY = float(np.mean(keypointXs)), float(np.mean(keypointYs))
    else:
        boxStrip = CutBoxStrip(depthMeters, person)
        sampledPixelCount = boxStrip.size
        depthSamples = KeepValidDepths(boxStrip, settings)
        pixelX = (person.left + person.right) / 2
        pixelY = person.top + (person.bottom - person.top) * (STRIP_TOP_SHARE + STRIP_BOTTOM_SHARE) / 2
    if depthSamples.size < settings.minimumDepthSamples:
        return None
    if depthSamples.size < settings.minimumValidDepthShare * sampledPixelCount:
        return None
    depth = float(np.median(depthSamples))
    spread = MEDIAN_DEVIATION_TO_SIGMA * float(np.median(np.abs(depthSamples - depth)))
    return pixelX, pixelY, depth, spread


def CutPatch(depthMeters, centerX, centerY, patchPixels):
    imageHeight, imageWidth = depthMeters.shape
    left, right = max(0, int(centerX) - patchPixels), min(imageWidth, int(centerX) + patchPixels + 1)
    top, bottom = max(0, int(centerY) - patchPixels), min(imageHeight, int(centerY) + patchPixels + 1)
    return depthMeters[top:bottom, left:right]


def CutBoxStrip(depthMeters, person):
    imageHeight, imageWidth = depthMeters.shape
    boxWidth, boxHeight = person.right - person.left, person.bottom - person.top
    left = max(0, int(person.left + boxWidth * STRIP_LEFT_SHARE))
    right = min(imageWidth, int(person.left + boxWidth * STRIP_RIGHT_SHARE))
    top = max(0, int(person.top + boxHeight * STRIP_TOP_SHARE))
    bottom = min(imageHeight, int(person.top + boxHeight * STRIP_BOTTOM_SHARE))
    return depthMeters[top:bottom, left:right]


def KeepValidDepths(depthValues, settings):
    flatDepths = depthValues.ravel()
    return flatDepths[(flatDepths >= settings.depthMin) & (flatDepths <= settings.depthMax)]


def CalculateHorizontalDistance(position):
    return math.hypot(position.x, position.y)
