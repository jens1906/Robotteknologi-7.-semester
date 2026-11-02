"""Camera depth cloud -> Go2 lidar cloud registration for calibrate_camera. Plain numpy + OpenCV, no ROS.

- Camera points: per-pixel median of the aligned depth frames, rays from camera_info (lens distortion removed),
  averaged per cube of voxelSize. In the camera optical frame.
- Lidar: the stacked clouds (base_link) cut into cubes ("patches"); every flat patch gets a plane (centre + normal).
- Point-to-plane ICP: each camera point is pulled onto the plane of the nearest patch centre among the 27
  neighbouring cubes, patch size coarse to fine (patchSizes), Huber weights. Result: base_link <- optical frame.
- Fit quality on the finest patches (FindQualityProblems also refuses a big jump from the saved mount): residual (robust sigma), matched points, direction points (smallest eigenvalue of
  the matched normals' scatter, ~ points on surfaces facing weakestDirection: 0 = the camera can slide that way).
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Camera points
MINIMUM_VALID_FRAME_SHARE = 0.5

# Lidar patches
PATCH_FLATNESS = 0.1
MINIMUM_PATCH_SPREAD = 0.1
MINIMUM_PATCH_POINTS = 10
NEIGHBOUR_OFFSETS = np.array([[offsetX, offsetY, offsetZ] for offsetX in (-1, 0, 1) for offsetY in (-1, 0, 1)
                              for offsetZ in (-1, 0, 1)], dtype=np.int64)

# Cube keys
KEY_OFFSET = 1 << 19
KEY_SPAN = 1 << 20

# Point-to-plane alignment
ROBUST_WIDTH = 0.02
MAXIMUM_ITERATIONS = 20
MINIMUM_STEP_POINTS = 6
SMALLEST_STEP = 1e-6
PSEUDO_INVERSE_CUTOFF = 1e-6
MEDIAN_DEVIATION_TO_SIGMA = 1.4826


@dataclass
class CameraMount:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class PatchMap:
    patchSize: float
    keys: np.ndarray
    centers: np.ndarray
    normals: np.ndarray


@dataclass
class FitQuality:
    residual: float
    matchedPoints: int
    directionPoints: float
    weakestDirection: np.ndarray


@dataclass
class RegistrationResult:
    transform: np.ndarray
    startQuality: FitQuality
    quality: FitQuality
    cameraPointCount: int
    lidarPointCount: int


def BuildCameraPoints(depthFrames, cameraMatrix, distortion, settings):
    depthStack = np.stack(depthFrames)
    validStack = (depthStack >= settings.depthMin) & (depthStack <= settings.depthMax)
    usedPixels = validStack.sum(axis=0) >= MINIMUM_VALID_FRAME_SHARE * len(depthFrames)
    if not usedPixels.any():
        return np.zeros((0, 3))
    pixelRows, pixelColumns = np.nonzero(usedPixels)
    depths = np.nanmedian(np.where(validStack[:, usedPixels], depthStack[:, usedPixels], np.nan), axis=0)
    pixels = np.column_stack([pixelColumns, pixelRows]).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(pixels, cameraMatrix, distortion).reshape(-1, 2)
    return DownsampleVoxels(np.column_stack([rays * depths[:, None], depths]), settings.voxelSize)


def StackLidarPoints(lidarClouds, settings):
    lidarPoints = np.concatenate(lidarClouds).astype(np.float64)
    lidarRanges = np.linalg.norm(lidarPoints, axis=1)
    return lidarPoints[(lidarRanges >= settings.depthMin) & (lidarRanges <= settings.depthMax + max(settings.patchSizes))]


def DownsampleVoxels(points, voxelSize):
    if len(points) == 0:
        return points
    _, voxelIndices, counts = np.unique(MakeCubeKeys(points, voxelSize), return_inverse=True, return_counts=True)
    return SumPerCube(points, voxelIndices.ravel(), len(counts)) / counts[:, None]


def MakeCubeKeys(points, cubeSize):
    return CombineCellKeys(np.floor(points / cubeSize).astype(np.int64))


def CombineCellKeys(cells):
    shiftedCells = cells + KEY_OFFSET
    return (shiftedCells[:, 0] * KEY_SPAN + shiftedCells[:, 1]) * KEY_SPAN + shiftedCells[:, 2]


def SumPerCube(values, cubeIndices, cubeCount):
    return np.column_stack([np.bincount(cubeIndices, weights=column, minlength=cubeCount) for column in values.T])


def BuildPatchMap(lidarPoints, patchSize):
    keys, patchIndices, counts = np.unique(MakeCubeKeys(lidarPoints, patchSize), return_inverse=True,
                                           return_counts=True)
    patchIndices = patchIndices.ravel()
    centers = SumPerCube(lidarPoints, patchIndices, len(keys)) / counts[:, None]
    offsets = lidarPoints - centers[patchIndices]
    outerProducts = (offsets[:, :, None] * offsets[:, None, :]).reshape(-1, 9)
    covariances = (SumPerCube(outerProducts, patchIndices, len(keys)) / counts[:, None]).reshape(-1, 3, 3)
    eigenValues, eigenVectors = np.linalg.eigh(covariances)
    spreadSums = np.maximum(eigenValues.sum(axis=1), SMALLEST_STEP)
    flat = ((counts >= MINIMUM_PATCH_POINTS) & (eigenValues[:, 0] <= PATCH_FLATNESS * spreadSums)
            & (eigenValues[:, 1] >= MINIMUM_PATCH_SPREAD * spreadSums))
    return PatchMap(patchSize, keys[flat], centers[flat], eigenVectors[flat, :, 0])


def FindNearestPatches(patchMap, points):
    nearestPatches = np.zeros(len(points), dtype=np.int64)
    nearestDistances = np.full(len(points), np.inf)
    if len(patchMap.keys) == 0:
        return nearestPatches, nearestDistances
    pointCells = np.floor(points / patchMap.patchSize).astype(np.int64)
    for neighbourOffset in NEIGHBOUR_OFFSETS:
        neighbourKeys = CombineCellKeys(pointCells + neighbourOffset)
        patchPositions = np.minimum(np.searchsorted(patchMap.keys, neighbourKeys), len(patchMap.keys) - 1)
        found = patchMap.keys[patchPositions] == neighbourKeys
        distances = np.where(found, np.linalg.norm(points - patchMap.centers[patchPositions], axis=1), np.inf)
        closer = distances < nearestDistances
        nearestPatches[closer] = patchPositions[closer]
        nearestDistances[closer] = distances[closer]
    return nearestPatches, nearestDistances


def MatchCameraPoints(cameraPoints, patchMap, transform):
    movedPoints = TransformPoints(transform, cameraPoints)
    nearestPatches, nearestDistances = FindNearestPatches(patchMap, movedPoints)
    matched = nearestDistances <= patchMap.patchSize
    matchedPatches = nearestPatches[matched]
    normals = patchMap.normals[matchedPatches]
    residuals = np.einsum('ij,ij->i', movedPoints[matched] - patchMap.centers[matchedPatches], normals)
    return movedPoints[matched], normals, residuals


def AlignCameraToLidar(cameraPoints, lidarPoints, startTransform, settings):
    finePatchMap = BuildPatchMap(lidarPoints, settings.patchSizes[-1])
    startQuality = MeasureFit(cameraPoints, finePatchMap, startTransform)
    transform = startTransform
    for patchSize in settings.patchSizes:
        transform = AlignOnPatches(cameraPoints, BuildPatchMap(lidarPoints, patchSize), transform)
    return RegistrationResult(transform, startQuality, MeasureFit(cameraPoints, finePatchMap, transform),
                              len(cameraPoints), len(lidarPoints))


def AlignOnPatches(cameraPoints, patchMap, transform):
    for iteration in range(MAXIMUM_ITERATIONS):
        matchedPoints, normals, residuals = MatchCameraPoints(cameraPoints, patchMap, transform)
        if len(residuals) < MINIMUM_STEP_POINTS:
            break
        jacobian = np.hstack([np.cross(matchedPoints, normals), normals])
        weights = np.minimum(1.0, ROBUST_WIDTH / np.maximum(np.abs(residuals), SMALLEST_STEP))
        weightedJacobian = jacobian * weights[:, None]
        step = -np.linalg.pinv(weightedJacobian.T @ jacobian, rcond=PSEUDO_INVERSE_CUTOFF) @ (weightedJacobian.T @ residuals)
        transform = ApplyStep(step, transform)
        if np.linalg.norm(step) < SMALLEST_STEP:
            break
    return transform


def ApplyStep(step, transform):
    stepRotation, _ = cv2.Rodrigues(step[:3])
    stepTransform = np.eye(4)
    stepTransform[:3, :3] = stepRotation
    stepTransform[:3, 3] = step[3:]
    return stepTransform @ transform


def MeasureFit(cameraPoints, patchMap, transform):
    matchedPoints, normals, residuals = MatchCameraPoints(cameraPoints, patchMap, transform)
    if len(residuals) == 0:
        return FitQuality(math.inf, 0, 0.0, np.zeros(3))
    residual = MEDIAN_DEVIATION_TO_SIGMA * float(np.median(np.abs(residuals)))
    eigenValues, eigenVectors = np.linalg.eigh(normals.T @ normals)
    return FitQuality(residual, len(residuals), float(eigenValues[0]), eigenVectors[:, 0])


def FindQualityProblems(result, changeDistance, changeAngle, settings):
    quality = result.quality
    qualityProblems = []
    if changeDistance > settings.maximumChangeDistance or changeAngle > settings.maximumChangeAngle:
        qualityProblems.append(f'moved {changeDistance * 1000:.0f} mm / {math.degrees(changeAngle):.1f} deg from the saved '
                               f'mount (> {settings.maximumChangeDistance * 1000:.0f} mm / '
                               f'{math.degrees(settings.maximumChangeAngle):.0f} deg): wrong match or the saved mount is '
                               'far off, measure it again roughly')
    if quality.residual > settings.maximumResidual:
        qualityProblems.append(f'residual {quality.residual * 1000:.0f} mm > {settings.maximumResidual * 1000:.0f} mm '
                               '(robot moved, something moved in view, or a wrong start mount)')
    if quality.matchedPoints < settings.minimumMatchedPoints:
        qualityProblems.append(f'only {quality.matchedPoints} camera points on lidar surfaces '
                               f'(< {settings.minimumMatchedPoints}): more wall in view at ~3 m')
    if quality.directionPoints < settings.minimumDirectionPoints:
        weakestX, weakestY, weakestZ = quality.weakestDirection
        qualityProblems.append(f'the scene does not hold the camera along ({weakestX:.2f}, {weakestY:.2f}, '
                               f'{weakestZ:.2f}) in base_link ({quality.directionPoints:.0f} direction points < '
                               f'{settings.minimumDirectionPoints}): add a surface facing that way in view of both '
                               'sensors (up/down: lean the board against a wall)')
    return qualityProblems


def TransformPoints(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def MakeMountTransform(cameraMount):
    return MakeTransform(MakeRotationFromRollPitchYaw(cameraMount.roll, cameraMount.pitch, cameraMount.yaw),
                         (cameraMount.x, cameraMount.y, cameraMount.z))


def MakeCameraMount(transform):
    rotation = transform[:3, :3]
    return CameraMount(float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]),
                       math.atan2(rotation[2, 1], rotation[2, 2]),
                       math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2])),
                       math.atan2(rotation[1, 0], rotation[0, 0]))


def MakeRotationFromRollPitchYaw(roll, pitch, yaw):
    cosRoll, sinRoll = math.cos(roll), math.sin(roll)
    cosPitch, sinPitch = math.cos(pitch), math.sin(pitch)
    cosYaw, sinYaw = math.cos(yaw), math.sin(yaw)
    rollRotation = np.array([[1.0, 0.0, 0.0], [0.0, cosRoll, -sinRoll], [0.0, sinRoll, cosRoll]])
    pitchRotation = np.array([[cosPitch, 0.0, sinPitch], [0.0, 1.0, 0.0], [-sinPitch, 0.0, cosPitch]])
    yawRotation = np.array([[cosYaw, -sinYaw, 0.0], [sinYaw, cosYaw, 0.0], [0.0, 0.0, 1.0]])
    return yawRotation @ pitchRotation @ rollRotation


def MakeRotationFromQuaternion(quaternionX, quaternionY, quaternionZ, quaternionW):
    return np.array([
        [1 - 2 * (quaternionY ** 2 + quaternionZ ** 2), 2 * (quaternionX * quaternionY - quaternionZ * quaternionW),
         2 * (quaternionX * quaternionZ + quaternionY * quaternionW)],
        [2 * (quaternionX * quaternionY + quaternionZ * quaternionW), 1 - 2 * (quaternionX ** 2 + quaternionZ ** 2),
         2 * (quaternionY * quaternionZ - quaternionX * quaternionW)],
        [2 * (quaternionX * quaternionZ - quaternionY * quaternionW),
         2 * (quaternionY * quaternionZ + quaternionX * quaternionW), 1 - 2 * (quaternionX ** 2 + quaternionY ** 2)]])


def MakeTransform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def MeasureMountChange(oldMount, newMount):
    oldTransform = MakeMountTransform(oldMount)
    newTransform = MakeMountTransform(newMount)
    relativeRotation = oldTransform[:3, :3].T @ newTransform[:3, :3]
    rotationAngle = math.acos(min(1.0, max(-1.0, (np.trace(relativeRotation) - 1.0) / 2.0)))
    return float(np.linalg.norm(newTransform[:3, 3] - oldTransform[:3, 3])), rotationAngle
