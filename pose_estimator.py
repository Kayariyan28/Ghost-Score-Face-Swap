"""6-DOF head pose estimation via OpenCV solvePnP.

Uses InsightFace's 5 keypoints (left-eye, right-eye, nose-tip, left-mouth-corner,
right-mouth-corner) against a canonical 3D face model. Returns yaw / pitch /
roll in degrees.

This is a standalone module — it does NOT modify any existing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Canonical 3D face model (in arbitrary world units — only ratios matter for
# pose). Derived from the standard Multi-PIE / BFM2009 face model used by
# OpenFace and dlib. Coordinate frame: +X right, +Y down, +Z forward (into
# the screen from the camera's POV).
CANONICAL_5_KPS_3D = np.array([
    [-30.0, -30.0,  30.0],  # left eye outer
    [ 30.0, -30.0,  30.0],  # right eye outer
    [  0.0,   0.0,   0.0],  # nose tip (origin)
    [-25.0,  30.0,  20.0],  # left mouth corner
    [ 25.0,  30.0,  20.0],  # right mouth corner
], dtype=np.float64)


@dataclass
class PoseAngles:
    yaw: float    # left/right rotation, degrees. +ve = head turned RIGHT.
    pitch: float  # up/down tilt, degrees. +ve = head tilted DOWN.
    roll: float   # in-plane rotation, degrees. +ve = clockwise.

    @property
    def magnitude(self) -> float:
        """Largest single-axis deviation from frontal."""
        return float(max(abs(self.yaw), abs(self.pitch), abs(self.roll)))

    def to_dict(self):
        return {'yaw': round(self.yaw, 1), 'pitch': round(self.pitch, 1),
                'roll': round(self.roll, 1), 'magnitude': round(self.magnitude, 1)}


def estimate_pose(image: np.ndarray, face) -> PoseAngles:
    """Return head pose for an InsightFace face object.

    Uses the 5 detection keypoints in face.kps and solves PnP against the
    canonical 3D model. Falls back to zeros if PnP fails.
    """
    h, w = image.shape[:2]
    focal_length = float(max(w, h))
    K = np.array([
        [focal_length, 0, w / 2.0],
        [0, focal_length, h / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    kps_2d = np.asarray(face.kps, dtype=np.float64).reshape(-1, 2)
    if kps_2d.shape[0] < 5:
        return PoseAngles(0.0, 0.0, 0.0)

    # ITERATIVE needs ≥6 points (uses DLT for the initial guess); we only
    # have 5. EPNP works with ≥4 points and is fine for pose estimation.
    success, rvec, _tvec = cv2.solvePnP(
        CANONICAL_5_KPS_3D, kps_2d[:5], K, dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return PoseAngles(0.0, 0.0, 0.0)

    rmat, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])  # x-axis
        yaw = np.arctan2(-rmat[2, 0], sy)           # y-axis
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])   # z-axis
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    return PoseAngles(
        yaw=float(np.degrees(yaw)),
        pitch=float(np.degrees(pitch)),
        roll=float(np.degrees(roll)),
    )


def pose_diff_magnitude(a: PoseAngles, b: PoseAngles) -> float:
    """Maximum single-axis difference between two poses (degrees)."""
    return float(max(
        abs(a.yaw - b.yaw),
        abs(a.pitch - b.pitch),
        abs(a.roll - b.roll),
    ))
