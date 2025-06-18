"""Orientation tracking in NED coordinate system."""

import math
import numpy as np
from typing import Tuple


class OrientationTracker:
    """Tracks orientation in NED coordinate system using Euler angles, quaternions, and rotation matrices"""
    
    def __init__(self):
        self._euler_ned = None      # (roll, pitch, yaw) in degrees
        self._quaternion_ned = None # (w, x, y, z) normalized quaternion
        self._matrix_ned = None     # 3x3 rotation matrix
    
    def set_orientation_euler_ned(self, roll: float, pitch: float, yaw: float) -> None:
        """Set orientation using Euler angles in NED frame (degrees)"""
        self._euler_ned = (roll, pitch, yaw)
        # Convert to quaternion and matrix for internal consistency
        self._quaternion_ned = self._euler_to_quaternion(roll, pitch, yaw)
        self._matrix_ned = self._quaternion_to_matrix(self._quaternion_ned)
    
    def set_orientation_quaternion_ned(self, w: float, x: float, y: float, z: float) -> None:
        """Set orientation using quaternion in NED frame"""
        # Normalize quaternion
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            raise ValueError("Quaternion cannot have zero magnitude")
        self._quaternion_ned = (w/norm, x/norm, y/norm, z/norm)
        # Convert to Euler and matrix for internal consistency
        self._euler_ned = self._quaternion_to_euler(self._quaternion_ned)
        self._matrix_ned = self._quaternion_to_matrix(self._quaternion_ned)
    
    def set_orientation_matrix_ned(self, matrix: np.ndarray) -> None:
        """Set orientation using rotation matrix in NED frame"""
        if matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        if not np.allclose(np.dot(matrix, matrix.T), np.eye(3), atol=1e-6):
            raise ValueError("Matrix must be orthogonal")
        if not np.allclose(np.linalg.det(matrix), 1.0, atol=1e-6):
            raise ValueError("Matrix must have determinant of 1")
        
        self._matrix_ned = matrix.copy()
        # Convert to quaternion and Euler for internal consistency
        self._quaternion_ned = self._matrix_to_quaternion(matrix)
        self._euler_ned = self._quaternion_to_euler(self._quaternion_ned)
    
    def get_orientation_euler_ned(self) -> Tuple[float, float, float]:
        """Get orientation as Euler angles in NED frame (degrees)"""
        if self._euler_ned is None:
            raise ValueError("Orientation not set")
        return self._euler_ned
    
    def get_orientation_quaternion_ned(self) -> Tuple[float, float, float, float]:
        """Get orientation as quaternion in NED frame (w, x, y, z)"""
        if self._quaternion_ned is None:
            raise ValueError("Orientation not set")
        return self._quaternion_ned
    
    def get_orientation_matrix_ned(self) -> np.ndarray:
        """Get orientation as rotation matrix in NED frame"""
        if self._matrix_ned is None:
            raise ValueError("Orientation not set")
        return self._matrix_ned.copy()
    
    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        """Convert Euler angles (degrees) to quaternion"""
        # Convert to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        # Half angles
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        
        # Quaternion multiplication (ZYX order)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)
    
    def _quaternion_to_euler(self, quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (degrees)"""
        w, x, y, z = quaternion
        
        # Roll (rotation around x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (rotation around y-axis)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (rotation around z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    
    def _quaternion_to_matrix(self, quaternion: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quaternion
        
        # Rotation matrix from quaternion
        matrix = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return matrix
    
    def _matrix_to_quaternion(self, matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert rotation matrix to quaternion"""
        trace = np.trace(matrix)
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
        
        return (w, x, y, z)