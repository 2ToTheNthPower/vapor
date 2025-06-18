"""Complete platform state tracking with position, velocity, acceleration, and orientation."""

import numpy as np
from typing import Tuple

from ..trackers.position import PositionTracker
from ..trackers.velocity import VelocityTracker
from ..trackers.acceleration import AccelerationTracker
from ..trackers.orientation import OrientationTracker


class Platform:
    """Complete platform state tracking with position, velocity, acceleration, and orientation"""
    
    def __init__(self):
        self.position = PositionTracker()
        self.velocity = VelocityTracker()
        self.acceleration = AccelerationTracker()
        self.orientation = OrientationTracker()
    
    def set_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        """Set reference point for all coordinate systems"""
        self.position.set_reference_lla(lat, lon, alt)
        self.velocity.set_reference_lla(lat, lon, alt)
        self.acceleration.set_reference_lla(lat, lon, alt)
    
    # Position setters and getters
    def set_position_wcs(self, x: float, y: float, z: float) -> None:
        self.position.set_location_wcs(x, y, z)
    
    def set_position_lla(self, lat: float, lon: float, alt: float) -> None:
        self.position.set_location_lla(lat, lon, alt)
    
    def set_position_ned(self, north: float, east: float, down: float) -> None:
        self.position.set_location_ned(north, east, down)
    
    def get_position_wcs(self) -> Tuple[float, float, float]:
        return self.position.get_location_wcs()
    
    def get_position_lla(self) -> Tuple[float, float, float]:
        return self.position.get_location_lla()
    
    def get_position_ned(self) -> Tuple[float, float, float]:
        return self.position.get_location_ned()
    
    # Velocity setters and getters
    def set_velocity_wcs(self, vx: float, vy: float, vz: float) -> None:
        self.velocity.set_velocity_wcs(vx, vy, vz)
    
    def set_velocity_ned(self, v_north: float, v_east: float, v_down: float) -> None:
        self.velocity.set_velocity_ned(v_north, v_east, v_down)
    
    def set_velocity_lla_rates(self, lat_rate: float, lon_rate: float, alt_rate: float) -> None:
        self.velocity.set_velocity_lla_rates(lat_rate, lon_rate, alt_rate)
    
    def get_velocity_wcs(self) -> Tuple[float, float, float]:
        return self.velocity.get_velocity_wcs()
    
    def get_velocity_ned(self) -> Tuple[float, float, float]:
        return self.velocity.get_velocity_ned()
    
    # Acceleration setters and getters
    def set_acceleration_wcs(self, ax: float, ay: float, az: float) -> None:
        self.acceleration.set_acceleration_wcs(ax, ay, az)
    
    def set_acceleration_ned(self, a_north: float, a_east: float, a_down: float) -> None:
        self.acceleration.set_acceleration_ned(a_north, a_east, a_down)
    
    def set_acceleration_lla_rates(self, lat_accel: float, lon_accel: float, alt_accel: float) -> None:
        self.acceleration.set_acceleration_lla_rates(lat_accel, lon_accel, alt_accel)
    
    def get_acceleration_wcs(self) -> Tuple[float, float, float]:
        return self.acceleration.get_acceleration_wcs()
    
    def get_acceleration_ned(self) -> Tuple[float, float, float]:
        return self.acceleration.get_acceleration_ned()
    
    # Orientation setters and getters
    def set_orientation_euler_ned(self, roll: float, pitch: float, yaw: float) -> None:
        self.orientation.set_orientation_euler_ned(roll, pitch, yaw)
    
    def set_orientation_quaternion_ned(self, w: float, x: float, y: float, z: float) -> None:
        self.orientation.set_orientation_quaternion_ned(w, x, y, z)
    
    def set_orientation_matrix_ned(self, matrix: np.ndarray) -> None:
        self.orientation.set_orientation_matrix_ned(matrix)
    
    def get_orientation_euler_ned(self) -> Tuple[float, float, float]:
        return self.orientation.get_orientation_euler_ned()
    
    def get_orientation_quaternion_ned(self) -> Tuple[float, float, float, float]:
        return self.orientation.get_orientation_quaternion_ned()
    
    def get_orientation_matrix_ned(self) -> np.ndarray:
        return self.orientation.get_orientation_matrix_ned()
    
    def get_relative_state(self, other_platform: 'Platform') -> dict:
        """Get relative state of another platform from this platform's perspective
        
        Returns relative position, velocity, acceleration, and orientation where:
        - x-axis points through the nose of this platform
        - y-axis points through the roof of this platform
        - z-axis completes the right-handed coordinate system
        
        Args:
            other_platform: The platform to get relative state for
            
        Returns:
            Dictionary containing relative position, velocity, acceleration, and orientation
        """
        # Get NED states for both platforms
        self_pos_ned = np.array(self.get_position_ned())
        other_pos_ned = np.array(other_platform.get_position_ned())
        
        self_vel_ned = np.array(self.get_velocity_ned())
        other_vel_ned = np.array(other_platform.get_velocity_ned())
        
        self_accel_ned = np.array(self.get_acceleration_ned())
        other_accel_ned = np.array(other_platform.get_acceleration_ned())
        
        # Calculate relative vectors in NED
        rel_pos_ned = other_pos_ned - self_pos_ned
        rel_vel_ned = other_vel_ned - self_vel_ned
        rel_accel_ned = other_accel_ned - self_accel_ned
        
        # Get this platform's orientation matrix (NED to body frame)
        # The body frame has x=nose, y=roof, z=right-wing
        ned_to_body_matrix = self.get_orientation_matrix_ned().T  # Transpose for NED->Body
        
        # Transform relative vectors from NED to body frame
        rel_pos_body = ned_to_body_matrix @ rel_pos_ned
        rel_vel_body = ned_to_body_matrix @ rel_vel_ned
        rel_accel_body = ned_to_body_matrix @ rel_accel_ned
        
        # Get relative orientation
        self_quat = np.array(self.get_orientation_quaternion_ned())
        other_quat = np.array(other_platform.get_orientation_quaternion_ned())
        
        # Calculate relative quaternion: q_rel = q_self^-1 * q_other
        rel_quat = self._quaternion_multiply(self._quaternion_conjugate(self_quat), other_quat)
        
        return {
            'position': {
                'x': float(rel_pos_body[0]),  # Forward/backward (nose direction)
                'y': float(rel_pos_body[1]),  # Up/down (roof direction) 
                'z': float(rel_pos_body[2])   # Left/right (wing direction)
            },
            'velocity': {
                'x': float(rel_vel_body[0]),
                'y': float(rel_vel_body[1]),
                'z': float(rel_vel_body[2])
            },
            'acceleration': {
                'x': float(rel_accel_body[0]),
                'y': float(rel_accel_body[1]),
                'z': float(rel_accel_body[2])
            },
            'orientation': {
                'quaternion': {
                    'w': float(rel_quat[0]),
                    'x': float(rel_quat[1]),
                    'y': float(rel_quat[2]),
                    'z': float(rel_quat[3])
                }
            }
        }
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Calculate quaternion conjugate"""
        w, x, y, z = q
        return np.array([w, -x, -y, -z])