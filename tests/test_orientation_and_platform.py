import pytest
import numpy as np
import math
from vapor import OrientationTracker, Platform


class TestOrientationTracker:
    def test_euler_angles_ned(self):
        """Test setting and getting Euler angles in NED frame"""
        tracker = OrientationTracker()
        
        # Set orientation using Euler angles (roll, pitch, yaw in degrees)
        tracker.set_orientation_euler_ned(roll=10.0, pitch=5.0, yaw=90.0)
        
        # Get orientation back
        roll, pitch, yaw = tracker.get_orientation_euler_ned()
        
        # Verify angles are preserved
        assert abs(roll - 10.0) < 1e-6
        assert abs(pitch - 5.0) < 1e-6
        assert abs(yaw - 90.0) < 1e-6
        
    def test_quaternion_ned(self):
        """Test setting and getting quaternion in NED frame"""
        tracker = OrientationTracker()
        
        # Set orientation using quaternion (w, x, y, z)
        tracker.set_orientation_quaternion_ned(w=1.0, x=0.0, y=0.0, z=0.0)
        
        # Get quaternion back
        w, x, y, z = tracker.get_orientation_quaternion_ned()
        
        # Verify quaternion is normalized and preserved
        assert abs(w - 1.0) < 1e-6
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(z - 0.0) < 1e-6
        assert abs(w*w + x*x + y*y + z*z - 1.0) < 1e-6
        
    def test_rotation_matrix_ned(self):
        """Test setting and getting rotation matrix in NED frame"""
        tracker = OrientationTracker()
        
        # Set identity rotation matrix
        identity = np.eye(3)
        tracker.set_orientation_matrix_ned(identity)
        
        # Get rotation matrix back
        matrix = tracker.get_orientation_matrix_ned()
        
        # Verify matrix is preserved and orthogonal
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, identity)
        assert np.allclose(np.dot(matrix, matrix.T), np.eye(3))
        
    def test_euler_to_quaternion_conversion(self):
        """Test conversion from Euler angles to quaternion"""
        tracker = OrientationTracker()
        
        # Set Euler angles
        tracker.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        # Get as quaternion
        w, x, y, z = tracker.get_orientation_quaternion_ned()
        
        # For 90 degree yaw rotation, expect specific quaternion values
        expected_w = math.cos(math.radians(45))  # cos(yaw/2)
        expected_z = math.sin(math.radians(45))  # sin(yaw/2) for yaw rotation
        
        assert abs(w - expected_w) < 1e-6
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(z - expected_z) < 1e-6
        
    def test_quaternion_to_matrix_conversion(self):
        """Test conversion from quaternion to rotation matrix"""
        tracker = OrientationTracker()
        
        # Set identity quaternion
        tracker.set_orientation_quaternion_ned(w=1.0, x=0.0, y=0.0, z=0.0)
        
        # Get as rotation matrix
        matrix = tracker.get_orientation_matrix_ned()
        
        # Should be identity matrix
        assert np.allclose(matrix, np.eye(3))


class TestPlatform:
    def test_platform_initialization(self):
        """Test Platform object initialization"""
        platform = Platform()
        
        # Set reference point
        platform.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Verify platform has all required attributes
        assert hasattr(platform, 'position')
        assert hasattr(platform, 'velocity')
        assert hasattr(platform, 'acceleration')
        assert hasattr(platform, 'orientation')
        
    def test_platform_setters_and_getters(self):
        """Test Platform setters and getters for all attributes"""
        platform = Platform()
        platform.set_reference_lla(lat=0.0, lon=0.0, alt=0.0)
        
        # Test position
        platform.set_position_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        lat, lon, alt = platform.get_position_lla()
        assert abs(lat - 37.7749) < 1e-6
        assert abs(lon + 122.4194) < 1e-6
        assert abs(alt - 100.0) < 1e-3
        
        # Test velocity
        platform.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=-2.0)
        v_n, v_e, v_d = platform.get_velocity_ned()
        assert abs(v_n - 10.0) < 1e-6
        assert abs(v_e - 5.0) < 1e-6
        assert abs(v_d + 2.0) < 1e-6
        
        # Test acceleration
        platform.set_acceleration_ned(a_north=1.0, a_east=0.5, a_down=0.1)
        a_n, a_e, a_d = platform.get_acceleration_ned()
        assert abs(a_n - 1.0) < 1e-6
        assert abs(a_e - 0.5) < 1e-6
        assert abs(a_d - 0.1) < 1e-6
        
        # Test orientation
        platform.set_orientation_euler_ned(roll=10.0, pitch=5.0, yaw=90.0)
        roll, pitch, yaw = platform.get_orientation_euler_ned()
        assert abs(roll - 10.0) < 1e-6
        assert abs(pitch - 5.0) < 1e-6
        assert abs(yaw - 90.0) < 1e-6

    def test_platform_set_position_ned(self):
        """Test Platform's set_position_ned method"""
        platform = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set position using NED coordinates
        north, east, down = 1000.0, 500.0, -200.0
        platform.set_position_ned(north=north, east=east, down=down)
        
        # Verify the position is set correctly in NED
        retrieved_ned = platform.get_position_ned()
        assert abs(retrieved_ned[0] - north) < 1e-3
        assert abs(retrieved_ned[1] - east) < 1e-3
        assert abs(retrieved_ned[2] - down) < 1e-3
        
        # Verify it can be retrieved in other coordinate systems
        lla_pos = platform.get_position_lla()
        wcs_pos = platform.get_position_wcs()
        
        assert isinstance(lla_pos[0], float)
        assert isinstance(lla_pos[1], float)
        assert isinstance(lla_pos[2], float)
        assert isinstance(wcs_pos[0], float)
        assert isinstance(wcs_pos[1], float)
        assert isinstance(wcs_pos[2], float)

    def test_platform_set_position_ned_aircraft_example(self):
        """Test Platform's set_position_ned with aircraft example values"""
        platform = Platform()
        
        # Set reference point (airport)
        airport_lat, airport_lon, airport_alt = 37.7749, -122.4194, 100.0
        platform.set_reference_lla(airport_lat, airport_lon, airport_alt)
        
        # Set aircraft position: 5km north, 2km east, 914m up (like main.py example)
        platform.set_position_ned(north=5000.0, east=2000.0, down=-914.0)
        
        # Verify NED position
        ned_pos = platform.get_position_ned()
        assert abs(ned_pos[0] - 5000.0) < 1e-3
        assert abs(ned_pos[1] - 2000.0) < 1e-3
        assert abs(ned_pos[2] + 914.0) < 1e-3
        
        # Verify LLA position is reasonable
        lla_pos = platform.get_position_lla()
        assert lla_pos[0] > airport_lat  # North of reference
        assert lla_pos[1] > airport_lon  # East of reference
        assert lla_pos[2] > airport_alt  # Above reference

    def test_platform_set_position_ned_consistency_with_lla(self):
        """Test that Platform's NED position setting is consistent with LLA"""
        platform = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set position using NED
        original_north, original_east, original_down = 2000.0, -500.0, 300.0
        platform.set_position_ned(north=original_north, east=original_east, down=original_down)
        
        # Get LLA and set it back
        lat, lon, alt = platform.get_position_lla()
        platform.set_position_lla(lat=lat, lon=lon, alt=alt)
        
        # Verify NED coordinates are preserved
        final_ned = platform.get_position_ned()
        assert abs(final_ned[0] - original_north) < 1e-3
        assert abs(final_ned[1] - original_east) < 1e-3
        assert abs(final_ned[2] - original_down) < 1e-3
        
    def test_relative_positioning_same_platform(self):
        """Test relative positioning when both platforms are at same location"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point and position for both
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set same velocity and acceleration
        platform1.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        
        platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Set same orientation
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Should all be zero for identical platforms
        assert abs(relative_state['position']['x']) < 1e-3
        assert abs(relative_state['position']['y']) < 1e-3
        assert abs(relative_state['position']['z']) < 1e-3
        
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        assert abs(relative_state['acceleration']['x']) < 1e-6
        assert abs(relative_state['acceleration']['y']) < 1e-6
        assert abs(relative_state['acceleration']['z']) < 1e-6
        
    def test_relative_positioning_different_positions(self):
        """Test relative positioning with different platform positions"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Platform 1 at reference, Platform 2 offset north and east
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon + 0.001, alt=ref_alt + 10.0)
        
        # Set orientations (Platform 1 facing north)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Set zero velocities and accelerations for this test
        platform1.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Platform 2 should appear ahead (positive x) and to the right (positive y)
        assert relative_state['position']['x'] > 0  # Ahead (north)
        assert relative_state['position']['y'] > 0  # Right (east)
        assert relative_state['position']['z'] < 0  # Above (negative down)
        
    def test_relative_positioning_with_rotation(self):
        """Test relative positioning when platform 1 is rotated"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Same positions
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)  # North of platform1
        
        # Platform 1 facing east (90 degree yaw)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Set zero velocities and accelerations for this test
        platform1.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Platform 2 should now appear to the left (negative y) from platform1's perspective
        assert abs(relative_state['position']['x']) < 50  # Should be close to zero
        assert relative_state['position']['y'] < 0  # Left side
        
    def test_relative_velocity_and_acceleration(self):
        """Test relative velocity and acceleration calculations"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point and position
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set different velocities (Platform 1 moving north, Platform 2 moving east)
        platform1.set_velocity_ned(v_north=10.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=0.0, v_east=15.0, v_down=0.0)
        
        # Set different accelerations
        platform1.set_acceleration_ned(a_north=1.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=2.0, a_down=0.0)
        
        # Set same orientation (both facing north)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Relative velocity should be platform2's velocity minus platform1's velocity
        # In platform1's body frame (facing north), platform2 appears to move:
        # - backward (negative x) at 10 m/s (platform1 moving forward at 10 m/s)
        # - right (positive y) at 15 m/s (platform2 moving east at 15 m/s)
        assert abs(relative_state['velocity']['x'] + 10.0) < 1e-6  # -10 m/s
        assert abs(relative_state['velocity']['y'] - 15.0) < 1e-6  # +15 m/s
        
        # Relative acceleration should follow similar logic
        assert abs(relative_state['acceleration']['x'] + 1.0) < 1e-6  # -1 m/s²
        assert abs(relative_state['acceleration']['y'] - 2.0) < 1e-6  # +2 m/s²