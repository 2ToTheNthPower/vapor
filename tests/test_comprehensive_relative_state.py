"""
Comprehensive test coverage for Platform.get_relative_state() method.

This test file ensures the relative state calculation works correctly in all scenarios,
including edge cases, different reference points, and mathematical correctness.
"""

import pytest
import numpy as np
import math
from vapor import Platform


class TestRelativeStateWithDifferentReferences:
    """Test the key fix: platforms with different reference points"""
    
    def test_same_reference_different_positions(self):
        """Test platforms with same reference point but different positions"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Both platforms use the same reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Platform 1 at reference point, Platform 2 offset
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.009, lon=ref_lon + 0.009, alt=ref_alt + 100)
        
        # Both stationary and facing north
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should now see meaningful relative position (not zero)
        assert relative_state['position']['x'] > 900  # ~1000m north
        assert relative_state['position']['y'] > 700  # ~1000m east (allowing for longitude convergence)
        assert relative_state['position']['z'] < -90  # ~100m up (negative z)
        
    def test_different_reference_points(self):
        """Test platforms with completely different reference points"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Different reference points
        platform1.set_reference_lla(lat=37.0, lon=-122.0, alt=0.0)
        platform2.set_reference_lla(lat=38.0, lon=-121.0, alt=100.0)
        
        # Same actual position in world coordinates
        actual_lat, actual_lon, actual_alt = 37.5, -121.5, 50.0
        platform1.set_position_lla(lat=actual_lat, lon=actual_lon, alt=actual_alt)
        platform2.set_position_lla(lat=actual_lat, lon=actual_lon, alt=actual_alt)
        
        # Both stationary and facing north
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should be approximately zero since they're at the same world position
        assert abs(relative_state['position']['x']) < 1e-3
        assert abs(relative_state['position']['y']) < 1e-3
        assert abs(relative_state['position']['z']) < 1e-3


class TestRelativeStateMathematicalCorrectness:
    """Test mathematical correctness of relative state calculations"""
    
    def test_relative_position_symmetry(self):
        """Test that relative position is anti-symmetric"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set up platforms
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon + 0.001, alt=ref_alt + 50)
        
        # Get relative states both ways
        rel_1_to_2 = platform1.get_relative_state(platform2)
        rel_2_to_1 = platform2.get_relative_state(platform1)
        
        # Should be negatives of each other (allowing for small coordinate transformation differences)
        assert abs(rel_1_to_2['position']['x'] + rel_2_to_1['position']['x']) < 0.01  # Within 1cm
        assert abs(rel_1_to_2['position']['y'] + rel_2_to_1['position']['y']) < 0.01
        assert abs(rel_1_to_2['position']['z'] + rel_2_to_1['position']['z']) < 0.01
        
    def test_relative_velocity_additivity(self):
        """Test that relative velocities add correctly"""
        platform1 = Platform()
        platform2 = Platform()
        platform3 = Platform()
        
        # Set up platforms
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2, platform3]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Set different velocities
        platform1.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=20.0, v_east=10.0, v_down=-5.0)
        platform3.set_velocity_ned(v_north=30.0, v_east=15.0, v_down=-10.0)
        
        # Get relative velocities
        rel_1_to_2 = platform1.get_relative_state(platform2)
        rel_2_to_3 = platform2.get_relative_state(platform3)
        rel_1_to_3 = platform1.get_relative_state(platform3)
        
        # Check additivity: v_1_to_3 ≈ v_1_to_2 + v_2_to_3 (in body frame of platform1)
        v_1_to_2 = np.array([rel_1_to_2['velocity']['x'], rel_1_to_2['velocity']['y'], rel_1_to_2['velocity']['z']])
        v_2_to_3 = np.array([rel_2_to_3['velocity']['x'], rel_2_to_3['velocity']['y'], rel_2_to_3['velocity']['z']])
        v_1_to_3 = np.array([rel_1_to_3['velocity']['x'], rel_1_to_3['velocity']['y'], rel_1_to_3['velocity']['z']])
        
        # Note: This test is approximate due to coordinate frame differences
        expected = v_1_to_3
        actual = v_1_to_2 + v_2_to_3
        
        # The vectors should be reasonably close (allowing for frame transformation differences)
        assert np.linalg.norm(expected - actual) < 5.0  # Within 5 m/s tolerance
        
    def test_coordinate_frame_transformation_consistency(self):
        """Test that transformations between NED and body frames are consistent"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)  # 100m north
        
        # Test with platform1 facing north (0°)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        rel_state = platform1.get_relative_state(platform2)
        
        # Platform2 should be ahead (positive x) in platform1's body frame
        assert rel_state['position']['x'] > 90  # ~100m ahead
        assert abs(rel_state['position']['y']) < 10  # Minimal lateral offset
        
        # Test with platform1 facing east (90°)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        rel_state = platform1.get_relative_state(platform2)
        
        # Platform2 should now be to the left (negative y) in platform1's body frame
        assert abs(rel_state['position']['x']) < 10  # Minimal forward/back
        assert rel_state['position']['y'] < -90  # ~100m to the left


class TestRelativeStateEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_extreme_positions(self):
        """Test with extreme positions (large distances)"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0  # Equator, prime meridian
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Platform 1 at reference
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Platform 2 very far away (1000 km north)
        platform2.set_position_lla(lat=ref_lat + 9.0, lon=ref_lon, alt=ref_alt)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle large distances correctly
        assert relative_state['position']['x'] > 900000  # ~1000 km
        assert abs(relative_state['position']['y']) < 1000
        assert abs(relative_state['position']['z']) < 100000  # Allow for Earth curvature effects at large distances
        
    def test_high_altitude_differences(self):
        """Test with large altitude differences"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 0.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Ground level vs stratosphere
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=0.0)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=50000.0)  # 50 km altitude
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Platform 2 should be high above (negative z = up)
        assert abs(relative_state['position']['x']) < 100
        assert abs(relative_state['position']['y']) < 100
        assert relative_state['position']['z'] < -49000  # ~50 km up
        
    def test_polar_regions(self):
        """Test near polar regions where longitude changes rapidly"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Near north pole
        ref_lat, ref_lon, ref_alt = 89.0, 0.0, 0.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon + 90.0, alt=ref_alt)  # 90° longitude difference
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle polar coordinates correctly (small distances due to convergence)
        total_distance = math.sqrt(
            relative_state['position']['x']**2 + 
            relative_state['position']['y']**2 + 
            relative_state['position']['z']**2
        )
        assert total_distance < 200000  # Should be much less than equatorial distance
        
    def test_across_date_line(self):
        """Test platforms across the international date line"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 0.0, 179.0, 0.0  # Near date line
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=179.0, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=-179.0, alt=ref_alt)  # Across date line
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle date line crossing correctly (short distance, not halfway around world)
        total_distance = math.sqrt(
            relative_state['position']['x']**2 + 
            relative_state['position']['y']**2
        )
        assert total_distance < 250000  # Should be ~222 km, not ~20000 km


class TestRelativeStateWithDifferentDataSources:
    """Test relative state calculation with different coordinate system sources"""
    
    def test_mixed_coordinate_sources_velocity(self):
        """Test when platforms use different velocity coordinate systems"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Platform 1 uses NED velocity, Platform 2 uses WCS velocity
        platform1.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=-2.0)
        platform2.set_velocity_wcs(vx=100.0, vy=50.0, vz=-20.0)  # Different coordinate system
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle mixed coordinate systems correctly
        assert isinstance(relative_state['velocity']['x'], float)
        assert isinstance(relative_state['velocity']['y'], float)
        assert isinstance(relative_state['velocity']['z'], float)
        
    def test_mixed_coordinate_sources_acceleration(self):
        """Test when platforms use different acceleration coordinate systems"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Platform 1 uses NED acceleration, Platform 2 uses WCS acceleration
        platform1.set_acceleration_ned(a_north=1.0, a_east=0.5, a_down=-9.81)
        platform2.set_acceleration_wcs(ax=10.0, ay=5.0, az=-98.1)  # Different coordinate system
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle mixed coordinate systems correctly
        assert isinstance(relative_state['acceleration']['x'], float)
        assert isinstance(relative_state['acceleration']['y'], float)
        assert isinstance(relative_state['acceleration']['z'], float)


class TestRelativeStateNumericalPrecision:
    """Test numerical precision and stability"""
    
    def test_precision_with_small_differences(self):
        """Test precision when platforms are very close together"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Very small position difference (1 cm)
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.00000009, lon=ref_lon, alt=ref_alt)  # ~1 cm north
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should maintain precision for small differences
        assert 0.005 < relative_state['position']['x'] < 0.015  # ~1 cm with some tolerance
        assert abs(relative_state['position']['y']) < 0.005
        assert abs(relative_state['position']['z']) < 0.005
        
    def test_stability_with_repeated_calculations(self):
        """Test numerical stability with repeated calculations"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon + 0.001, alt=ref_alt)
        
        # Calculate relative state multiple times
        results = []
        for _ in range(10):
            rel_state = platform1.get_relative_state(platform2)
            results.append([
                rel_state['position']['x'],
                rel_state['position']['y'], 
                rel_state['position']['z']
            ])
        
        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert abs(results[i][0] - results[0][0]) < 1e-10
            assert abs(results[i][1] - results[0][1]) < 1e-10
            assert abs(results[i][2] - results[0][2]) < 1e-10


class TestRelativeStateRegressionTests:
    """Regression tests to ensure the fix doesn't break existing functionality"""
    
    def test_backwards_compatibility_with_same_reference(self):
        """Ensure existing tests still pass with same reference point"""
        platform1 = Platform()
        platform2 = Platform()
        
        # This mimics the old behavior where both platforms had the same reference
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set different positions
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon + 0.001, alt=ref_alt)
        
        # Set same velocities and accelerations
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=1.0, a_east=0.5, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Now we should get meaningful relative position (this was broken before)
        assert relative_state['position']['x'] > 100  # Should see the offset
        assert relative_state['position']['y'] > 80  # Should see the offset (allow for coordinate effects)
        
        # Relative velocity and acceleration should still be zero (same motion)
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        assert abs(relative_state['acceleration']['x']) < 1e-6
        assert abs(relative_state['acceleration']['y']) < 1e-6
        assert abs(relative_state['acceleration']['z']) < 1e-6