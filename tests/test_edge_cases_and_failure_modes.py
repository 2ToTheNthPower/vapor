"""
Tests for edge cases and potential failure modes in the VAPOr library.

This test file focuses on scenarios that could cause the system to fail,
including numerical precision issues, coordinate system singularities,
invalid inputs, and mathematical edge cases.
"""

import pytest
import numpy as np
import math
from vapor import Platform, PositionTracker, VelocityTracker, AccelerationTracker, OrientationTracker


class TestCoordinateSystemSingularities:
    """Test coordinate system edge cases and singularities"""
    
    def test_exact_north_pole_position(self):
        """Test platform at exact North Pole where longitude is undefined"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Both platforms at North Pole with different longitudes
        platform1.set_reference_lla(lat=90.0, lon=0.0, alt=0.0)
        platform2.set_reference_lla(lat=90.0, lon=180.0, alt=0.0)  # Opposite longitude
        
        platform1.set_position_lla(lat=90.0, lon=0.0, alt=0.0)
        platform2.set_position_lla(lat=90.0, lon=90.0, alt=100.0)  # Same pole, different longitude
        
        # Set up remaining state
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Should handle pole calculation without errors
        relative_state = platform1.get_relative_state(platform2)
        
        # Results should be reasonable despite coordinate singularity
        assert isinstance(relative_state['position']['x'], float)
        assert isinstance(relative_state['position']['y'], float)
        assert isinstance(relative_state['position']['z'], float)
        assert not math.isnan(relative_state['position']['x'])
        assert not math.isnan(relative_state['position']['y'])
        assert not math.isnan(relative_state['position']['z'])
    
    def test_exact_south_pole_position(self):
        """Test platform at exact South Pole"""
        platform = Platform()
        
        platform.set_reference_lla(lat=-90.0, lon=0.0, alt=0.0)
        platform.set_position_lla(lat=-90.0, lon=0.0, alt=0.0)
        platform.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=0.0)
        platform.set_acceleration_ned(a_north=1.0, a_east=0.5, a_down=0.0)
        platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Should handle South Pole calculations
        pos_ned = platform.get_position_ned()
        vel_ned = platform.get_velocity_ned()
        accel_ned = platform.get_acceleration_ned()
        
        # All should be valid numbers, not NaN or infinite
        for coord in [pos_ned, vel_ned, accel_ned]:
            for component in coord:
                assert isinstance(component, float)
                assert not math.isnan(component)
                assert not math.isinf(component)
    
    def test_date_line_crossing_rapid_changes(self):
        """Test rapid longitude changes across international date line"""
        platform = Platform()
        
        platform.set_reference_lla(lat=0.0, lon=179.9, alt=0.0)
        
        # Series of positions crossing date line back and forth
        longitudes = [179.9, -179.9, 179.8, -179.8, 179.7, -179.7]
        
        for lon in longitudes:
            platform.set_position_lla(lat=0.0, lon=lon, alt=0.0)
            pos_ned = platform.get_position_ned()
            
            # Should handle date line crossings without discontinuities
            assert isinstance(pos_ned[0], float)
            assert isinstance(pos_ned[1], float) 
            assert isinstance(pos_ned[2], float)
            assert not math.isnan(pos_ned[0])
            assert not math.isnan(pos_ned[1])
            assert not math.isnan(pos_ned[2])
    
    def test_zero_latitude_longitude_edge_cases(self):
        """Test coordinates at exactly 0° with positive/negative distinctions"""
        platform = Platform()
        
        # Test various zero combinations
        zero_coords = [
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, 0.0),
            (0.0, -0.0, 0.0),
            (-0.0, -0.0, 0.0),
            (0.0, 0.0, -0.0),
        ]
        
        for lat, lon, alt in zero_coords:
            platform.set_reference_lla(lat=lat, lon=lon, alt=alt)
            platform.set_position_lla(lat=lat, lon=lon, alt=alt)
            
            pos_ned = platform.get_position_ned()
            
            # Should handle zero coordinates gracefully
            assert abs(pos_ned[0]) < 1e-6  # Should be essentially zero
            assert abs(pos_ned[1]) < 1e-6
            assert abs(pos_ned[2]) < 1e-6


class TestQuaternionAndRotationMatrixEdgeCases:
    """Test quaternion and rotation matrix edge cases"""
    
    def test_gimbal_lock_scenarios(self):
        """Test orientations near gimbal lock (pitch = ±90°)"""
        orientation = OrientationTracker()
        
        # Test pitch values near ±90°
        gimbal_lock_pitches = [89.9, 90.0, -89.9, -90.0, 89.99999, -89.99999]
        
        for pitch in gimbal_lock_pitches:
            # Should handle near-gimbal-lock orientations
            orientation.set_orientation_euler_ned(roll=0.0, pitch=pitch, yaw=45.0)
            
            euler_result = orientation.get_orientation_euler_ned()
            quat_result = orientation.get_orientation_quaternion_ned()
            matrix_result = orientation.get_orientation_matrix_ned()
            
            # All representations should be valid
            assert isinstance(euler_result[0], float)
            assert isinstance(euler_result[1], float)
            assert isinstance(euler_result[2], float)
            
            for component in quat_result:
                assert not math.isnan(component)
                assert not math.isinf(component)
            
            assert not np.any(np.isnan(matrix_result))
            assert not np.any(np.isinf(matrix_result))
    
    def test_quaternion_normalization_edge_cases(self):
        """Test quaternion normalization with very small magnitudes"""
        orientation = OrientationTracker()
        
        # Test very small quaternions (near-zero magnitude)
        small_quaternions = [
            (1e-10, 1e-10, 1e-10, 1e-10),
            (1e-15, 0, 0, 0),
            (0, 1e-15, 0, 0),
            (0, 0, 1e-15, 0),
            (0, 0, 0, 1e-15),
        ]
        
        for w, x, y, z in small_quaternions:
            # Should handle very small but non-zero quaternions
            orientation.set_orientation_quaternion_ned(w, x, y, z)
            
            result_quat = orientation.get_orientation_quaternion_ned()
            
            # Result should be normalized
            magnitude = math.sqrt(sum(comp**2 for comp in result_quat))
            assert abs(magnitude - 1.0) < 1e-10
    
    def test_matrix_orthogonality_boundary_cases(self):
        """Test rotation matrices at the boundary of orthogonality tolerance"""
        orientation = OrientationTracker()
        
        # Create nearly orthogonal matrices (just within tolerance)
        tolerance = 1e-6
        
        # Slightly non-orthogonal matrix (should still pass)
        matrix = np.array([
            [1.0, tolerance * 0.5, 0.0],
            [-tolerance * 0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Should accept matrix within tolerance
        orientation.set_orientation_matrix_ned(matrix)
        result_matrix = orientation.get_orientation_matrix_ned()
        
        assert np.allclose(result_matrix, matrix, atol=1e-5)
    
    def test_matrix_to_quaternion_edge_cases(self):
        """Test matrix-to-quaternion conversion at algorithmic boundaries"""
        orientation = OrientationTracker()
        
        # Test matrices that trigger different branches in matrix-to-quaternion conversion
        
        # Case 1: trace > 0 (normal case)
        matrix1 = np.eye(3)  # Identity matrix
        orientation.set_orientation_matrix_ned(matrix1)
        quat1 = orientation.get_orientation_quaternion_ned()
        assert abs(quat1[0] - 1.0) < 1e-10  # Should be identity quaternion
        
        # Case 2: matrix[0,0] is largest diagonal element (180° rotation around x-axis)
        matrix2 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        orientation.set_orientation_matrix_ned(matrix2)
        quat2 = orientation.get_orientation_quaternion_ned()
        assert not any(math.isnan(q) for q in quat2)
        
        # Case 3: matrix[1,1] is largest diagonal element (180° rotation around y-axis)
        matrix3 = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        orientation.set_orientation_matrix_ned(matrix3)
        quat3 = orientation.get_orientation_quaternion_ned()
        assert not any(math.isnan(q) for q in quat3)
        
        # Case 4: matrix[2,2] is largest diagonal element (180° rotation around z-axis)
        matrix4 = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        orientation.set_orientation_matrix_ned(matrix4)
        quat4 = orientation.get_orientation_quaternion_ned()
        assert not any(math.isnan(q) for q in quat4)
    
    def test_180_degree_rotations(self):
        """Test 180-degree rotations which can have quaternion sign ambiguity"""
        orientation = OrientationTracker()
        
        # 180-degree rotations around each axis
        rotations_180 = [
            (180.0, 0.0, 0.0),    # 180° roll
            (0.0, 180.0, 0.0),    # 180° pitch (gimbal lock)
            (0.0, 0.0, 180.0),    # 180° yaw
            (180.0, 180.0, 0.0),  # Combined rotations
            (180.0, 0.0, 180.0),
        ]
        
        for roll, pitch, yaw in rotations_180:
            orientation.set_orientation_euler_ned(roll=roll, pitch=pitch, yaw=yaw)
            
            euler_result = orientation.get_orientation_euler_ned()
            quat_result = orientation.get_orientation_quaternion_ned()
            
            # Results should be mathematically valid
            assert not any(math.isnan(angle) for angle in euler_result)
            assert not any(math.isnan(comp) for comp in quat_result)
            
            # Quaternion should be normalized
            quat_magnitude = math.sqrt(sum(comp**2 for comp in quat_result))
            assert abs(quat_magnitude - 1.0) < 1e-10


class TestInvalidInputDataHandling:
    """Test handling of invalid input data"""
    
    def test_nan_coordinate_inputs(self):
        """Test system behavior with NaN coordinate inputs
        
        NOTE: Currently the system does NOT validate for NaN inputs.
        This test documents current behavior and should be updated 
        when input validation is added.
        """
        platform = Platform()
        
        # Currently, NaN inputs are accepted (this is a gap in validation)
        platform.set_position_lla(lat=float('nan'), lon=0.0, alt=0.0)
        
        # Getting position back should preserve the NaN
        pos_lla = platform.get_position_lla()
        assert math.isnan(pos_lla[0])  # Latitude should be NaN
        
        # This demonstrates the need for input validation
        # TODO: Add input validation to reject NaN values
    
    def test_infinite_coordinate_inputs(self):
        """Test system behavior with infinite coordinate inputs
        
        NOTE: Currently the system does NOT validate for infinite inputs.
        This test documents current behavior.
        """
        platform = Platform()
        
        # Currently, infinite inputs are accepted (this is a gap in validation)
        platform.set_position_lla(lat=float('inf'), lon=0.0, alt=0.0)
        
        # Getting position back should preserve the infinity
        pos_lla = platform.get_position_lla()
        assert math.isinf(pos_lla[0])  # Latitude should be infinite
        
        # Same for velocity
        platform.set_velocity_ned(v_north=float('inf'), v_east=0.0, v_down=0.0)
        # TODO: Add input validation to reject infinite values
    
    def test_out_of_range_coordinates(self):
        """Test coordinates outside valid ranges
        
        NOTE: Currently the system does NOT validate coordinate ranges.
        This test documents current behavior.
        """
        platform = Platform()
        
        # Currently, out-of-range coordinates are accepted
        platform.set_position_lla(lat=91.0, lon=0.0, alt=0.0)  # Invalid latitude
        platform.set_position_lla(lat=-91.0, lon=0.0, alt=0.0)  # Invalid latitude
        platform.set_position_lla(lat=0.0, lon=181.0, alt=0.0)  # Invalid longitude
        platform.set_position_lla(lat=0.0, lon=-181.0, alt=0.0)  # Invalid longitude
        
        # System accepts these values (demonstrates need for validation)
        pos_lla = platform.get_position_lla()
        assert pos_lla[1] == -181.0  # Longitude preserved as-is
        
        # TODO: Add validation for coordinate ranges:
        # - Latitude: -90° to +90°
        # - Longitude: -180° to +180°
    
    def test_zero_magnitude_quaternion(self):
        """Test zero-magnitude quaternion handling"""
        orientation = OrientationTracker()
        
        # Zero quaternion should raise error
        with pytest.raises(ValueError, match="Quaternion cannot have zero magnitude"):
            orientation.set_orientation_quaternion_ned(0.0, 0.0, 0.0, 0.0)
    
    def test_invalid_rotation_matrices(self):
        """Test invalid rotation matrix inputs"""
        orientation = OrientationTracker()
        
        # Non-3x3 matrix
        with pytest.raises(ValueError, match="Rotation matrix must be 3x3"):
            orientation.set_orientation_matrix_ned(np.array([[1, 0], [0, 1]]))
        
        # Non-orthogonal matrix
        non_orthogonal = np.array([
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        with pytest.raises(ValueError, match="Matrix must be orthogonal"):
            orientation.set_orientation_matrix_ned(non_orthogonal)
        
        # Matrix with determinant ≠ 1
        reflection_matrix = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        with pytest.raises(ValueError, match="Matrix must have determinant of 1"):
            orientation.set_orientation_matrix_ned(reflection_matrix)


class TestNumericalPrecisionLimits:
    """Test numerical precision and floating-point limits"""
    
    def test_identical_position_precision(self):
        """Test platforms at identical positions (potential division by zero)"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Exactly identical positions
        identical_lat, identical_lon, identical_alt = 37.7749, -122.4194, 100.0
        
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=identical_lat, lon=identical_lon, alt=identical_alt)
            platform.set_position_lla(lat=identical_lat, lon=identical_lon, alt=identical_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Should handle identical positions without division by zero
        relative_state = platform1.get_relative_state(platform2)
        
        # All relative values should be zero or very close to zero
        assert abs(relative_state['position']['x']) < 1e-10
        assert abs(relative_state['position']['y']) < 1e-10
        assert abs(relative_state['position']['z']) < 1e-10
    
    def test_extreme_precision_differences(self):
        """Test extremely small position differences (sub-millimeter)"""
        platform1 = Platform()
        platform2 = Platform()
        
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        
        for platform in [platform1, platform2]:
            platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        # 0.1 mm difference in latitude (extremely small)
        platform2.set_position_lla(lat=ref_lat + 0.000000001, lon=ref_lon, alt=ref_alt)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle tiny differences without precision loss
        assert isinstance(relative_state['position']['x'], float)
        assert not math.isnan(relative_state['position']['x'])
        assert not math.isinf(relative_state['position']['x'])
        
        # Should detect the small difference
        assert relative_state['position']['x'] > 0  # Platform 2 is slightly north
    
    def test_extreme_distance_calculations(self):
        """Test calculations with extreme distances (interplanetary scale)"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Platform 1 on Earth
        platform1.set_reference_lla(lat=0.0, lon=0.0, alt=0.0)
        platform1.set_position_lla(lat=0.0, lon=0.0, alt=0.0)
        
        # Platform 2 at extreme distance (simulating satellite distance)
        platform2.set_reference_lla(lat=0.0, lon=0.0, alt=35786000.0)  # Geostationary orbit
        platform2.set_position_lla(lat=0.0, lon=0.0, alt=35786000.0)
        
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        relative_state = platform1.get_relative_state(platform2)
        
        # Should handle extreme distances without overflow
        assert isinstance(relative_state['position']['z'], float)
        assert not math.isnan(relative_state['position']['z'])
        assert not math.isinf(relative_state['position']['z'])
        assert relative_state['position']['z'] < -35000000  # Very far up
    
    def test_floating_point_accumulation_errors(self):
        """Test accumulated floating-point errors through multiple transformations"""
        platform = Platform()
        
        platform.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        original_lat, original_lon, original_alt = 37.7749, -122.4194, 100.0
        
        # Perform many round-trip coordinate transformations
        current_lat, current_lon, current_alt = original_lat, original_lon, original_alt
        
        for i in range(100):  # Many transformations
            platform.set_position_lla(lat=current_lat, lon=current_lon, alt=current_alt)
            
            # Get WCS coordinates
            wcs_coords = platform.get_position_wcs()
            
            # Set WCS coordinates back
            platform.set_position_wcs(x=wcs_coords[0], y=wcs_coords[1], z=wcs_coords[2])
            
            # Get LLA coordinates back
            current_lat, current_lon, current_alt = platform.get_position_lla()
        
        # After many transformations, should still be close to original
        assert abs(current_lat - original_lat) < 1e-6
        assert abs(current_lon - original_lon) < 1e-6
        assert abs(current_alt - original_alt) < 1e-3


class TestErrorPropagationAndSensitivity:
    """Test error propagation and sensitivity analysis"""
    
    def test_reference_point_sensitivity(self):
        """Test sensitivity to reference point selection"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set up platforms
        actual_lat, actual_lon, actual_alt = 37.7749, -122.4194, 100.0
        offset_lat, offset_lon, offset_alt = 37.7750, -122.4193, 105.0
        
        for platform in [platform1, platform2]:
            platform.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
            platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        platform1.set_position_lla(lat=actual_lat, lon=actual_lon, alt=actual_alt)
        platform2.set_position_lla(lat=offset_lat, lon=offset_lon, alt=offset_alt)
        
        # Test with different reference points
        ref_points = [
            (actual_lat, actual_lon, actual_alt),           # Reference at platform 1
            (offset_lat, offset_lon, offset_alt),           # Reference at platform 2  
            (37.775, -122.420, 50.0),                       # Reference between platforms
            (0.0, 0.0, 0.0),                               # Reference far away
        ]
        
        results = []
        for ref_lat, ref_lon, ref_alt in ref_points:
            platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
            
            relative_state = platform1.get_relative_state(platform2)
            results.append(relative_state['position'])
        
        # Relative positions should be similar regardless of reference point
        # (but not identical due to Earth curvature and coordinate system effects)
        for i in range(1, len(results)):
            # Should be within reasonable tolerance (Earth curvature effects)
            assert abs(results[i]['x'] - results[0]['x']) < 50  # Within 50m
            assert abs(results[i]['y'] - results[0]['y']) < 50
            assert abs(results[i]['z'] - results[0]['z']) < 10
    
    def test_coordinate_transformation_error_bounds(self):
        """Test error bounds in coordinate transformations"""
        tracker = PositionTracker()
        
        # Set reference point
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Test transformation accuracy at various distances
        test_offsets = [
            (0.001, 0.001, 10),      # ~100m
            (0.01, 0.01, 100),       # ~1km  
            (0.1, 0.1, 1000),        # ~10km
            (1.0, 1.0, 10000),       # ~100km
        ]
        
        for lat_offset, lon_offset, alt_offset in test_offsets:
            # Set position
            test_lat = 37.7749 + lat_offset
            test_lon = -122.4194 + lon_offset
            test_alt = 100.0 + alt_offset
            
            tracker.set_location_lla(lat=test_lat, lon=test_lon, alt=test_alt)
            
            # Round-trip: LLA -> WCS -> LLA
            wcs_coords = tracker.get_location_wcs()
            tracker.set_location_wcs(x=wcs_coords[0], y=wcs_coords[1], z=wcs_coords[2])
            final_lla = tracker.get_location_lla()
            
            # Error should be within reasonable bounds
            lat_error = abs(final_lla[0] - test_lat)
            lon_error = abs(final_lla[1] - test_lon)
            alt_error = abs(final_lla[2] - test_alt)
            
            assert lat_error < 1e-10, f"Latitude error {lat_error} too large for offset {lat_offset}"
            assert lon_error < 1e-10, f"Longitude error {lon_error} too large for offset {lon_offset}"
            assert alt_error < 1e-6, f"Altitude error {alt_error} too large for offset {alt_offset}"


class TestPlatformStateConsistency:
    """Test platform state consistency and validation"""
    
    def test_uninitialized_state_access(self):
        """Test accessing uninitialized platform states"""
        platform = Platform()
        
        # Accessing position before setting should raise error
        with pytest.raises(ValueError):
            platform.get_position_lla()
        
        with pytest.raises(ValueError):
            platform.get_position_wcs()
        
        with pytest.raises(ValueError):
            platform.get_position_ned()
    
    def test_missing_reference_point_errors(self):
        """Test operations requiring reference point when not set"""
        tracker = PositionTracker()
        
        # Set position but not reference
        tracker.set_location_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Getting NED without reference should raise error
        with pytest.raises(ValueError, match="Reference point not set"):
            tracker.get_location_ned()
    
    def test_partial_platform_state_relative_calculation(self):
        """Test relative state calculation with partially initialized platforms"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Only set position, not velocity/acceleration/orientation
        platform1.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        platform2.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        platform1.set_position_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        platform2.set_position_lla(lat=37.7750, lon=-122.4193, alt=105.0)
        
        # Should raise errors for missing state components
        with pytest.raises(ValueError):
            platform1.get_relative_state(platform2)


class TestTemporalConsistencyScenarios:
    """Test temporal consistency and integration scenarios"""
    
    def test_velocity_position_consistency_over_time(self):
        """Test that velocity and position changes are consistent over simulated time"""
        platform = Platform()
        
        platform.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Initial state
        initial_lat, initial_lon, initial_alt = 37.7749, -122.4194, 100.0
        platform.set_position_lla(lat=initial_lat, lon=initial_lon, alt=initial_alt)
        
        # Constant velocity north
        v_north = 10.0  # 10 m/s north
        platform.set_velocity_ned(v_north=v_north, v_east=0.0, v_down=0.0)
        platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Simulate time progression
        dt = 1.0  # 1 second time step
        total_time = 10.0  # 10 seconds
        
        for t in np.arange(dt, total_time + dt, dt):
            # Expected position based on constant velocity
            expected_distance_north = v_north * t
            
            # Manually update position (simulating integration)
            # Convert distance to approximate latitude change
            lat_change = expected_distance_north / 111000.0  # Rough m to degrees conversion
            new_lat = initial_lat + lat_change
            
            platform.set_position_lla(lat=new_lat, lon=initial_lon, alt=initial_alt)
            
            # Get NED position
            ned_pos = platform.get_position_ned()
            
            # Position should be consistent with velocity * time
            assert abs(ned_pos[0] - expected_distance_north) < 1.0  # Within 1 meter tolerance
    
    def test_acceleration_velocity_integration_consistency(self):
        """Test that acceleration and velocity changes are mathematically consistent"""
        platform = Platform()
        
        platform.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        platform.set_position_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Initial velocity
        initial_v_north = 5.0
        platform.set_velocity_ned(v_north=initial_v_north, v_east=0.0, v_down=0.0)
        
        # Constant acceleration
        a_north = 2.0  # 2 m/s² north
        platform.set_acceleration_ned(a_north=a_north, a_east=0.0, a_down=0.0)
        
        # Test velocity integration over time
        dt = 0.1  # 0.1 second time step
        total_time = 5.0  # 5 seconds
        
        for t in np.arange(dt, total_time + dt, dt):
            # Expected velocity based on constant acceleration
            expected_v_north = initial_v_north + a_north * t
            
            # Update velocity (simulating integration)
            platform.set_velocity_ned(v_north=expected_v_north, v_east=0.0, v_down=0.0)
            
            # Get velocity and acceleration
            vel_ned = platform.get_velocity_ned()
            accel_ned = platform.get_acceleration_ned()
            
            # Velocity should match expected value
            assert abs(vel_ned[0] - expected_v_north) < 0.01  # Within 1 cm/s
            
            # Acceleration should remain constant
            assert abs(accel_ned[0] - a_north) < 1e-10