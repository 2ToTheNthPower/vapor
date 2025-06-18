import pytest
import numpy as np
import math
from vapor import PositionTracker, VelocityTracker, AccelerationTracker, OrientationTracker


class TestPositionTransformationReversibility:
    def test_wcs_lla_wcs_round_trip(self):
        """Test WCS -> LLA -> WCS round trip transformation"""
        tracker = PositionTracker()
        
        # Original WCS position (San Francisco area in ECEF)
        original_x, original_y, original_z = -2707942.0, -4282625.0, 3857502.0
        
        # WCS -> LLA -> WCS
        tracker.set_location_wcs(original_x, original_y, original_z)
        lat, lon, alt = tracker.get_location_lla()
        tracker.set_location_lla(lat, lon, alt)
        final_x, final_y, final_z = tracker.get_location_wcs()
        
        # Verify round trip accuracy (within 1mm)
        assert abs(final_x - original_x) < 1e-3
        assert abs(final_y - original_y) < 1e-3
        assert abs(final_z - original_z) < 1e-3
    
    def test_lla_wcs_lla_round_trip(self):
        """Test LLA -> WCS -> LLA round trip transformation"""
        tracker = PositionTracker()
        
        # Original LLA position (San Francisco)
        original_lat, original_lon, original_alt = 37.7749, -122.4194, 100.0
        
        # LLA -> WCS -> LLA
        tracker.set_location_lla(original_lat, original_lon, original_alt)
        x, y, z = tracker.get_location_wcs()
        tracker.set_location_wcs(x, y, z)
        final_lat, final_lon, final_alt = tracker.get_location_lla()
        
        # Verify round trip accuracy
        assert abs(final_lat - original_lat) < 1e-9
        assert abs(final_lon - original_lon) < 1e-9
        assert abs(final_alt - original_alt) < 1e-3
    
    def test_wcs_shift_reversibility(self):
        """Test WCS coordinate shifts are reversible"""
        tracker = PositionTracker()
        
        # Start with a position in WCS
        start_x, start_y, start_z = -2707942.0, -4282625.0, 3857502.0
        tracker.set_location_wcs(start_x, start_y, start_z)
        
        # Get LLA representation
        original_lat, original_lon, original_alt = tracker.get_location_lla()
        
        # Shift in WCS space (100m in each direction)
        shift_x, shift_y, shift_z = 100.0, 50.0, -25.0
        shifted_x = start_x + shift_x
        shifted_y = start_y + shift_y
        shifted_z = start_z + shift_z
        
        # Set shifted WCS position, convert to LLA, then back to WCS
        tracker.set_location_wcs(shifted_x, shifted_y, shifted_z)
        shifted_lat, shifted_lon, shifted_alt = tracker.get_location_lla()
        tracker.set_location_lla(shifted_lat, shifted_lon, shifted_alt)
        final_x, final_y, final_z = tracker.get_location_wcs()
        
        # Shift back the same amount
        unshifted_x = final_x - shift_x
        unshifted_y = final_y - shift_y
        unshifted_z = final_z - shift_z
        
        # Convert back to LLA and verify we get the original position
        tracker.set_location_wcs(unshifted_x, unshifted_y, unshifted_z)
        final_lat, final_lon, final_alt = tracker.get_location_lla()
        
        # Verify we're back to original position (within precision limits)
        assert abs(final_lat - original_lat) < 1e-9
        assert abs(final_lon - original_lon) < 1e-9
        assert abs(final_alt - original_alt) < 1e-3
    
    def test_ned_transformations_reversibility(self):
        """Test NED transformations are reversible"""
        tracker = PositionTracker()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        tracker.set_reference_lla(ref_lat, ref_lon, ref_alt)
        
        # Original position in LLA
        original_lat, original_lon, original_alt = 37.7750, -122.4193, 105.0
        tracker.set_location_lla(original_lat, original_lon, original_alt)
        
        # LLA -> NED -> (via WCS) -> LLA
        original_north, original_east, original_down = tracker.get_location_ned()
        
        # Convert to WCS and back
        x, y, z = tracker.get_location_wcs()
        tracker.set_location_wcs(x, y, z)
        
        # Get NED again and verify consistency
        final_north, final_east, final_down = tracker.get_location_ned()
        
        assert abs(final_north - original_north) < 1e-3
        assert abs(final_east - original_east) < 1e-3
        assert abs(final_down - original_down) < 1e-3
        
        # Verify LLA is also preserved
        final_lat, final_lon, final_alt = tracker.get_location_lla()
        assert abs(final_lat - original_lat) < 1e-9
        assert abs(final_lon - original_lon) < 1e-9
        assert abs(final_alt - original_alt) < 1e-3
    
    def test_multiple_coordinate_system_chain(self):
        """Test chain of transformations: WCS -> LLA -> WCS -> LLA -> WCS"""
        tracker = PositionTracker()
        
        # Original WCS position
        original_x, original_y, original_z = -2707942.0, -4282625.0, 3857502.0
        
        # Chain: WCS -> LLA -> WCS -> LLA -> WCS
        tracker.set_location_wcs(original_x, original_y, original_z)
        lat1, lon1, alt1 = tracker.get_location_lla()
        
        tracker.set_location_lla(lat1, lon1, alt1)
        x2, y2, z2 = tracker.get_location_wcs()
        
        tracker.set_location_wcs(x2, y2, z2)
        lat3, lon3, alt3 = tracker.get_location_lla()
        
        tracker.set_location_lla(lat3, lon3, alt3)
        final_x, final_y, final_z = tracker.get_location_wcs()
        
        # Verify final position matches original
        assert abs(final_x - original_x) < 1e-3
        assert abs(final_y - original_y) < 1e-3
        assert abs(final_z - original_z) < 1e-3


class TestVelocityTransformationReversibility:
    def test_wcs_ned_wcs_velocity_round_trip(self):
        """Test WCS -> NED -> WCS velocity round trip"""
        tracker = VelocityTracker()
        
        # Set reference point for NED
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original WCS velocity
        original_vx, original_vy, original_vz = 10.0, 5.0, -2.0
        
        # WCS -> NED -> WCS
        tracker.set_velocity_wcs(original_vx, original_vy, original_vz)
        v_north, v_east, v_down = tracker.get_velocity_ned()
        
        tracker.set_velocity_ned(v_north, v_east, v_down)
        final_vx, final_vy, final_vz = tracker.get_velocity_wcs()
        
        # Verify round trip accuracy
        assert abs(final_vx - original_vx) < 1e-6
        assert abs(final_vy - original_vy) < 1e-6
        assert abs(final_vz - original_vz) < 1e-6
    
    def test_ned_wcs_ned_velocity_round_trip(self):
        """Test NED -> WCS -> NED velocity round trip"""
        tracker = VelocityTracker()
        
        # Set reference point for NED
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original NED velocity
        original_vnorth, original_veast, original_vdown = 15.0, -8.0, 3.0
        
        # NED -> WCS -> NED
        tracker.set_velocity_ned(original_vnorth, original_veast, original_vdown)
        vx, vy, vz = tracker.get_velocity_wcs()
        
        tracker.set_velocity_wcs(vx, vy, vz)
        final_vnorth, final_veast, final_vdown = tracker.get_velocity_ned()
        
        # Verify round trip accuracy
        assert abs(final_vnorth - original_vnorth) < 1e-6
        assert abs(final_veast - original_veast) < 1e-6
        assert abs(final_vdown - original_vdown) < 1e-6
    
    def test_lla_rates_ned_reversibility(self):
        """Test LLA rates -> NED -> NED consistency"""
        tracker = VelocityTracker()
        
        # Set reference point
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original LLA rates
        original_lat_rate, original_lon_rate, original_alt_rate = 0.001, 0.0005, 2.0
        
        # LLA rates -> NED
        tracker.set_velocity_lla_rates(original_lat_rate, original_lon_rate, original_alt_rate)
        v_north1, v_east1, v_down1 = tracker.get_velocity_ned()
        
        # Set the same NED values directly and verify consistency
        tracker.set_velocity_ned(v_north1, v_east1, v_down1)
        v_north2, v_east2, v_down2 = tracker.get_velocity_ned()
        
        # Verify NED velocities are consistent
        assert abs(v_north2 - v_north1) < 1e-6
        assert abs(v_east2 - v_east1) < 1e-6
        assert abs(v_down2 - v_down1) < 1e-6


class TestAccelerationTransformationReversibility:
    def test_wcs_ned_wcs_acceleration_round_trip(self):
        """Test WCS -> NED -> WCS acceleration round trip"""
        tracker = AccelerationTracker()
        
        # Set reference point for NED
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original WCS acceleration
        original_ax, original_ay, original_az = 2.0, 1.5, -9.81
        
        # WCS -> NED -> WCS
        tracker.set_acceleration_wcs(original_ax, original_ay, original_az)
        a_north, a_east, a_down = tracker.get_acceleration_ned()
        
        tracker.set_acceleration_ned(a_north, a_east, a_down)
        final_ax, final_ay, final_az = tracker.get_acceleration_wcs()
        
        # Verify round trip accuracy
        assert abs(final_ax - original_ax) < 1e-6
        assert abs(final_ay - original_ay) < 1e-6
        assert abs(final_az - original_az) < 1e-6
    
    def test_ned_wcs_ned_acceleration_round_trip(self):
        """Test NED -> WCS -> NED acceleration round trip"""
        tracker = AccelerationTracker()
        
        # Set reference point for NED
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original NED acceleration
        original_anorth, original_aeast, original_adown = 1.0, -0.5, 9.81
        
        # NED -> WCS -> NED
        tracker.set_acceleration_ned(original_anorth, original_aeast, original_adown)
        ax, ay, az = tracker.get_acceleration_wcs()
        
        tracker.set_acceleration_wcs(ax, ay, az)
        final_anorth, final_aeast, final_adown = tracker.get_acceleration_ned()
        
        # Verify round trip accuracy
        assert abs(final_anorth - original_anorth) < 1e-6
        assert abs(final_aeast - original_aeast) < 1e-6
        assert abs(final_adown - original_adown) < 1e-6
    
    def test_lla_rates_ned_acceleration_reversibility(self):
        """Test LLA acceleration rates -> NED -> NED consistency"""
        tracker = AccelerationTracker()
        
        # Set reference point
        tracker.set_reference_lla(37.7749, -122.4194, 100.0)
        
        # Original LLA acceleration rates
        original_lat_accel, original_lon_accel, original_alt_accel = 0.0001, -0.0002, -9.81
        
        # LLA rates -> NED
        tracker.set_acceleration_lla_rates(original_lat_accel, original_lon_accel, original_alt_accel)
        a_north1, a_east1, a_down1 = tracker.get_acceleration_ned()
        
        # Set the same NED values directly and verify consistency
        tracker.set_acceleration_ned(a_north1, a_east1, a_down1)
        a_north2, a_east2, a_down2 = tracker.get_acceleration_ned()
        
        # Verify NED accelerations are consistent
        assert abs(a_north2 - a_north1) < 1e-6
        assert abs(a_east2 - a_east1) < 1e-6
        assert abs(a_down2 - a_down1) < 1e-6


class TestOrientationTransformationReversibility:
    def test_euler_quaternion_euler_round_trip(self):
        """Test Euler -> Quaternion -> Euler round trip"""
        tracker = OrientationTracker()
        
        # Original Euler angles
        original_roll, original_pitch, original_yaw = 15.0, -10.0, 45.0
        
        # Euler -> Quaternion -> Euler
        tracker.set_orientation_euler_ned(original_roll, original_pitch, original_yaw)
        w, x, y, z = tracker.get_orientation_quaternion_ned()
        
        tracker.set_orientation_quaternion_ned(w, x, y, z)
        final_roll, final_pitch, final_yaw = tracker.get_orientation_euler_ned()
        
        # Verify round trip accuracy
        assert abs(final_roll - original_roll) < 1e-6
        assert abs(final_pitch - original_pitch) < 1e-6
        assert abs(final_yaw - original_yaw) < 1e-6
    
    def test_quaternion_matrix_quaternion_round_trip(self):
        """Test Quaternion -> Matrix -> Quaternion round trip"""
        tracker = OrientationTracker()
        
        # Original quaternion (normalized)
        original_w, original_x, original_y, original_z = 0.7071, 0.0, 0.0, 0.7071
        
        # Quaternion -> Matrix -> Quaternion
        tracker.set_orientation_quaternion_ned(original_w, original_x, original_y, original_z)
        matrix = tracker.get_orientation_matrix_ned()
        
        tracker.set_orientation_matrix_ned(matrix)
        final_w, final_x, final_y, final_z = tracker.get_orientation_quaternion_ned()
        
        # Verify round trip accuracy (account for quaternion sign ambiguity)
        # Quaternions q and -q represent the same rotation
        dot_product = (final_w * original_w + final_x * original_x + 
                      final_y * original_y + final_z * original_z)
        
        if dot_product < 0:
            # Flip sign if needed
            final_w, final_x, final_y, final_z = -final_w, -final_x, -final_y, -final_z
        
        assert abs(final_w - original_w) < 1e-5
        assert abs(final_x - original_x) < 1e-5
        assert abs(final_y - original_y) < 1e-5
        assert abs(final_z - original_z) < 1e-5
    
    def test_euler_matrix_euler_round_trip(self):
        """Test Euler -> Matrix -> Euler round trip"""
        tracker = OrientationTracker()
        
        # Original Euler angles
        original_roll, original_pitch, original_yaw = 30.0, 20.0, -60.0
        
        # Euler -> Matrix -> Euler
        tracker.set_orientation_euler_ned(original_roll, original_pitch, original_yaw)
        matrix = tracker.get_orientation_matrix_ned()
        
        tracker.set_orientation_matrix_ned(matrix)
        final_roll, final_pitch, final_yaw = tracker.get_orientation_euler_ned()
        
        # Verify round trip accuracy
        assert abs(final_roll - original_roll) < 1e-6
        assert abs(final_pitch - original_pitch) < 1e-6
        assert abs(final_yaw - original_yaw) < 1e-6
    
    def test_orientation_chain_transformations(self):
        """Test chain: Euler -> Quaternion -> Matrix -> Quaternion -> Euler"""
        tracker = OrientationTracker()
        
        # Original Euler angles
        original_roll, original_pitch, original_yaw = 25.0, -15.0, 90.0
        
        # Chain: Euler -> Quaternion -> Matrix -> Quaternion -> Euler
        tracker.set_orientation_euler_ned(original_roll, original_pitch, original_yaw)
        w1, x1, y1, z1 = tracker.get_orientation_quaternion_ned()
        
        tracker.set_orientation_quaternion_ned(w1, x1, y1, z1)
        matrix = tracker.get_orientation_matrix_ned()
        
        tracker.set_orientation_matrix_ned(matrix)
        w2, x2, y2, z2 = tracker.get_orientation_quaternion_ned()
        
        tracker.set_orientation_quaternion_ned(w2, x2, y2, z2)
        final_roll, final_pitch, final_yaw = tracker.get_orientation_euler_ned()
        
        # Verify final result matches original
        assert abs(final_roll - original_roll) < 1e-6
        assert abs(final_pitch - original_pitch) < 1e-6
        assert abs(final_yaw - original_yaw) < 1e-6
    
    def test_gimbal_lock_cases(self):
        """Test transformations near gimbal lock conditions"""
        tracker = OrientationTracker()
        
        # Test near +90 degree pitch
        test_cases = [
            (0.0, 89.9, 45.0),
            (0.0, -89.9, 45.0),
            (10.0, 89.9, -30.0),
            (10.0, -89.9, -30.0)
        ]
        
        for original_roll, original_pitch, original_yaw in test_cases:
            # Euler -> Quaternion -> Euler
            tracker.set_orientation_euler_ned(original_roll, original_pitch, original_yaw)
            w, x, y, z = tracker.get_orientation_quaternion_ned()
            
            tracker.set_orientation_quaternion_ned(w, x, y, z)
            final_roll, final_pitch, final_yaw = tracker.get_orientation_euler_ned()
            
            # For near-gimbal lock cases, the total rotation should be preserved
            # even if individual angles might be represented differently
            # Test by converting both to rotation matrices and comparing
            tracker.set_orientation_euler_ned(original_roll, original_pitch, original_yaw)
            original_matrix = tracker.get_orientation_matrix_ned()
            
            tracker.set_orientation_euler_ned(final_roll, final_pitch, final_yaw)
            final_matrix = tracker.get_orientation_matrix_ned()
            
            # Matrices should be nearly identical
            assert np.allclose(original_matrix, final_matrix, atol=1e-6)