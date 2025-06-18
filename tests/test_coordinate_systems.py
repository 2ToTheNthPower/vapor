import pytest
import numpy as np
from vapor import PositionTracker, VelocityTracker, AccelerationTracker


class TestPositionTracker:
    def test_wcs_to_lla_conversion(self):
        """Test World Coordinate System to Latitude/Longitude/Altitude conversion"""
        tracker = PositionTracker()
        
        # Set position in WCS (example: ECEF coordinates in meters)
        tracker.set_location_wcs(x=4194304.0, y=171885.0, z=4780000.0)
        
        # Get LLA coordinates
        lat, lon, alt = tracker.get_location_lla()
        
        # Verify reasonable values for a position on Earth
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert alt > -11000  # Below Dead Sea level
        
    def test_lla_to_wcs_conversion(self):
        """Test Latitude/Longitude/Altitude to World Coordinate System conversion"""
        tracker = PositionTracker()
        
        # Set position in LLA (San Francisco coordinates)
        tracker.set_location_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Get WCS coordinates
        x, y, z = tracker.get_location_wcs()
        
        # Verify coordinates are reasonable ECEF values
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert abs(x) < 7000000  # Earth radius bounds
        assert abs(y) < 7000000
        assert abs(z) < 7000000
        
    def test_lla_to_ned_conversion(self):
        """Test Latitude/Longitude/Altitude to North/East/Down conversion"""
        tracker = PositionTracker()
        
        # Set reference point (origin for NED)
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set current position slightly offset
        tracker.set_location_lla(lat=37.7750, lon=-122.4193, alt=105.0)
        
        # Get NED coordinates
        north, east, down = tracker.get_location_ned()
        
        # Verify NED values are reasonable
        assert isinstance(north, float)
        assert isinstance(east, float)
        assert isinstance(down, float)
        assert abs(north) < 1000  # Small offset should be within km
        assert abs(east) < 1000
        assert abs(down) < 1000
        
    def test_wcs_to_ned_conversion(self):
        """Test World Coordinate System to North/East/Down conversion"""
        tracker = PositionTracker()
        
        # Set reference point in LLA first
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set position in WCS
        tracker.set_location_wcs(x=4194304.0, y=171885.0, z=4780000.0)
        
        # Get NED coordinates
        north, east, down = tracker.get_location_ned()
        
        # Verify NED values are reasonable
        assert isinstance(north, float)
        assert isinstance(east, float)
        assert isinstance(down, float)
        
    def test_coordinate_system_consistency(self):
        """Test that conversions are consistent across coordinate systems"""
        tracker = PositionTracker()
        
        # Set reference point
        tracker.set_reference_lla(lat=0.0, lon=0.0, alt=0.0)
        
        # Set initial position in LLA
        original_lat, original_lon, original_alt = 37.7749, -122.4194, 100.0
        tracker.set_location_lla(lat=original_lat, lon=original_lon, alt=original_alt)
        
        # Convert to WCS and back
        x, y, z = tracker.get_location_wcs()
        tracker.set_location_wcs(x=x, y=y, z=z)
        converted_lat, converted_lon, converted_alt = tracker.get_location_lla()
        
        # Verify consistency (within reasonable precision)
        assert abs(converted_lat - original_lat) < 1e-6
        assert abs(converted_lon - original_lon) < 1e-6
        assert abs(converted_alt - original_alt) < 1e-3

    def test_set_location_ned_basic(self):
        """Test setting position using NED coordinates"""
        tracker = PositionTracker()
        
        # Set reference point (San Francisco)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set position using NED coordinates
        north, east, down = 1000.0, 500.0, -200.0  # 1km north, 500m east, 200m up
        tracker.set_location_ned(north=north, east=east, down=down)
        
        # Verify the position can be retrieved in NED
        retrieved_ned = tracker.get_location_ned()
        assert abs(retrieved_ned[0] - north) < 1e-3
        assert abs(retrieved_ned[1] - east) < 1e-3
        assert abs(retrieved_ned[2] - down) < 1e-3
        
        # Verify it can also be retrieved in other coordinate systems
        lla_pos = tracker.get_location_lla()
        wcs_pos = tracker.get_location_wcs()
        
        assert isinstance(lla_pos[0], float)
        assert isinstance(lla_pos[1], float)
        assert isinstance(lla_pos[2], float)
        assert isinstance(wcs_pos[0], float)
        assert isinstance(wcs_pos[1], float)
        assert isinstance(wcs_pos[2], float)

    def test_set_location_ned_reversibility(self):
        """Test that NED position setting is reversible with other coordinate systems"""
        tracker = PositionTracker()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Test 1: Set NED, convert to LLA, convert back to NED
        original_north, original_east, original_down = 2000.0, -500.0, 300.0
        tracker.set_location_ned(north=original_north, east=original_east, down=original_down)
        
        # Get LLA and set it back
        lat, lon, alt = tracker.get_location_lla()
        tracker.set_location_lla(lat=lat, lon=lon, alt=alt)
        
        # Verify NED coordinates are preserved
        final_ned = tracker.get_location_ned()
        assert abs(final_ned[0] - original_north) < 1e-3
        assert abs(final_ned[1] - original_east) < 1e-3
        assert abs(final_ned[2] - original_down) < 1e-3
        
        # Test 2: Set NED, convert to WCS, convert back to NED
        tracker.set_location_ned(north=original_north, east=original_east, down=original_down)
        
        # Get WCS and set it back
        x, y, z = tracker.get_location_wcs()
        tracker.set_location_wcs(x=x, y=y, z=z)
        
        # Verify NED coordinates are preserved
        final_ned = tracker.get_location_ned()
        assert abs(final_ned[0] - original_north) < 1e-3
        assert abs(final_ned[1] - original_east) < 1e-3
        assert abs(final_ned[2] - original_down) < 1e-3

    def test_set_location_ned_multiple_positions(self):
        """Test setting multiple different NED positions"""
        tracker = PositionTracker()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Test various NED positions
        test_positions = [
            (0.0, 0.0, 0.0),        # At reference point
            (1000.0, 0.0, 0.0),     # 1km north
            (0.0, 1000.0, 0.0),     # 1km east
            (0.0, 0.0, -1000.0),    # 1km up
            (-500.0, -300.0, 200.0), # Southwest and down
            (5000.0, 2000.0, -914.0) # Like the aircraft example
        ]
        
        for north, east, down in test_positions:
            tracker.set_location_ned(north=north, east=east, down=down)
            
            # Verify the position is set correctly
            retrieved_ned = tracker.get_location_ned()
            assert abs(retrieved_ned[0] - north) < 1e-3, f"North mismatch: {retrieved_ned[0]} != {north}"
            assert abs(retrieved_ned[1] - east) < 1e-3, f"East mismatch: {retrieved_ned[1]} != {east}"
            assert abs(retrieved_ned[2] - down) < 1e-3, f"Down mismatch: {retrieved_ned[2]} != {down}"
            
            # Verify other coordinate systems are reasonable
            lat, lon, alt = tracker.get_location_lla()
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
            assert alt > -12000  # Reasonable altitude bounds

    def test_set_location_ned_no_reference_error(self):
        """Test that setting NED position without reference point raises error"""
        tracker = PositionTracker()
        
        # Try to set NED position without setting reference point first
        with pytest.raises(ValueError, match="Reference point not set"):
            tracker.set_location_ned(north=100.0, east=200.0, down=-50.0)


class TestVelocityTracker:
    def test_velocity_wcs_to_ned(self):
        """Test velocity conversion from WCS to NED"""
        tracker = VelocityTracker()
        
        # Set reference point for NED frame
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set velocity in WCS (m/s)
        tracker.set_velocity_wcs(vx=10.0, vy=5.0, vz=-2.0)
        
        # Get velocity in NED
        v_north, v_east, v_down = tracker.get_velocity_ned()
        
        # Verify velocity values are reasonable
        assert isinstance(v_north, float)
        assert isinstance(v_east, float)
        assert isinstance(v_down, float)
        
    def test_velocity_lla_rate_to_ned(self):
        """Test velocity conversion from LLA rates to NED"""
        tracker = VelocityTracker()
        
        # Set reference point
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set velocity as LLA rates (degrees/s, degrees/s, m/s)
        tracker.set_velocity_lla_rates(lat_rate=0.0001, lon_rate=0.0001, alt_rate=1.0)
        
        # Get velocity in NED
        v_north, v_east, v_down = tracker.get_velocity_ned()
        
        # Verify velocity values are reasonable
        assert isinstance(v_north, float)
        assert isinstance(v_east, float)
        assert isinstance(v_down, float)
        assert v_north > 0  # Positive lat rate should give positive north velocity
        assert v_east > 0   # Positive lon rate should give positive east velocity
        assert v_down < 0   # Positive alt rate should give negative down velocity


class TestAccelerationTracker:
    def test_acceleration_wcs_to_ned(self):
        """Test acceleration conversion from WCS to NED"""
        tracker = AccelerationTracker()
        
        # Set reference point for NED frame
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set acceleration in WCS (m/s²)
        tracker.set_acceleration_wcs(ax=1.0, ay=0.5, az=-9.81)
        
        # Get acceleration in NED
        a_north, a_east, a_down = tracker.get_acceleration_ned()
        
        # Verify acceleration values are reasonable
        assert isinstance(a_north, float)
        assert isinstance(a_east, float)
        assert isinstance(a_down, float)
        
    def test_acceleration_lla_to_ned(self):
        """Test acceleration conversion from LLA to NED"""
        tracker = AccelerationTracker()
        
        # Set reference point
        tracker.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
        
        # Set acceleration as LLA second derivatives
        tracker.set_acceleration_lla_rates(lat_accel=0.00001, lon_accel=0.00001, alt_accel=-9.81)
        
        # Get acceleration in NED
        a_north, a_east, a_down = tracker.get_acceleration_ned()
        
        # Verify acceleration values are reasonable
        assert isinstance(a_north, float)
        assert isinstance(a_east, float)
        assert isinstance(a_down, float)


class TestIntegratedTracking:
    def test_combined_position_velocity_acceleration(self):
        """Test that position, velocity, and acceleration tracking work together"""
        pos_tracker = PositionTracker()
        vel_tracker = VelocityTracker()
        accel_tracker = AccelerationTracker()
        
        # Set common reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        pos_tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        vel_tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        accel_tracker.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set initial conditions
        pos_tracker.set_location_lla(lat=37.7750, lon=-122.4193, alt=105.0)
        vel_tracker.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=-1.0)
        accel_tracker.set_acceleration_ned(a_north=0.5, a_east=0.2, a_down=0.1)
        
        # Verify all systems can provide their respective outputs
        pos_ned = pos_tracker.get_location_ned()
        vel_ned = vel_tracker.get_velocity_ned()
        accel_ned = accel_tracker.get_acceleration_ned()
        
        assert len(pos_ned) == 3
        assert len(vel_ned) == 3
        assert len(accel_ned) == 3