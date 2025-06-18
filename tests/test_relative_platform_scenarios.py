import pytest
import numpy as np
import math
from vapor import Platform


class TestSameDirectionSameSpeed:
    def test_identical_platforms_relative_state(self):
        """Test that two identical platforms show zero relative state"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set identical positions
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Set identical velocities (both moving north at 20 m/s)
        platform1.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        
        # Set identical accelerations
        platform1.set_acceleration_ned(a_north=1.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=1.0, a_east=0.0, a_down=0.0)
        
        # Set identical orientations (both facing north)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # All relative values should be zero
        assert abs(relative_state['position']['x']) < 1e-3
        assert abs(relative_state['position']['y']) < 1e-3
        assert abs(relative_state['position']['z']) < 1e-3
        
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        assert abs(relative_state['acceleration']['x']) < 1e-6
        assert abs(relative_state['acceleration']['y']) < 1e-6
        assert abs(relative_state['acceleration']['z']) < 1e-6
        
        # Relative orientation should be identity quaternion
        quat = relative_state['orientation']['quaternion']
        assert abs(quat['w'] - 1.0) < 1e-6
        assert abs(quat['x']) < 1e-6
        assert abs(quat['y']) < 1e-6
        assert abs(quat['z']) < 1e-6
    
    def test_parallel_motion_different_positions(self):
        """Test platforms moving in parallel but at different positions"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set same reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Platform 1 at reference, Platform 2 offset 100m east
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon + 0.001, alt=ref_alt)  # ~100m east
        
        # Both moving north at same speed
        platform1.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)
        
        # Same acceleration
        platform1.set_acceleration_ned(a_north=0.5, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.5, a_east=0.0, a_down=0.0)
        
        # Same orientation (both facing north)
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Position should show platform2 to the right (positive y in body frame)
        assert abs(relative_state['position']['x']) < 10  # Should be minimal forward/back
        assert relative_state['position']['y'] > 80  # Should be to the right (east)
        assert abs(relative_state['position']['z']) < 1e-3  # Same altitude
        
        # Velocity should be zero (same relative motion)
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        # Acceleration should be zero (same relative acceleration)
        assert abs(relative_state['acceleration']['x']) < 1e-6
        assert abs(relative_state['acceleration']['y']) < 1e-6
        assert abs(relative_state['acceleration']['z']) < 1e-6
    
    def test_formation_flying_scenario(self):
        """Test two aircraft in formation (same speed, direction, but offset)"""
        leader = Platform()
        wingman = Platform()
        
        # Set reference point (airfield)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 1000.0
        leader.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        wingman.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Leader position
        leader.set_position_lla(lat=ref_lat + 0.01, lon=ref_lon + 0.01, alt=ref_alt + 500)
        # Wingman 50m behind and 30m to the right
        wingman_lat = ref_lat + 0.01 - 0.00045  # ~50m south (behind)
        wingman_lon = ref_lon + 0.01 + 0.00035  # ~30m east (right)
        wingman.set_position_lla(lat=wingman_lat, lon=wingman_lon, alt=ref_alt + 500)
        
        # Both flying northeast at 100 m/s
        v_north, v_east = 70.7, 70.7  # 100 m/s at 45 degrees
        leader.set_velocity_ned(v_north=v_north, v_east=v_east, v_down=0.0)
        wingman.set_velocity_ned(v_north=v_north, v_east=v_east, v_down=0.0)
        
        # Both turning with same acceleration (banking left)
        leader.set_acceleration_ned(a_north=-2.0, a_east=3.0, a_down=0.0)
        wingman.set_acceleration_ned(a_north=-2.0, a_east=3.0, a_down=0.0)
        
        # Both facing northeast (45 degree heading)
        leader.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=45.0)
        wingman.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=45.0)
        
        # Get relative state from leader's perspective
        relative_state = leader.get_relative_state(wingman)
        
        # Wingman should appear behind and to the right in leader's body frame
        assert relative_state['position']['x'] < -10  # Behind (negative x)
        assert relative_state['position']['y'] > 10   # Right (positive y)
        assert abs(relative_state['position']['z']) < 10  # Same altitude
        
        # Relative velocity should be zero (formation flying)
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        # Relative acceleration should be zero (same maneuver)
        assert abs(relative_state['acceleration']['x']) < 1e-6
        assert abs(relative_state['acceleration']['y']) < 1e-6
        assert abs(relative_state['acceleration']['z']) < 1e-6


class TestDifferentVelocitiesAndAccelerations:
    def test_head_on_collision_course(self):
        """Test two platforms approaching each other head-on"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Platform 1 at reference, Platform 2 1km north
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat + 0.009, lon=ref_lon, alt=ref_alt)  # ~1km north
        
        # Platform 1 moving north at 30 m/s, Platform 2 moving south at 20 m/s
        platform1.set_velocity_ned(v_north=30.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=-20.0, v_east=0.0, v_down=0.0)
        
        # No acceleration
        platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Platform 1 facing north, Platform 2 facing south
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=180.0)
        
        # Get relative state from platform1's perspective
        relative_state = platform1.get_relative_state(platform2)
        
        # Platform 2 should be ahead (positive x)
        assert relative_state['position']['x'] > 900  # Ahead ~1km
        assert abs(relative_state['position']['y']) < 10  # Directly ahead
        assert abs(relative_state['position']['z']) < 0.1  # Same altitude
        
        # Relative velocity should show platform2 approaching at 50 m/s
        assert relative_state['velocity']['x'] < -49  # Approaching (negative x)
        assert abs(relative_state['velocity']['y']) < 1e-6  # No lateral motion
        assert abs(relative_state['velocity']['z']) < 1e-6  # No vertical motion
        
        # Relative orientation should show 180 degree difference
        quat = relative_state['orientation']['quaternion']
        # For 180 degree yaw difference, expect w≈0, z≈±1
        assert abs(quat['w']) < 0.1
        assert abs(abs(quat['z']) - 1.0) < 0.1
    
    def test_overtaking_scenario(self):
        """Test one platform overtaking another from behind"""
        slower_platform = Platform()
        faster_platform = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        slower_platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        faster_platform.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Slower platform ahead, faster platform behind
        slower_platform.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)
        faster_platform.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Slower platform at 15 m/s, faster at 25 m/s (both north)
        slower_platform.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)
        faster_platform.set_velocity_ned(v_north=25.0, v_east=0.0, v_down=0.0)
        
        # Faster platform accelerating to overtake
        slower_platform.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        faster_platform.set_acceleration_ned(a_north=2.0, a_east=0.0, a_down=0.0)
        
        # Both facing north
        slower_platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        faster_platform.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state from slower platform's perspective
        relative_state = slower_platform.get_relative_state(faster_platform)
        
        # Faster platform should be behind (negative x)
        assert relative_state['position']['x'] < -90  # Behind
        assert abs(relative_state['position']['y']) < 10  # Directly behind
        
        # Relative velocity should show faster platform approaching from behind
        assert relative_state['velocity']['x'] > 9  # Approaching (positive x)
        assert abs(relative_state['velocity']['y']) < 1e-6
        
        # Relative acceleration should show faster platform accelerating toward
        assert relative_state['acceleration']['x'] > 1.9  # Accelerating forward
        assert abs(relative_state['acceleration']['y']) < 1e-6
    
    def test_perpendicular_crossing_paths(self):
        """Test two platforms with perpendicular crossing paths"""
        platform_north = Platform()
        platform_east = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform_north.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform_east.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Position platforms to cross at reference point
        platform_north.set_position_lla(lat=ref_lat - 0.002, lon=ref_lon, alt=ref_alt)  # South
        platform_east.set_position_lla(lat=ref_lat, lon=ref_lon - 0.002, alt=ref_alt)   # West
        
        # Platform going north at 20 m/s, platform going east at 15 m/s
        platform_north.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        platform_east.set_velocity_ned(v_north=0.0, v_east=15.0, v_down=0.0)
        
        # No acceleration
        platform_north.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform_east.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Platform north facing north, platform east facing east
        platform_north.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform_east.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        # Get relative state from north-bound platform's perspective
        relative_state = platform_north.get_relative_state(platform_east)
        
        # East-bound platform should be to the left and ahead  
        assert relative_state['position']['x'] > 200  # Ahead 
        assert relative_state['position']['y'] < -150  # To the left (west from north platform's perspective)
        
        # Relative velocity should show eastward motion
        assert relative_state['velocity']['x'] < -15  # Moving away in north direction
        assert relative_state['velocity']['y'] > 14  # Moving right (east)
        assert abs(relative_state['velocity']['z']) < 1e-6


class TestDifferentOrientations:
    def test_same_position_different_headings(self):
        """Test platforms at same position but facing different directions"""
        platform1 = Platform()
        platform2 = Platform()
        
        # Set reference point and same position
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        platform2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Both stationary
        platform1.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        platform2.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        
        platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Platform 1 facing north, Platform 2 facing east
        platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        # Get relative state
        relative_state = platform1.get_relative_state(platform2)
        
        # Position should be zero (same location)
        assert abs(relative_state['position']['x']) < 1e-3
        assert abs(relative_state['position']['y']) < 1e-3
        assert abs(relative_state['position']['z']) < 1e-3
        
        # Velocity should be zero (both stationary)
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        assert abs(relative_state['velocity']['z']) < 1e-6
        
        # Orientation should show 90 degree yaw difference
        quat = relative_state['orientation']['quaternion']
        # For 90 degree yaw rotation: w = cos(45°), z = sin(45°)
        expected_w = math.cos(math.radians(45))
        expected_z = math.sin(math.radians(45))
        assert abs(quat['w'] - expected_w) < 1e-5
        assert abs(quat['x']) < 1e-5
        assert abs(quat['y']) < 1e-5
        assert abs(quat['z'] - expected_z) < 1e-5
    
    def test_observer_rotation_effect_on_relative_position(self):
        """Test how observer rotation affects relative position calculations"""
        observer = Platform()
        target = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        observer.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        target.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Observer at reference, target 100m north
        observer.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        target.set_position_lla(lat=ref_lat + 0.0009, lon=ref_lon, alt=ref_alt)
        
        # Both stationary
        observer.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        target.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        
        observer.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        target.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        target.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Test different observer orientations
        test_cases = [
            (0.0, "north"),    # Observer facing north
            (90.0, "east"),    # Observer facing east  
            (180.0, "south"),  # Observer facing south
            (270.0, "west")    # Observer facing west
        ]
        
        for yaw, direction in test_cases:
            observer.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=yaw)
            relative_state = observer.get_relative_state(target)
            
            if direction == "north":
                # Target should be ahead (positive x)
                assert relative_state['position']['x'] > 90
                assert abs(relative_state['position']['y']) < 10
            elif direction == "east":
                # Target should be to the left (negative y)
                assert abs(relative_state['position']['x']) < 10
                assert relative_state['position']['y'] < -90
            elif direction == "south":
                # Target should be behind (negative x)
                assert relative_state['position']['x'] < -90
                assert abs(relative_state['position']['y']) < 10
            elif direction == "west":
                # Target should be to the right (positive y)
                assert abs(relative_state['position']['x']) < 10
                assert relative_state['position']['y'] > 90


class TestComplexMultiPlatformScenarios:
    def test_aircraft_carrier_landing_approach(self):
        """Test aircraft approaching carrier for landing"""
        carrier = Platform()
        aircraft = Platform()
        
        # Set reference point (open ocean)
        ref_lat, ref_lon, ref_alt = 37.0, -123.0, 0.0
        carrier.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        aircraft.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Carrier position and motion
        carrier.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        carrier.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)  # 30 knots north
        carrier.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        carrier.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)  # Heading north
        
        # Aircraft on approach (3 miles behind, 1000 ft altitude)
        aircraft_lat = ref_lat - 0.045  # ~3 miles south
        aircraft.set_position_lla(lat=aircraft_lat, lon=ref_lon, alt=300.0)  # 1000 ft
        aircraft.set_velocity_ned(v_north=35.0, v_east=0.0, v_down=-2.0)  # Approaching and descending
        aircraft.set_acceleration_ned(a_north=-1.0, a_east=0.0, a_down=-0.5)  # Decelerating
        aircraft.set_orientation_euler_ned(roll=0.0, pitch=-3.0, yaw=0.0)  # Slight nose down
        
        # Get relative state from carrier's perspective
        relative_state = carrier.get_relative_state(aircraft)
        
        # Aircraft should be behind and above
        assert relative_state['position']['x'] < -4000  # Behind (3+ miles)
        assert abs(relative_state['position']['y']) < 100  # On centerline
        assert relative_state['position']['z'] < -250  # Above (negative z = up)
        
        # Relative velocity should show aircraft approaching
        assert relative_state['velocity']['x'] > 15  # Approaching (20 m/s closure)
        assert abs(relative_state['velocity']['y']) < 5  # On track
        assert relative_state['velocity']['z'] < 0  # Descending relative to carrier
        
        # Relative acceleration should show aircraft slowing down
        assert relative_state['acceleration']['x'] < -0.5  # Decelerating approach
        assert relative_state['acceleration']['z'] < 0  # Increasing descent rate
    
    def test_convoy_vehicle_spacing(self):
        """Test vehicles in convoy maintaining spacing"""
        lead_vehicle = Platform()
        follow_vehicle = Platform()
        
        # Set reference point (highway)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        lead_vehicle.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        follow_vehicle.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Lead vehicle position
        lead_vehicle.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)
        # Following vehicle 50m behind
        follow_vehicle.set_position_lla(lat=ref_lat + 0.001 - 0.00045, lon=ref_lon, alt=ref_alt)
        
        # Both traveling north at highway speed (30 m/s = 67 mph)
        highway_speed = 30.0
        lead_vehicle.set_velocity_ned(v_north=highway_speed, v_east=0.0, v_down=0.0)
        follow_vehicle.set_velocity_ned(v_north=highway_speed, v_east=0.0, v_down=0.0)
        
        # Lead vehicle starts braking, following vehicle reacts with delay
        lead_vehicle.set_acceleration_ned(a_north=-3.0, a_east=0.0, a_down=0.0)  # Braking
        follow_vehicle.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)  # Reaction delay
        
        # Both facing north
        lead_vehicle.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        follow_vehicle.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Get relative state from following vehicle's perspective
        relative_state = follow_vehicle.get_relative_state(lead_vehicle)
        
        # Lead vehicle should be ahead
        assert relative_state['position']['x'] > 40  # Ahead ~50m
        assert abs(relative_state['position']['y']) < 5  # Same lane
        assert abs(relative_state['position']['z']) < 1  # Same road level
        
        # Initially no relative velocity (same speed)
        assert abs(relative_state['velocity']['x']) < 1e-6
        assert abs(relative_state['velocity']['y']) < 1e-6
        
        # Relative acceleration shows lead vehicle slowing down
        assert relative_state['acceleration']['x'] < -2.5  # Lead vehicle decelerating
    
    def test_satellite_tracking_ground_station(self):
        """Test satellite motion relative to ground station"""
        ground_station = Platform()
        satellite = Platform()
        
        # Set reference point (ground station location)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
        ground_station.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        satellite.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Ground station stationary
        ground_station.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        ground_station.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        ground_station.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        ground_station.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Low Earth orbit satellite (400 km altitude, moving east)
        satellite.set_position_lla(lat=ref_lat, lon=ref_lon, alt=400000.0)
        satellite.set_velocity_ned(v_north=0.0, v_east=7700.0, v_down=0.0)  # ~7.7 km/s orbital velocity
        satellite.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=-9.8)  # Gravity (simplified)
        satellite.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)  # Facing east
        
        # Get relative state from ground station's perspective
        relative_state = ground_station.get_relative_state(satellite)
        
        # Satellite should be directly overhead
        assert abs(relative_state['position']['x']) < 1000  # Directly overhead
        assert abs(relative_state['position']['y']) < 1000  # Directly overhead
        assert relative_state['position']['z'] < -399000  # Far above (negative z = up)
        
        # Relative velocity should show eastward motion
        assert abs(relative_state['velocity']['x']) < 100  # Minimal north-south
        assert relative_state['velocity']['y'] > 7600  # Fast eastward motion
        assert abs(relative_state['velocity']['z']) < 100  # Minimal vertical
        
        # Satellite orientation should show 90 degree difference from ground station
        quat = relative_state['orientation']['quaternion']
        expected_w = math.cos(math.radians(45))  # 90 degree yaw difference
        expected_z = math.sin(math.radians(45))
        assert abs(quat['w'] - expected_w) < 1e-5
        assert abs(quat['z'] - expected_z) < 1e-5