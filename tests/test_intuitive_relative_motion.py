"""
Comprehensive and intuitive tests for relative motion calculations.

This test file focuses on realistic scenarios that thoroughly test position, 
velocity, acceleration, and orientation relative calculations in intuitive,
real-world contexts.
"""

import pytest
import numpy as np
import math
from vapor import Platform


class TestAircraftFormationFlying:
    """Test relative motion during aircraft formation flying scenarios"""
    
    def test_formation_turn_coordinated_maneuver(self):
        """Test formation flying during a coordinated 90-degree turn"""
        leader = Platform()
        wingman = Platform()
        
        # Set reference point (airfield)
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 3000.0  # 3000ft altitude
        leader.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        wingman.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial formation: leader ahead, wingman 50m behind and 30m right
        leader.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)
        wingman_lat = ref_lat + 0.001 - 0.00045  # 50m behind leader
        wingman_lon = ref_lon + 0.00035          # 30m right of leader
        wingman.set_position_lla(lat=wingman_lat, lon=wingman_lon, alt=ref_alt)
        
        # Both flying north initially at 100 m/s
        leader.set_velocity_ned(v_north=100.0, v_east=0.0, v_down=0.0)
        wingman.set_velocity_ned(v_north=100.0, v_east=0.0, v_down=0.0)
        
        # Both facing north initially
        leader.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        wingman.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Test initial formation
        leader.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        wingman.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        initial_relative_state = leader.get_relative_state(wingman)
        
        # Wingman should be behind and to the right
        assert initial_relative_state['position']['x'] < -40  # Behind (negative x)
        assert initial_relative_state['position']['y'] > 25   # Right (positive y) 
        assert abs(initial_relative_state['position']['z']) < 5  # Same altitude
        
        # No relative velocity in formation
        assert abs(initial_relative_state['velocity']['x']) < 1e-6
        assert abs(initial_relative_state['velocity']['y']) < 1e-6
        assert abs(initial_relative_state['velocity']['z']) < 1e-6
        
        # Now simulate coordinated right turn (90° turn with 2.5 m/s² centripetal acceleration)
        turn_radius = 100.0 ** 2 / 2.5  # v²/a = 4000m radius
        centripetal_accel = 2.5  # m/s²
        
        # 45 degrees into the turn (halfway through 90° turn)
        turn_angle = 45.0  # degrees
        
        # Leader and wingman both turning with same parameters
        # Centripetal acceleration points toward center of turn (west for right turn from north)
        leader.set_acceleration_ned(a_north=0.0, a_east=-centripetal_accel, a_down=0.0)
        wingman.set_acceleration_ned(a_north=0.0, a_east=-centripetal_accel, a_down=0.0)
        
        # Velocity during turn (45° heading = northeast)
        v_magnitude = 100.0
        v_north_turn = v_magnitude * math.cos(math.radians(turn_angle))
        v_east_turn = v_magnitude * math.sin(math.radians(turn_angle))
        
        leader.set_velocity_ned(v_north=v_north_turn, v_east=v_east_turn, v_down=0.0)
        wingman.set_velocity_ned(v_north=v_north_turn, v_east=v_east_turn, v_down=0.0)
        
        # Orientation during turn
        leader.set_orientation_euler_ned(roll=15.0, pitch=0.0, yaw=turn_angle)  # Banking into turn
        wingman.set_orientation_euler_ned(roll=15.0, pitch=0.0, yaw=turn_angle) # Same orientation
        
        turn_relative_state = leader.get_relative_state(wingman)
        
        # Formation should be maintained during coordinated turn
        # Relative position should be similar to initial (rotated by turn angle)
        expected_x = initial_relative_state['position']['x']
        expected_y = initial_relative_state['position']['y']
        
        # Allow for some variation due to coordinate frame rotation
        assert abs(turn_relative_state['position']['x'] - expected_x) < 10  # Within 10m
        assert abs(turn_relative_state['position']['y'] - expected_y) < 10
        
        # Relative velocity should still be near zero (coordinated maneuver)
        assert abs(turn_relative_state['velocity']['x']) < 1.0  # Within 1 m/s tolerance
        assert abs(turn_relative_state['velocity']['y']) < 1.0
        assert abs(turn_relative_state['velocity']['z']) < 1e-6
        
        # Relative acceleration should be near zero (same maneuver)
        assert abs(turn_relative_state['acceleration']['x']) < 0.1
        assert abs(turn_relative_state['acceleration']['y']) < 0.1
        assert abs(turn_relative_state['acceleration']['z']) < 1e-6
        
        # Relative orientation should be minimal (same bank angle and heading)
        quat = turn_relative_state['orientation']['quaternion']
        assert abs(quat['w'] - 1.0) < 0.1  # Near identity quaternion
        assert abs(quat['x']) < 0.1
        assert abs(quat['y']) < 0.1
        assert abs(quat['z']) < 0.1
    
    def test_formation_breakaway_maneuver(self):
        """Test relative motion when wingman breaks away from formation"""
        leader = Platform()
        wingman = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 3000.0
        leader.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        wingman.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial formation positions
        leader.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        wingman.set_position_lla(lat=ref_lat - 0.00045, lon=ref_lon + 0.00035, alt=ref_alt)
        
        # Leader continues straight and level
        leader.set_velocity_ned(v_north=100.0, v_east=0.0, v_down=0.0)
        leader.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        leader.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Wingman performs breakaway: climbing turn to the right
        wingman.set_velocity_ned(v_north=80.0, v_east=30.0, v_down=-10.0)  # Slowing, turning right, climbing
        wingman.set_acceleration_ned(a_north=-5.0, a_east=2.0, a_down=-2.0)  # Decelerating, turning, climbing
        wingman.set_orientation_euler_ned(roll=30.0, pitch=-10.0, yaw=20.0)  # Banking, climbing, turning
        
        relative_state = leader.get_relative_state(wingman)
        
        # Wingman should appear behind and to the right initially
        assert relative_state['position']['x'] < -40  # Behind
        assert relative_state['position']['y'] > 25   # Right
        
        # Relative velocity should show wingman moving away
        assert relative_state['velocity']['x'] < -15  # Moving backward relative to leader
        assert relative_state['velocity']['y'] > 25   # Moving right relative to leader  
        assert relative_state['velocity']['z'] < -5   # Moving up relative to leader
        
        # Relative acceleration should show divergence
        assert relative_state['acceleration']['x'] < -4  # Decelerating relative to leader
        assert relative_state['acceleration']['y'] > 1   # Accelerating right relative to leader
        assert relative_state['acceleration']['z'] < -1  # Accelerating up relative to leader
        
        # Relative orientation should show different attitudes
        quat = relative_state['orientation']['quaternion']
        # Should not be identity quaternion (significant relative rotation)
        quat_magnitude_change = abs(quat['w'] - 1.0) + abs(quat['x']) + abs(quat['y']) + abs(quat['z'])
        assert quat_magnitude_change > 0.2  # Significant relative orientation change


class TestOrbitalMechanicsScenarios:
    """Test relative motion in orbital mechanics scenarios"""
    
    def test_circular_orbital_motion(self):
        """Test satellite in circular orbit relative to ground station"""
        ground_station = Platform()
        satellite = Platform()
        
        # Ground station at reference point
        ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0  # Equator for simplicity
        ground_station.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        satellite.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Ground station stationary
        ground_station.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        ground_station.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        ground_station.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        ground_station.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Satellite in circular orbit (400 km altitude)
        orbital_altitude = 400000.0  # 400 km
        orbital_velocity = 7670.0    # ~7.67 km/s orbital velocity at 400 km
        
        # Test satellite at different orbital positions
        orbital_positions = [
            (0.0, 0.0),      # Directly overhead (0° in orbit)
            (90.0, 0.0),     # 90° in orbit (to the east)
            (180.0, 0.0),    # 180° in orbit (opposite side, not visible)
            (270.0, 0.0),    # 270° in orbit (to the west)
        ]
        
        for angle_deg, expected_visibility in orbital_positions:
            angle_rad = math.radians(angle_deg)
            
            # Calculate satellite position in circular orbit
            sat_lat = ref_lat
            sat_lon = ref_lon + math.degrees(angle_rad * orbital_altitude / 6371000.0)  # Rough approximation
            
            satellite.set_position_lla(lat=sat_lat, lon=sat_lon, alt=orbital_altitude)
            
            # Orbital velocity (tangent to orbit, pointing east at 0°, north at 90°, etc.)
            v_north = orbital_velocity * math.cos(angle_rad + math.pi/2)  # Perpendicular to radius
            v_east = orbital_velocity * math.sin(angle_rad + math.pi/2)
            
            satellite.set_velocity_ned(v_north=v_north, v_east=v_east, v_down=0.0)
            
            # Centripetal acceleration toward Earth center
            centripetal_accel = orbital_velocity ** 2 / (6371000.0 + orbital_altitude)
            a_north = -centripetal_accel * math.cos(angle_rad)
            a_east = -centripetal_accel * math.sin(angle_rad)
            
            satellite.set_acceleration_ned(a_north=a_north, a_east=a_east, a_down=centripetal_accel)
            satellite.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=angle_deg + 90)  # Facing velocity direction
            
            relative_state = ground_station.get_relative_state(satellite)
            
            # Verify orbital characteristics
            if angle_deg == 0.0:  # Directly overhead
                assert abs(relative_state['position']['x']) < 10000  # Near overhead
                assert abs(relative_state['position']['y']) < 10000
                assert relative_state['position']['z'] < -390000  # High above
                
                # Should be moving east at orbital velocity
                assert abs(relative_state['velocity']['x']) < 1000  # Minimal north-south
                assert relative_state['velocity']['y'] > 7000  # Fast eastward motion
                
            elif angle_deg == 90.0:  # To the east
                assert relative_state['position']['y'] > 300000  # Far to the east
                assert abs(relative_state['position']['z']) > 300000  # Below horizon
                
                # Should be moving north
                assert relative_state['velocity']['x'] > 7000  # Fast northward motion
                assert abs(relative_state['velocity']['y']) < 1000  # Minimal east-west
    
    def test_elliptical_orbital_motion(self):
        """Test satellite in elliptical orbit with varying velocity"""
        ground_station = Platform()
        satellite = Platform()
        
        # Reference at equator
        ref_lat, ref_lon, ref_alt = 0.0, 0.0, 0.0
        ground_station.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        satellite.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Ground station stationary
        ground_station.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        ground_station.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        ground_station.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        ground_station.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Elliptical orbit: perigee 400 km, apogee 1000 km
        perigee_alt = 400000.0
        apogee_alt = 1000000.0
        perigee_velocity = 8200.0  # Higher velocity at perigee
        apogee_velocity = 6800.0   # Lower velocity at apogee
        
        # Test at perigee (closest approach)
        satellite.set_position_lla(lat=ref_lat, lon=ref_lon, alt=perigee_alt)
        satellite.set_velocity_ned(v_north=0.0, v_east=perigee_velocity, v_down=0.0)
        
        # Higher centripetal acceleration at perigee
        perigee_centripetal = perigee_velocity ** 2 / (6371000.0 + perigee_alt)
        satellite.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=perigee_centripetal)
        satellite.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        perigee_relative_state = ground_station.get_relative_state(satellite)
        
        # Test at apogee (farthest point)
        satellite.set_position_lla(lat=ref_lat, lon=ref_lon + 10.0, alt=apogee_alt)  # Farther around orbit
        satellite.set_velocity_ned(v_north=0.0, v_east=apogee_velocity, v_down=0.0)
        
        # Lower centripetal acceleration at apogee
        apogee_centripetal = apogee_velocity ** 2 / (6371000.0 + apogee_alt)
        satellite.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=apogee_centripetal)
        satellite.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)
        
        apogee_relative_state = ground_station.get_relative_state(satellite)
        
        # Verify elliptical orbital characteristics
        # At perigee: closer, faster
        assert perigee_relative_state['position']['z'] > -410000  # Closer altitude
        assert perigee_relative_state['velocity']['y'] > 8000     # Faster velocity
        assert perigee_relative_state['acceleration']['z'] > 8.0  # Higher acceleration
        
        # At apogee: farther, slower
        assert apogee_relative_state['position']['z'] < -990000  # Farther altitude
        assert apogee_relative_state['velocity']['y'] < 7000    # Slower velocity
        assert apogee_relative_state['acceleration']['z'] < 5.0 # Lower acceleration


class TestNavalConvoyScenarios:
    """Test relative motion during naval convoy operations"""
    
    def test_convoy_formation_course_change(self):
        """Test convoy maintaining formation during course change"""
        lead_ship = Platform()
        escort_ship = Platform()
        
        # Set reference point (open ocean)
        ref_lat, ref_lon, ref_alt = 35.0, -140.0, 0.0  # Pacific Ocean
        lead_ship.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        escort_ship.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial formation: escort ship 500m to starboard (right) of lead ship
        lead_ship.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        escort_lat = ref_lat
        escort_lon = ref_lon + 0.005  # ~500m east
        escort_ship.set_position_lla(lat=escort_lat, lon=escort_lon, alt=ref_alt)
        
        # Both ships initially heading north at 15 m/s (30 knots)
        ship_speed = 15.0
        lead_ship.set_velocity_ned(v_north=ship_speed, v_east=0.0, v_down=0.0)
        escort_ship.set_velocity_ned(v_north=ship_speed, v_east=0.0, v_down=0.0)
        
        # No initial acceleration
        lead_ship.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        escort_ship.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Both facing north
        lead_ship.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        escort_ship.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        initial_relative_state = lead_ship.get_relative_state(escort_ship)
        
        # Escort should be to starboard (right, positive y)
        assert abs(initial_relative_state['position']['x']) < 50  # Abeam
        assert initial_relative_state['position']['y'] > 450     # To starboard
        assert abs(initial_relative_state['position']['z']) < 5  # Same sea level
        
        # No relative motion initially
        assert abs(initial_relative_state['velocity']['x']) < 1e-6
        assert abs(initial_relative_state['velocity']['y']) < 1e-6
        
        # Now simulate coordinated course change: 30° turn to starboard (right)
        turn_angle = 30.0  # degrees
        turn_rate = 2.0    # degrees per minute = slow naval turn
        
        # Ships maintain formation during turn
        new_heading_rad = math.radians(turn_angle)
        new_v_north = ship_speed * math.cos(new_heading_rad)
        new_v_east = ship_speed * math.sin(new_heading_rad)
        
        lead_ship.set_velocity_ned(v_north=new_v_north, v_east=new_v_east, v_down=0.0)
        escort_ship.set_velocity_ned(v_north=new_v_north, v_east=new_v_east, v_down=0.0)
        
        # Both ships have slight centripetal acceleration during turn
        turn_radius = ship_speed / math.radians(turn_rate / 60.0)  # Large radius for slow turn
        centripetal_accel = ship_speed ** 2 / turn_radius
        
        # Acceleration points toward center of turn (west for right turn from north)
        accel_north = -centripetal_accel * math.sin(new_heading_rad)
        accel_east = -centripetal_accel * math.cos(new_heading_rad)
        
        lead_ship.set_acceleration_ned(a_north=accel_north, a_east=accel_east, a_down=0.0)
        escort_ship.set_acceleration_ned(a_north=accel_north, a_east=accel_east, a_down=0.0)
        
        # Both ships turn to new heading
        lead_ship.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=turn_angle)
        escort_ship.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=turn_angle)
        
        turn_relative_state = lead_ship.get_relative_state(escort_ship)
        
        # Formation should be maintained during coordinated turn
        assert abs(turn_relative_state['position']['x'] - initial_relative_state['position']['x']) < 20
        assert abs(turn_relative_state['position']['y'] - initial_relative_state['position']['y']) < 20
        
        # Relative velocity should remain near zero (coordinated maneuver)
        assert abs(turn_relative_state['velocity']['x']) < 0.5  # Small tolerance for coordinate effects
        assert abs(turn_relative_state['velocity']['y']) < 0.5
        
        # Relative acceleration should be near zero (same maneuver)
        assert abs(turn_relative_state['acceleration']['x']) < 0.1
        assert abs(turn_relative_state['acceleration']['y']) < 0.1
    
    def test_emergency_evasive_maneuver(self):
        """Test convoy during emergency evasive maneuver"""
        lead_ship = Platform()
        following_ship = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 35.0, -140.0, 0.0
        lead_ship.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        following_ship.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Following ship 1000m behind lead ship
        lead_ship.set_position_lla(lat=ref_lat + 0.009, lon=ref_lon, alt=ref_alt)  # Lead ship ahead
        following_ship.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)     # Following ship behind
        
        # Both initially moving north at 20 m/s
        lead_ship.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        following_ship.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        
        # Lead ship performs emergency hard turn to port (left) and increases speed
        lead_ship.set_velocity_ned(v_north=15.0, v_east=-15.0, v_down=0.0)  # Turning left, slight speed increase
        lead_ship.set_acceleration_ned(a_north=-2.0, a_east=-8.0, a_down=0.0)  # Hard left turn, deceleration
        lead_ship.set_orientation_euler_ned(roll=-15.0, pitch=0.0, yaw=-30.0)  # Banking left, turning left
        
        # Following ship reacts slower (human reaction time)
        following_ship.set_velocity_ned(v_north=20.0, v_east=-2.0, v_down=0.0)  # Starting to turn
        following_ship.set_acceleration_ned(a_north=-1.0, a_east=-3.0, a_down=0.0)  # Gentler initial reaction
        following_ship.set_orientation_euler_ned(roll=-5.0, pitch=0.0, yaw=-10.0)  # Less aggressive turn
        
        relative_state = lead_ship.get_relative_state(following_ship)
        
        # Following ship should appear behind and to the right (since lead ship turned left)
        assert relative_state['position']['x'] < -800  # Behind
        assert relative_state['position']['y'] > 100   # To the right (lead ship turned left)
        
        # Relative velocity shows divergence
        assert relative_state['velocity']['x'] > 5   # Following ship approaching slower
        assert relative_state['velocity']['y'] > 10  # Following ship not turning as hard
        
        # Relative acceleration shows different maneuver intensities
        assert relative_state['acceleration']['x'] > 0.5  # Different longitudinal acceleration
        assert relative_state['acceleration']['y'] > 4    # Different lateral acceleration


class TestDroneSwarmCoordination:
    """Test relative motion in drone swarm scenarios"""
    
    def test_swarm_formation_reconfiguration(self):
        """Test drones reconfiguring from line formation to diamond formation"""
        drone1 = Platform()  # Leader
        drone2 = Platform()  # Left wing
        drone3 = Platform()  # Right wing
        drone4 = Platform()  # Trailer
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 200.0  # 200m altitude
        for drone in [drone1, drone2, drone3, drone4]:
            drone.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial line formation (all in a row, moving north)
        drone1.set_position_lla(lat=ref_lat + 0.0003, lon=ref_lon, alt=ref_alt)        # Lead
        drone2.set_position_lla(lat=ref_lat + 0.0002, lon=ref_lon, alt=ref_alt)        # 2nd
        drone3.set_position_lla(lat=ref_lat + 0.0001, lon=ref_lon, alt=ref_alt)        # 3rd  
        drone4.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)                 # Tail
        
        # All moving north at 15 m/s initially
        for drone in [drone1, drone2, drone3, drone4]:
            drone.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)
            drone.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
            drone.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Test initial line formation relative positions
        rel_1_to_2 = drone1.get_relative_state(drone2)
        rel_1_to_3 = drone1.get_relative_state(drone3)
        rel_1_to_4 = drone1.get_relative_state(drone4)
        
        # All drones should be behind leader in line
        assert rel_1_to_2['position']['x'] < -10  # Drone 2 behind
        assert rel_1_to_3['position']['x'] < -20  # Drone 3 further behind
        assert rel_1_to_4['position']['x'] < -30  # Drone 4 furthest behind
        
        # Minimal lateral separation initially
        assert abs(rel_1_to_2['position']['y']) < 5
        assert abs(rel_1_to_3['position']['y']) < 5
        assert abs(rel_1_to_4['position']['y']) < 5
        
        # Now reconfigure to diamond formation
        # Drone 1 continues straight (apex of diamond)
        drone1.set_velocity_ned(v_north=15.0, v_east=0.0, v_down=0.0)
        drone1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        
        # Drone 2 moves to left flank position
        drone2.set_velocity_ned(v_north=15.0, v_east=-5.0, v_down=0.0)  # Moving left while maintaining forward speed
        drone2.set_acceleration_ned(a_north=0.0, a_east=-2.0, a_down=0.0)  # Accelerating left
        drone2.set_orientation_euler_ned(roll=-10.0, pitch=0.0, yaw=-15.0)  # Banking left
        
        # Drone 3 moves to right flank position  
        drone3.set_velocity_ned(v_north=15.0, v_east=5.0, v_down=0.0)   # Moving right while maintaining forward speed
        drone3.set_acceleration_ned(a_north=0.0, a_east=2.0, a_down=0.0)   # Accelerating right
        drone3.set_orientation_euler_ned(roll=10.0, pitch=0.0, yaw=15.0)   # Banking right
        
        # Drone 4 becomes trailing element
        drone4.set_velocity_ned(v_north=12.0, v_east=0.0, v_down=0.0)   # Slightly slower to drop back
        drone4.set_acceleration_ned(a_north=-1.0, a_east=0.0, a_down=0.0)  # Decelerating to create trailing position
        drone4.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)     # Straight flight
        
        # Test diamond formation relative positions
        diamond_rel_1_to_2 = drone1.get_relative_state(drone2)
        diamond_rel_1_to_3 = drone1.get_relative_state(drone3)
        diamond_rel_1_to_4 = drone1.get_relative_state(drone4)
        
        # Verify diamond formation is developing
        # Drone 2 should be moving to the left
        assert diamond_rel_1_to_2['velocity']['y'] < -3  # Moving left relative to leader
        assert diamond_rel_1_to_2['acceleration']['y'] < -1  # Accelerating left
        
        # Drone 3 should be moving to the right
        assert diamond_rel_1_to_3['velocity']['y'] > 3   # Moving right relative to leader
        assert diamond_rel_1_to_3['acceleration']['y'] > 1   # Accelerating right
        
        # Drone 4 should be falling back
        assert diamond_rel_1_to_4['velocity']['x'] < -2  # Moving backward relative to leader
        assert diamond_rel_1_to_4['acceleration']['x'] < -0.5  # Decelerating relative to leader
    
    def test_swarm_obstacle_avoidance(self):
        """Test swarm coordinated obstacle avoidance maneuver"""
        drone1 = Platform()  # Leader (detects obstacle)
        drone2 = Platform()  # Follower
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 150.0
        drone1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        drone2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial formation: drone2 following 20m behind drone1
        drone1.set_position_lla(lat=ref_lat + 0.00018, lon=ref_lon, alt=ref_alt)  # Leader ahead
        drone2.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)            # Follower behind
        
        # Both initially flying north at 20 m/s
        drone1.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        drone2.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
        
        # Leader detects obstacle and initiates avoidance: climb and turn right
        drone1.set_velocity_ned(v_north=18.0, v_east=8.0, v_down=-5.0)  # Slight right turn and climb
        drone1.set_acceleration_ned(a_north=-1.0, a_east=3.0, a_down=-2.0)  # Decel, turn right, climb
        drone1.set_orientation_euler_ned(roll=15.0, pitch=-10.0, yaw=25.0)  # Bank right, pitch up, turn right
        
        # Follower reacts to leader's maneuver (follows with delay/modification)
        drone2.set_velocity_ned(v_north=19.0, v_east=5.0, v_down=-3.0)  # Less aggressive avoidance
        drone2.set_acceleration_ned(a_north=-0.5, a_east=2.0, a_down=-1.5)  # Gentler maneuver
        drone2.set_orientation_euler_ned(roll=10.0, pitch=-5.0, yaw=15.0)  # Less aggressive attitude
        
        relative_state = drone1.get_relative_state(drone2)
        
        # Follower should be behind and slightly to the left (leader turned right more)
        assert relative_state['position']['x'] < -15  # Behind
        assert relative_state['position']['y'] < 5    # Slightly left or straight
        assert relative_state['position']['z'] > 0    # Below (leader climbed more)
        
        # Relative velocity shows follower's less aggressive avoidance
        assert relative_state['velocity']['x'] > 0.5  # Follower catching up (less deceleration)
        assert relative_state['velocity']['y'] < -2   # Follower not turning as hard right
        assert relative_state['velocity']['z'] > 1    # Follower not climbing as fast
        
        # Relative acceleration shows different avoidance strategies
        assert relative_state['acceleration']['x'] > 0.3  # Different deceleration rates
        assert relative_state['acceleration']['y'] < -0.8 # Different turn rates
        assert relative_state['acceleration']['z'] > 0.3  # Different climb rates


class TestRealWorldIntegrationScenarios:
    """Test complex real-world scenarios integrating all motion types"""
    
    def test_aircraft_approach_and_landing(self):
        """Test aircraft approach relative to runway/ground control"""
        aircraft = Platform()
        ground_control = Platform()
        
        # Ground control at airport
        airport_lat, airport_lon, airport_alt = 37.6213, -122.3790, 4.0  # SFO coordinates
        ground_control.set_reference_lla(lat=airport_lat, lon=airport_lon, alt=airport_alt)
        aircraft.set_reference_lla(lat=airport_lat, lon=airport_lon, alt=airport_alt)
        
        # Ground control stationary at airport
        ground_control.set_position_lla(lat=airport_lat, lon=airport_lon, alt=airport_alt)
        ground_control.set_velocity_ned(v_north=0.0, v_east=0.0, v_down=0.0)
        ground_control.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
        ground_control.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)
        
        # Aircraft on final approach: 5 miles out, 1500 ft altitude, 3° glide slope
        approach_distance = 8000.0  # ~5 miles in meters
        approach_altitude = 457.0   # 1500 ft in meters
        approach_speed = 70.0       # ~140 knots approach speed
        glide_slope_angle = 3.0     # degrees
        
        # Calculate aircraft position on approach
        aircraft_lat = airport_lat - (approach_distance * math.cos(math.radians(0))) / 111000.0  # South of airport
        aircraft_lon = airport_lon
        aircraft_alt = airport_alt + approach_altitude
        
        aircraft.set_position_lla(lat=aircraft_lat, lon=aircraft_lon, alt=aircraft_alt)
        
        # Aircraft velocity: approaching from south, descending on glide slope
        v_north = approach_speed * math.cos(math.radians(glide_slope_angle))  # Forward component
        v_down = approach_speed * math.sin(math.radians(glide_slope_angle))   # Descent component
        
        aircraft.set_velocity_ned(v_north=v_north, v_east=0.0, v_down=v_down)
        
        # Aircraft on stable approach (minimal acceleration)
        aircraft.set_acceleration_ned(a_north=-0.5, a_east=0.0, a_down=0.2)  # Slight deceleration, steady descent
        
        # Aircraft oriented on approach heading with slight nose-down attitude
        aircraft.set_orientation_euler_ned(roll=0.0, pitch=glide_slope_angle, yaw=0.0)
        
        relative_state = ground_control.get_relative_state(aircraft)
        
        # Verify approach geometry
        assert relative_state['position']['x'] < -7500  # Aircraft approaching from south
        assert abs(relative_state['position']['y']) < 100  # On centerline
        assert relative_state['position']['z'] < -400  # Above airport
        
        # Verify approach dynamics
        assert relative_state['velocity']['x'] > 65  # Approaching at correct speed
        assert abs(relative_state['velocity']['y']) < 5  # Staying on centerline
        assert relative_state['velocity']['z'] > 3   # Descending at correct rate
        
        # Verify approach is stabilized (low acceleration)
        assert abs(relative_state['acceleration']['x']) < 1.0  # Stable approach speed
        assert abs(relative_state['acceleration']['y']) < 0.5  # Stable lateral position
        assert abs(relative_state['acceleration']['z']) < 0.5  # Stable descent rate
    
    def test_multi_platform_rendezvous(self):
        """Test coordinated rendezvous between two moving platforms"""
        platform_a = Platform()
        platform_b = Platform()
        
        # Set reference point (meeting point)
        rendezvous_lat, rendezvous_lon, rendezvous_alt = 37.7749, -122.4194, 1000.0
        platform_a.set_reference_lla(lat=rendezvous_lat, lon=rendezvous_lon, alt=rendezvous_alt)
        platform_b.set_reference_lla(lat=rendezvous_lat, lon=rendezvous_lon, alt=rendezvous_alt)
        
        # Platform A approaching from the west
        platform_a_lat = rendezvous_lat
        platform_a_lon = rendezvous_lon - 0.01  # ~1 km west
        platform_a.set_position_lla(lat=platform_a_lat, lon=platform_a_lon, alt=rendezvous_alt)
        
        # Platform B approaching from the south
        platform_b_lat = rendezvous_lat - 0.009  # ~1 km south
        platform_b_lon = rendezvous_lon
        platform_b.set_position_lla(lat=platform_b_lat, lon=platform_b_lon, alt=rendezvous_alt)
        
        # Both platforms timing their approach for simultaneous arrival
        # Platform A moving east at 20 m/s
        platform_a.set_velocity_ned(v_north=0.0, v_east=20.0, v_down=0.0)
        platform_a.set_acceleration_ned(a_north=0.0, a_east=-0.5, a_down=0.0)  # Slight deceleration
        platform_a.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=90.0)  # Facing east
        
        # Platform B moving north at 18 m/s (slightly slower to coordinate timing)
        platform_b.set_velocity_ned(v_north=18.0, v_east=0.0, v_down=0.0)
        platform_b.set_acceleration_ned(a_north=-0.3, a_east=0.0, a_down=0.0)  # Slight deceleration
        platform_b.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)   # Facing north
        
        relative_state = platform_a.get_relative_state(platform_b)
        
        # Platform B should appear ahead and to the right from Platform A's perspective
        assert relative_state['position']['x'] > 800   # Ahead (Platform B will arrive first at current speeds)
        assert relative_state['position']['y'] > 800   # To the right (south from east-facing A's perspective)
        assert abs(relative_state['position']['z']) < 10  # Same altitude
        
        # Relative velocity shows convergence pattern
        assert relative_state['velocity']['x'] < -15  # Platform B approaching faster northward
        assert relative_state['velocity']['y'] < -18  # Platform B moving away from A's right side
        
        # Both decelerating for rendezvous
        assert relative_state['acceleration']['x'] > 0.1  # Different deceleration patterns
        assert abs(relative_state['acceleration']['y']) < 0.6  # Minimal lateral acceleration difference
        
        # Relative orientation shows perpendicular approach angles
        quat = relative_state['orientation']['quaternion']
        # 90-degree difference in heading
        expected_90deg_quat_w = math.cos(math.radians(45))  # For 90-degree rotation
        assert abs(quat['w'] - expected_90deg_quat_w) < 0.1
    
    def test_pursuit_and_evasion_scenario(self):
        """Test dynamic pursuit scenario with evasive maneuvers"""
        pursuer = Platform()
        evader = Platform()
        
        # Set reference point
        ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 500.0
        pursuer.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        evader.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
        
        # Initial positions: evader ahead, pursuer behind
        evader.set_position_lla(lat=ref_lat + 0.005, lon=ref_lon, alt=ref_alt)    # Evader ahead
        pursuer.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)           # Pursuer behind
        
        # Evader performing evasive maneuver: climbing spiral to the right
        spiral_speed = 80.0
        spiral_angle = 30.0  # degrees from north
        climb_rate = 10.0    # m/s
        
        evader_v_north = spiral_speed * math.cos(math.radians(spiral_angle))
        evader_v_east = spiral_speed * math.sin(math.radians(spiral_angle))
        evader.set_velocity_ned(v_north=evader_v_north, v_east=evader_v_east, v_down=-climb_rate)
        
        # Evader's acceleration: centripetal (for spiral) plus vertical acceleration
        spiral_radius = 500.0  # meters
        centripetal_accel = spiral_speed ** 2 / spiral_radius
        evader.set_acceleration_ned(a_north=-centripetal_accel * math.sin(math.radians(spiral_angle)),
                                  a_east=centripetal_accel * math.cos(math.radians(spiral_angle)),
                                  a_down=-2.0)  # Increasing climb rate
        
        evader.set_orientation_euler_ned(roll=45.0, pitch=-15.0, yaw=spiral_angle)  # Banking, climbing, turning
        
        # Pursuer attempting to intercept: higher speed, direct intercept course
        intercept_speed = 100.0
        intercept_angle = 45.0  # Leading the target
        
        pursuer_v_north = intercept_speed * math.cos(math.radians(intercept_angle))
        pursuer_v_east = intercept_speed * math.sin(math.radians(intercept_angle))
        pursuer.set_velocity_ned(v_north=pursuer_v_north, v_east=pursuer_v_east, v_down=-8.0)  # Also climbing
        
        # Pursuer accelerating to intercept
        pursuer.set_acceleration_ned(a_north=2.0, a_east=3.0, a_down=-1.5)  # Accelerating toward intercept
        pursuer.set_orientation_euler_ned(roll=20.0, pitch=-10.0, yaw=intercept_angle)  # Pursuing attitude
        
        relative_state = pursuer.get_relative_state(evader)
        
        # Evader should be ahead and to the right, climbing away
        assert relative_state['position']['x'] > 380   # Ahead
        assert relative_state['position']['y'] > 200   # To the right
        assert relative_state['position']['z'] < -50   # Above (climbing)
        
        # Relative velocity shows pursuit dynamics
        assert relative_state['velocity']['x'] < -15  # Evader pulling away forward
        assert relative_state['velocity']['y'] < -10  # Evader moving right relative to pursuer
        assert relative_state['velocity']['z'] < -1   # Evader climbing faster
        
        # Relative acceleration shows dynamic maneuvers
        assert abs(relative_state['acceleration']['x']) > 2   # Different longitudinal accelerations
        assert abs(relative_state['acceleration']['y']) > 2   # Different lateral accelerations  
        assert abs(relative_state['acceleration']['z']) > 0.3 # Different vertical accelerations
        
        # Complex relative orientation due to different maneuvers
        quat = relative_state['orientation']['quaternion']
        orientation_difference = abs(quat['w'] - 1.0) + abs(quat['x']) + abs(quat['y']) + abs(quat['z'])
        assert orientation_difference > 0.3  # Significant relative orientation difference