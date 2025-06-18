#!/usr/bin/env python3
"""
VAPOr - Velocity, Acceleration, Position, Orientation Example

This script demonstrates the core functionality of the VAPOr library,
including coordinate system tracking and relative state calculations.
"""

import numpy as np
from vapor import Platform, PositionTracker, VelocityTracker, AccelerationTracker, OrientationTracker


def demonstrate_individual_trackers():
    """Demonstrate individual tracker functionality"""
    print("=" * 60)
    print("INDIVIDUAL TRACKER EXAMPLES")
    print("=" * 60)
    
    # Set up reference point (San Francisco)
    ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
    
    # Position Tracking
    print("\n1. Position Tracking:")
    print("-" * 30)
    pos_tracker = PositionTracker()
    pos_tracker.set_reference_lla(ref_lat, ref_lon, ref_alt)
    
    # Set position slightly north and east of reference
    pos_tracker.set_location_lla(lat=37.7750, lon=-122.4193, alt=105.0)
    
    wcs_pos = pos_tracker.get_location_wcs()
    lla_pos = pos_tracker.get_location_lla()
    ned_pos = pos_tracker.get_location_ned()
    
    print(f"WCS (ECEF):  x={wcs_pos[0]:,.1f}m, y={wcs_pos[1]:,.1f}m, z={wcs_pos[2]:,.1f}m")
    print(f"LLA:         lat={lla_pos[0]:.6f}°, lon={lla_pos[1]:.6f}°, alt={lla_pos[2]:.1f}m")
    print(f"NED:         north={ned_pos[0]:.1f}m, east={ned_pos[1]:.1f}m, down={ned_pos[2]:.1f}m")
    
    # Velocity Tracking
    print("\n2. Velocity Tracking:")
    print("-" * 30)
    vel_tracker = VelocityTracker()
    vel_tracker.set_reference_lla(ref_lat, ref_lon, ref_alt)
    vel_tracker.set_velocity_ned(v_north=15.0, v_east=10.0, v_down=-2.0)
    
    ned_vel = vel_tracker.get_velocity_ned()
    wcs_vel = vel_tracker.get_velocity_wcs()
    
    print(f"NED Velocity: north={ned_vel[0]:.1f}m/s, east={ned_vel[1]:.1f}m/s, down={ned_vel[2]:.1f}m/s")
    print(f"WCS Velocity: vx={wcs_vel[0]:.1f}m/s, vy={wcs_vel[1]:.1f}m/s, vz={wcs_vel[2]:.1f}m/s")
    
    # Orientation Tracking
    print("\n3. Orientation Tracking:")
    print("-" * 30)
    orient_tracker = OrientationTracker()
    orient_tracker.set_orientation_euler_ned(roll=5.0, pitch=-2.0, yaw=45.0)
    
    euler = orient_tracker.get_orientation_euler_ned()
    quat = orient_tracker.get_orientation_quaternion_ned()
    
    print(f"Euler Angles: roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}°")
    print(f"Quaternion:   w={quat[0]:.3f}, x={quat[1]:.3f}, y={quat[2]:.3f}, z={quat[3]:.3f}")


def demonstrate_platform_integration():
    """Demonstrate integrated Platform functionality"""
    print("\n\n" + "=" * 60)
    print("PLATFORM INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Create a platform for an aircraft
    aircraft = Platform()
    
    # Set reference point (airport)
    airport_lat, airport_lon, airport_alt = 37.7749, -122.4194, 100.0
    aircraft.set_reference_lla(airport_lat, airport_lon, airport_alt)
    
    # Aircraft is 5km north, 2km east, at 3000ft altitude
    aircraft.set_position_ned(
        north=5000.0,    # 5km north
        east=2000.0,     # 2km east
        down=-914.0      # 3000ft = 914m above reference
    )
    
    # Aircraft heading northeast at 150 knots (77 m/s)
    velocity_magnitude = 77.0  # m/s
    heading = 45.0  # degrees
    v_north = velocity_magnitude * np.cos(np.radians(heading))
    v_east = velocity_magnitude * np.sin(np.radians(heading))
    aircraft.set_velocity_ned(v_north=v_north, v_east=v_east, v_down=-5.0)
    
    # Aircraft banking right with slight deceleration
    aircraft.set_acceleration_ned(a_north=-1.0, a_east=2.0, a_down=0.0)
    
    # Aircraft orientation: slight right bank, nose up, heading northeast
    aircraft.set_orientation_euler_ned(roll=10.0, pitch=3.0, yaw=45.0)
    
    # Display aircraft state
    print("\nAircraft State:")
    print("-" * 20)
    pos_lla = aircraft.get_position_lla()
    pos_ned = aircraft.get_position_ned()
    vel_ned = aircraft.get_velocity_ned()
    accel_ned = aircraft.get_acceleration_ned()
    euler = aircraft.get_orientation_euler_ned()
    
    print(f"Position (LLA): {pos_lla[0]:.6f}°N, {pos_lla[1]:.6f}°W, {pos_lla[2]:.0f}m")
    print(f"Position (NED): {pos_ned[0]:.0f}m N, {pos_ned[1]:.0f}m E, {pos_ned[2]:.0f}m D")
    print(f"Velocity (NED): {vel_ned[0]:.1f}m/s N, {vel_ned[1]:.1f}m/s E, {vel_ned[2]:.1f}m/s D")
    print(f"Speed: {np.linalg.norm(vel_ned[:2]):.1f} m/s ({np.linalg.norm(vel_ned[:2]) * 1.94384:.0f} knots)")
    print(f"Acceleration:   {accel_ned[0]:.1f}m/s² N, {accel_ned[1]:.1f}m/s² E, {accel_ned[2]:.1f}m/s² D")
    print(f"Orientation:    {euler[0]:.1f}° roll, {euler[1]:.1f}° pitch, {euler[2]:.1f}° yaw")


def demonstrate_relative_state_calculation():
    """Demonstrate relative state calculations between platforms"""
    print("\n\n" + "=" * 60)
    print("RELATIVE STATE CALCULATION EXAMPLE")
    print("=" * 60)
    
    # Create two platforms: leader and wingman aircraft
    leader = Platform()
    wingman = Platform()
    
    # Set common reference point
    ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 1000.0
    leader.set_reference_lla(ref_lat, ref_lon, ref_alt)
    wingman.set_reference_lla(ref_lat, ref_lon, ref_alt)
    
    # Leader aircraft position and state
    leader.set_position_lla(lat=ref_lat + 0.01, lon=ref_lon + 0.01, alt=ref_alt + 1500)
    leader.set_velocity_ned(v_north=60.0, v_east=60.0, v_down=0.0)  # Flying northeast
    leader.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
    leader.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=45.0)  # Heading northeast
    
    # Wingman aircraft - in formation, 100m behind and 50m to the right
    wingman_lat = ref_lat + 0.01 - 0.0009  # 100m behind (south)
    wingman_lon = ref_lon + 0.01 + 0.0006  # 50m to the right (east)
    wingman.set_position_lla(lat=wingman_lat, lon=wingman_lon, alt=ref_alt + 1500)
    wingman.set_velocity_ned(v_north=60.0, v_east=60.0, v_down=0.0)  # Same velocity
    wingman.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)  # Same acceleration
    wingman.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=45.0)  # Same heading
    
    # Calculate relative state from leader's perspective
    relative_state = leader.get_relative_state(wingman)
    
    print("\nFormation Flying Scenario:")
    print("-" * 30)
    print("Leader aircraft heading northeast at 85 m/s")
    print("Wingman aircraft in formation")
    print()
    print("Relative state from Leader's perspective:")
    print(f"  Position: {relative_state['position']['x']:.1f}m ahead/behind, "
          f"{relative_state['position']['y']:.1f}m left/right, "
          f"{relative_state['position']['z']:.1f}m up/down")
    print(f"  Velocity: {relative_state['velocity']['x']:.1f}m/s, "
          f"{relative_state['velocity']['y']:.1f}m/s, "
          f"{relative_state['velocity']['z']:.1f}m/s")
    print(f"  Acceleration: {relative_state['acceleration']['x']:.1f}m/s², "
          f"{relative_state['acceleration']['y']:.1f}m/s², "
          f"{relative_state['acceleration']['z']:.1f}m/s²")
    
    # Interpretation
    pos = relative_state['position']
    if pos['x'] > 0:
        front_back = f"{pos['x']:.1f}m ahead"
    else:
        front_back = f"{abs(pos['x']):.1f}m behind"
    
    if pos['y'] > 0:
        left_right = f"{pos['y']:.1f}m to the right"
    else:
        left_right = f"{abs(pos['y']):.1f}m to the left"
    
    print(f"\nInterpretation: Wingman is {front_back} and {left_right} of leader")
    
    vel_mag = np.sqrt(sum(v**2 for v in relative_state['velocity'].values()))
    if vel_mag < 0.1:
        print("Relative velocity: ~0 (maintaining formation)")
    else:
        print(f"Relative velocity magnitude: {vel_mag:.1f} m/s")


def demonstrate_real_world_scenario():
    """Demonstrate a real-world air traffic control scenario"""
    print("\n\n" + "=" * 60)
    print("REAL-WORLD SCENARIO: AIR TRAFFIC CONTROL")
    print("=" * 60)
    
    # Create platforms for ATC scenario
    aircraft1 = Platform()  # Commercial airliner
    aircraft2 = Platform()  # Private jet
    
    # Reference point: San Francisco International Airport
    sfo_lat, sfo_lon, sfo_alt = 37.6213, -122.3790, 0.0
    aircraft1.set_reference_lla(sfo_lat, sfo_lon, sfo_alt)
    aircraft2.set_reference_lla(sfo_lat, sfo_lon, sfo_alt)
    
    # Aircraft 1: Commercial airliner on approach, 10 miles out
    aircraft1.set_position_lla(
        lat=sfo_lat + 0.15,   # ~10 miles north
        lon=sfo_lon,          # Same longitude
        alt=sfo_alt + 914     # 3000 ft
    )
    aircraft1.set_velocity_ned(v_north=-50.0, v_east=0.0, v_down=-4.0)  # Approaching, descending
    aircraft1.set_acceleration_ned(a_north=-2.0, a_east=0.0, a_down=-0.5)  # Decelerating
    aircraft1.set_orientation_euler_ned(roll=0.0, pitch=-2.0, yaw=180.0)  # Heading south, nose down
    
    # Aircraft 2: Private jet crossing path, 5 miles east
    aircraft2.set_position_lla(
        lat=sfo_lat + 0.05,   # ~3 miles north
        lon=sfo_lon + 0.08,   # ~5 miles east
        alt=sfo_alt + 1524    # 5000 ft
    )
    aircraft2.set_velocity_ned(v_north=0.0, v_east=-80.0, v_down=0.0)  # Flying west
    aircraft2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
    aircraft2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=270.0)  # Heading west
    
    # Display scenario
    print("\nAir Traffic Control Scenario:")
    print("-" * 35)
    
    # Aircraft 1 details
    pos1_ned = aircraft1.get_position_ned()
    vel1_ned = aircraft1.get_velocity_ned()
    alt1_ft = (sfo_alt + 914) * 3.28084
    speed1_kts = np.linalg.norm(vel1_ned) * 1.94384
    
    print(f"Aircraft 1 (Commercial Airliner):")
    print(f"  Position: {pos1_ned[0]/1609:.1f} miles N, {pos1_ned[1]/1609:.1f} miles E, {alt1_ft:.0f} ft")
    print(f"  Speed: {speed1_kts:.0f} knots, heading South")
    print(f"  Status: On approach, descending at {abs(vel1_ned[2]):.1f} m/s")
    
    # Aircraft 2 details  
    pos2_ned = aircraft2.get_position_ned()
    vel2_ned = aircraft2.get_velocity_ned()
    alt2_ft = (sfo_alt + 1524) * 3.28084
    speed2_kts = np.linalg.norm(vel2_ned) * 1.94384
    
    print(f"\nAircraft 2 (Private Jet):")
    print(f"  Position: {pos2_ned[0]/1609:.1f} miles N, {pos2_ned[1]/1609:.1f} miles E, {alt2_ft:.0f} ft")
    print(f"  Speed: {speed2_kts:.0f} knots, heading West")
    print(f"  Status: Level flight, crossing approach path")
    
    # Calculate relative state for collision avoidance
    relative_state = aircraft1.get_relative_state(aircraft2)
    rel_pos = relative_state['position']
    rel_vel = relative_state['velocity']
    
    print(f"\nRelative State (from Aircraft 1's perspective):")
    print(f"  Aircraft 2 is {rel_pos['x']/1609:.1f} miles ahead/behind, "
          f"{rel_pos['y']/1609:.1f} miles left/right")
    print(f"  Altitude separation: {abs(rel_pos['z']):.0f}m ({abs(rel_pos['z'])*3.28084:.0f} ft)")
    print(f"  Relative velocity: {np.linalg.norm([rel_vel['x'], rel_vel['y']]):.0f} m/s")
    
    # Safety assessment
    horizontal_sep = np.sqrt(rel_pos['x']**2 + rel_pos['y']**2) / 1609  # miles
    vertical_sep = abs(rel_pos['z']) * 3.28084  # feet
    
    print(f"\nSafety Assessment:")
    print(f"  Horizontal separation: {horizontal_sep:.1f} miles")
    print(f"  Vertical separation: {vertical_sep:.0f} feet")
    
    if horizontal_sep > 3.0 and vertical_sep > 1000:
        print("  Status: SAFE - Adequate separation")
    elif horizontal_sep > 1.0 and vertical_sep > 500:
        print("  Status: CAUTION - Monitor closely")
    else:
        print("  Status: WARNING - Potential conflict")


def main():
    """Run all VAPOr demonstration examples"""
    print("VAPOr - Velocity, Acceleration, Position, Orientation")
    print("Demonstration Examples")
    print()
    
    # Run all demonstrations
    demonstrate_individual_trackers()
    demonstrate_platform_integration()
    demonstrate_relative_state_calculation()
    demonstrate_real_world_scenario()
    
    print("\n\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("VAPOr provides comprehensive tracking and relative state")
    print("calculations for aerospace, maritime, and ground applications.")


if __name__ == "__main__":
    main()