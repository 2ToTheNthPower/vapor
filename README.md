# VAPOr

A Python library for tracking Velocity, Acceleration, Position, and Orientation across multiple coordinate systems.

## Features

### Coordinate Systems Supported

- **WCS (World Coordinate System)**: Earth-Centered, Earth-Fixed (ECEF) coordinates in meters
- **LLA**: Latitude/Longitude/Altitude using WGS84 datum
- **NED**: North/East/Down relative to a local reference point

### Core Capabilities

- **Position Tracking**: Convert between WCS, LLA, and NED coordinate systems
- **Velocity Tracking**: Handle velocity vectors in multiple coordinate frames
- **Acceleration Tracking**: Track acceleration in various coordinate systems  
- **Orientation Tracking**: Support for Euler angles, quaternions, and rotation matrices
- **Relative State Calculation**: Compute relative position, velocity, acceleration, and orientation between platforms

### Transformation Features

- Automatic coordinate system conversions using pyproj
- Bidirectional transformations with precision validation
- NED frame calculations relative to user-defined reference points
- Body-fixed coordinate system transformations (nose/roof/wing axes)

## Installation

Since this package is not yet published to PyPI, install it directly from the source:

```bash
# Install with uv (recommended)
uv add https://github.com/2ToTheNthPower/vapor.git

# Or clone and install locally
git clone https://github.com/2ToTheNthPower/vapor.git
cd vapor

# Install with uv
uv pip install -e .

# Or install with pip
pip install -e .
```

## Basic Usage

### Individual Trackers

```python
from vapor import PositionTracker, VelocityTracker, AccelerationTracker, OrientationTracker

# Position tracking
pos = PositionTracker()
pos.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
pos.set_location_lla(lat=37.7750, lon=-122.4193, alt=105.0)

# Get position in different coordinate systems
wcs_pos = pos.get_location_wcs()  # ECEF coordinates
lla_pos = pos.get_location_lla()  # Lat/Lon/Alt
ned_pos = pos.get_location_ned()  # North/East/Down relative to reference

# Velocity tracking
vel = VelocityTracker()
vel.set_reference_lla(lat=37.7749, lon=-122.4194, alt=100.0)
vel.set_velocity_ned(v_north=10.0, v_east=5.0, v_down=-2.0)
wcs_vel = vel.get_velocity_wcs()

# Orientation tracking
orient = OrientationTracker()
orient.set_orientation_euler_ned(roll=10.0, pitch=5.0, yaw=90.0)
quat = orient.get_orientation_quaternion_ned()
matrix = orient.get_orientation_matrix_ned()
```

### Platform Integration

```python
from vapor import Platform

# Create two platforms
platform1 = Platform()
platform2 = Platform()

# Set reference point
ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 100.0
platform1.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
platform2.set_reference_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)

# Configure platform states
platform1.set_position_lla(lat=ref_lat, lon=ref_lon, alt=ref_alt)
platform1.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
platform1.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
platform1.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)

platform2.set_position_lla(lat=ref_lat + 0.001, lon=ref_lon, alt=ref_alt)
platform2.set_velocity_ned(v_north=20.0, v_east=0.0, v_down=0.0)
platform2.set_acceleration_ned(a_north=0.0, a_east=0.0, a_down=0.0)
platform2.set_orientation_euler_ned(roll=0.0, pitch=0.0, yaw=0.0)

# Calculate relative state from platform1's perspective
relative_state = platform1.get_relative_state(platform2)

print(f"Relative position: {relative_state['position']}")
print(f"Relative velocity: {relative_state['velocity']}")
print(f"Relative acceleration: {relative_state['acceleration']}")
print(f"Relative orientation: {relative_state['orientation']}")
```

## Coordinate System Details

### WCS (World Coordinate System)
- Earth-Centered, Earth-Fixed (ECEF) coordinates
- Origin at Earth's center of mass
- X-axis through equator at 0� longitude
- Y-axis through equator at 90� East longitude  
- Z-axis through North pole
- Units: meters

### LLA (Latitude/Longitude/Altitude)
- Latitude: degrees North (-90 to +90)
- Longitude: degrees East (-180 to +180)
- Altitude: meters above WGS84 ellipsoid
- Datum: WGS84

### NED (North/East/Down)
- Local coordinate system relative to reference point
- North: positive toward geographic North
- East: positive toward geographic East
- Down: positive toward Earth center
- Units: meters

### Body-Fixed Coordinates
- X-axis: through platform nose (forward)
- Y-axis: through platform roof (up)
- Z-axis: through platform right wing (right)
- Right-handed coordinate system

## Use Cases

### Aerospace Applications
- Aircraft position and attitude tracking
- Formation flying calculations
- Approach and landing systems
- Satellite tracking from ground stations

### Maritime Applications  
- Ship navigation and positioning
- Relative motion between vessels
- Harbor approach calculations

### Ground Vehicle Applications
- Convoy spacing and coordination
- Relative positioning for autonomous vehicles
- Surveying and mapping applications

### Robotics Applications
- Multi-robot coordination
- Relative state estimation
- Navigation system integration

## API Reference

### PositionTracker
- `set_location_wcs(x, y, z)`: Set position in ECEF coordinates
- `set_location_lla(lat, lon, alt)`: Set position in lat/lon/alt
- `get_location_wcs()`: Get ECEF coordinates
- `get_location_lla()`: Get lat/lon/alt coordinates  
- `get_location_ned()`: Get NED coordinates relative to reference
- `set_reference_lla(lat, lon, alt)`: Set NED reference point

### VelocityTracker
- `set_velocity_wcs(vx, vy, vz)`: Set velocity in ECEF frame
- `set_velocity_ned(v_north, v_east, v_down)`: Set velocity in NED frame
- `set_velocity_lla_rates(lat_rate, lon_rate, alt_rate)`: Set as LLA rates
- `get_velocity_wcs()`: Get ECEF velocity
- `get_velocity_ned()`: Get NED velocity

### AccelerationTracker  
- `set_acceleration_wcs(ax, ay, az)`: Set acceleration in ECEF frame
- `set_acceleration_ned(a_north, a_east, a_down)`: Set acceleration in NED frame
- `set_acceleration_lla_rates(lat_accel, lon_accel, alt_accel)`: Set as LLA rates
- `get_acceleration_wcs()`: Get ECEF acceleration
- `get_acceleration_ned()`: Get NED acceleration

### OrientationTracker
- `set_orientation_euler_ned(roll, pitch, yaw)`: Set using Euler angles (degrees)
- `set_orientation_quaternion_ned(w, x, y, z)`: Set using quaternion
- `set_orientation_matrix_ned(matrix)`: Set using 3x3 rotation matrix
- `get_orientation_euler_ned()`: Get Euler angles
- `get_orientation_quaternion_ned()`: Get quaternion
- `get_orientation_matrix_ned()`: Get rotation matrix

### Platform
- All methods from individual trackers
- `get_relative_state(other_platform)`: Calculate relative state in body-fixed coordinates

## Dependencies

- numpy: Numerical computations and matrix operations
- pyproj: Coordinate system transformations and projections
- pytest: Testing framework (development)

## Testing

The library includes comprehensive test coverage:

```bash
uv run pytest tests/ -v
```

Test categories:
- Basic coordinate system transformations
- Transformation reversibility validation  
- Platform relative state calculations
- Real-world scenario validation
- Edge case handling

## License

MIT License

## Contributing

Contributions welcome. Please ensure all tests pass and add tests for new functionality.