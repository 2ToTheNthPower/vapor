"""Utility functions for the VAPOr library."""

import math
from typing import Union


def validate_coordinate_value(value: float, name: str) -> None:
    """Validate a coordinate value for NaN and infinite values.
    
    Args:
        value: The value to validate
        name: Name of the coordinate for error messages
        
    Raises:
        ValueError: If value is NaN or infinite
    """
    if math.isnan(value):
        raise ValueError(f"{name} cannot be NaN")
    if math.isinf(value):
        raise ValueError(f"{name} cannot be infinite")


def validate_latitude(lat: float) -> None:
    """Validate latitude value.
    
    Args:
        lat: Latitude in degrees
        
    Raises:
        ValueError: If latitude is invalid
    """
    validate_coordinate_value(lat, "Latitude")
    if lat < -90.0 or lat > 90.0:
        raise ValueError(f"Latitude must be between -90° and +90°, got {lat}°")


def validate_longitude(lon: float) -> None:
    """Validate longitude value.
    
    Args:
        lon: Longitude in degrees
        
    Raises:
        ValueError: If longitude is invalid
    """
    validate_coordinate_value(lon, "Longitude")
    if lon < -180.0 or lon > 180.0:
        raise ValueError(f"Longitude must be between -180° and +180°, got {lon}°")


def validate_altitude(alt: float) -> None:
    """Validate altitude value.
    
    Args:
        alt: Altitude in meters
        
    Raises:
        ValueError: If altitude is invalid
    """
    validate_coordinate_value(alt, "Altitude")
    # Allow very large altitude ranges for space applications
    if alt < -15000.0 or alt > 1e9:  # Below Dead Sea to space
        raise ValueError(f"Altitude must be reasonable (between -15km and 1Gm), got {alt}m")


def validate_velocity_component(vel: float, name: str) -> None:
    """Validate velocity component.
    
    Args:
        vel: Velocity component in m/s
        name: Name of the component for error messages
        
    Raises:
        ValueError: If velocity is invalid
    """
    validate_coordinate_value(vel, f"Velocity {name}")
    # Allow very high velocities for space applications  
    if abs(vel) > 1e6:  # 1000 km/s should cover most applications
        raise ValueError(f"Velocity {name} magnitude is unreasonably large: {vel} m/s")


def validate_acceleration_component(accel: float, name: str) -> None:
    """Validate acceleration component.
    
    Args:
        accel: Acceleration component in m/s²
        name: Name of the component for error messages
        
    Raises:
        ValueError: If acceleration is invalid
    """
    validate_coordinate_value(accel, f"Acceleration {name}")
    # Allow very high accelerations for space/extreme applications
    if abs(accel) > 1e6:  # 1000 km/s² should cover most applications
        raise ValueError(f"Acceleration {name} magnitude is unreasonably large: {accel} m/s²")


def validate_angle(angle: float, name: str) -> None:
    """Validate angle value.
    
    Args:
        angle: Angle in degrees
        name: Name of the angle for error messages
        
    Raises:
        ValueError: If angle is invalid
    """
    validate_coordinate_value(angle, name)
    # Don't restrict angle ranges as they can be normalized