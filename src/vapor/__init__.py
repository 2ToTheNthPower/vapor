"""
VAPOr - Velocity, Acceleration, Position, Orientation tracking in various coordinate systems.

This package provides tools for tracking and converting between different coordinate systems:
- WCS (World Coordinate System / ECEF - Earth-Centered, Earth-Fixed)
- LLA (Latitude, Longitude, Altitude)
- NED (North, East, Down)
"""

from .trackers import PositionTracker, VelocityTracker, AccelerationTracker, OrientationTracker
from .core import Platform

__version__ = "0.1.0"
__all__ = ["PositionTracker", "VelocityTracker", "AccelerationTracker", "OrientationTracker", "Platform"]