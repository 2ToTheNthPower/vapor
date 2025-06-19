"""Acceleration tracking in various coordinate systems."""

import math
from typing import Tuple
from ..utils import validate_latitude, validate_longitude, validate_altitude, validate_acceleration_component, validate_coordinate_value


class AccelerationTracker:
    """Tracks acceleration in various coordinate systems"""
    
    def __init__(self):
        self._wcs_acceleration = None  # ECEF acceleration (ax, ay, az) in m/s²
        self._lla_rates = None         # (lat_accel, lon_accel, alt_accel) in deg/s², deg/s², m/s²
        self._ned_acceleration = None  # (a_north, a_east, a_down) in m/s²
        self._reference_lla = None
    
    def set_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        """Set the reference point for NED coordinate system"""
        validate_latitude(lat)
        validate_longitude(lon)
        validate_altitude(alt)
        self._reference_lla = (lat, lon, alt)
    
    def set_acceleration_wcs(self, ax: float, ay: float, az: float) -> None:
        """Set acceleration in World Coordinate System (ECEF)"""
        validate_acceleration_component(ax, "X")
        validate_acceleration_component(ay, "Y")
        validate_acceleration_component(az, "Z")
        self._wcs_acceleration = (ax, ay, az)
    
    def set_acceleration_lla_rates(self, lat_accel: float, lon_accel: float, alt_accel: float) -> None:
        """Set acceleration as LLA second derivatives (degrees/s², degrees/s², m/s²)"""
        validate_coordinate_value(lat_accel, "Latitude acceleration")
        validate_coordinate_value(lon_accel, "Longitude acceleration")
        validate_acceleration_component(alt_accel, "altitude acceleration")
        
        # Sanity check for angular accelerations
        if abs(lat_accel) > 3600.0:  # More than 1 deg/s² seems excessive for most applications
            raise ValueError(f"Latitude acceleration is unreasonably high: {lat_accel} deg/s²")
        if abs(lon_accel) > 3600.0:
            raise ValueError(f"Longitude acceleration is unreasonably high: {lon_accel} deg/s²")
        
        self._lla_rates = (lat_accel, lon_accel, alt_accel)
    
    def set_acceleration_ned(self, a_north: float, a_east: float, a_down: float) -> None:
        """Set acceleration in North/East/Down"""
        validate_acceleration_component(a_north, "north")
        validate_acceleration_component(a_east, "east")
        validate_acceleration_component(a_down, "down")
        self._ned_acceleration = (a_north, a_east, a_down)
    
    def get_acceleration_wcs(self) -> Tuple[float, float, float]:
        """Get acceleration in World Coordinate System (ECEF)"""
        if self._wcs_acceleration is not None:
            return self._wcs_acceleration
        elif self._ned_acceleration is not None and self._reference_lla is not None:
            return self._ned_to_wcs_acceleration()
        else:
            raise ValueError("Acceleration not set or insufficient data for conversion")
    
    def get_acceleration_ned(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Get acceleration in North/East/Down
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if self._ned_acceleration is not None:
            return self._ned_acceleration
        elif self._wcs_acceleration is not None:
            return self._wcs_to_ned_acceleration(reference_lla)
        elif self._lla_rates is not None:
            return self._lla_rates_to_ned_acceleration(reference_lla)
        else:
            raise ValueError("Acceleration not set or insufficient data for conversion")
    
    def _wcs_to_ned_acceleration(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Convert WCS acceleration to NED acceleration
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if reference_lla is not None:
            ref_lat, ref_lon, ref_alt = reference_lla
        elif self._reference_lla is not None:
            ref_lat, ref_lon, ref_alt = self._reference_lla
        else:
            raise ValueError("Reference point not set")
        
        ax, ay, az = self._wcs_acceleration
        
        # Convert using rotation matrix (same as velocity)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        a_north = -sin_lat * cos_lon * ax - sin_lat * sin_lon * ay + cos_lat * az
        a_east = -sin_lon * ax + cos_lon * ay
        a_down = -cos_lat * cos_lon * ax - cos_lat * sin_lon * ay - sin_lat * az
        
        return (a_north, a_east, a_down)
    
    def _ned_to_wcs_acceleration(self) -> Tuple[float, float, float]:
        """Convert NED acceleration to WCS acceleration"""
        if self._reference_lla is None:
            raise ValueError("Reference point not set")
        
        a_north, a_east, a_down = self._ned_acceleration
        ref_lat, ref_lon, _ = self._reference_lla
        
        # Convert using inverse rotation matrix
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        ax = -sin_lat * cos_lon * a_north - sin_lon * a_east - cos_lat * cos_lon * a_down
        ay = -sin_lat * sin_lon * a_north + cos_lon * a_east - cos_lat * sin_lon * a_down
        az = cos_lat * a_north - sin_lat * a_down
        
        return (ax, ay, az)
    
    def _lla_rates_to_ned_acceleration(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Convert LLA acceleration rates to NED acceleration
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if reference_lla is not None:
            ref_lat, ref_lon, ref_alt = reference_lla
        elif self._reference_lla is not None:
            ref_lat, ref_lon, ref_alt = self._reference_lla
        else:
            raise ValueError("Reference point not set")
        
        lat_accel, lon_accel, alt_accel = self._lla_rates
        
        # Convert angular accelerations to linear accelerations
        R_earth = 6378137.0  # WGS84 equatorial radius in meters
        
        # Convert degrees to radians for calculation
        lat_accel_rad = math.radians(lat_accel)
        lon_accel_rad = math.radians(lon_accel)
        ref_lat_rad = math.radians(ref_lat)
        
        # Calculate NED accelerations
        a_north = lat_accel_rad * R_earth
        a_east = lon_accel_rad * R_earth * math.cos(ref_lat_rad)
        a_down = -alt_accel  # Negative because down is positive in NED
        
        return (a_north, a_east, a_down)