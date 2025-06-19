"""Velocity tracking in various coordinate systems."""

import math
from typing import Tuple
from pyproj import CRS
from ..utils import validate_latitude, validate_longitude, validate_altitude, validate_velocity_component, validate_coordinate_value


class VelocityTracker:
    """Tracks velocity in various coordinate systems"""
    
    def __init__(self):
        self._wcs_velocity = None  # ECEF velocity (vx, vy, vz) in m/s
        self._lla_rates = None     # (lat_rate, lon_rate, alt_rate) in deg/s, deg/s, m/s
        self._ned_velocity = None  # (v_north, v_east, v_down) in m/s
        self._reference_lla = None
        
        # Initialize coordinate transformers
        self._ecef_crs = CRS.from_epsg(4978)
        self._wgs84_crs = CRS.from_epsg(4326)
    
    def set_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        """Set the reference point for NED coordinate system"""
        validate_latitude(lat)
        validate_longitude(lon)
        validate_altitude(alt)
        self._reference_lla = (lat, lon, alt)
    
    def set_velocity_wcs(self, vx: float, vy: float, vz: float) -> None:
        """Set velocity in World Coordinate System (ECEF)"""
        validate_velocity_component(vx, "X")
        validate_velocity_component(vy, "Y")
        validate_velocity_component(vz, "Z")
        self._wcs_velocity = (vx, vy, vz)
    
    def set_velocity_lla_rates(self, lat_rate: float, lon_rate: float, alt_rate: float) -> None:
        """Set velocity as LLA rates (degrees/s, degrees/s, m/s)"""
        validate_coordinate_value(lat_rate, "Latitude rate")
        validate_coordinate_value(lon_rate, "Longitude rate")
        validate_velocity_component(alt_rate, "altitude rate")
        
        # Sanity check for angular rates (shouldn't exceed reasonable rotation speeds)
        if abs(lat_rate) > 360.0:  # More than 1 revolution per second
            raise ValueError(f"Latitude rate is unreasonably high: {lat_rate} deg/s")
        if abs(lon_rate) > 360.0:  # More than 1 revolution per second
            raise ValueError(f"Longitude rate is unreasonably high: {lon_rate} deg/s")
        
        self._lla_rates = (lat_rate, lon_rate, alt_rate)
    
    def set_velocity_ned(self, v_north: float, v_east: float, v_down: float) -> None:
        """Set velocity in North/East/Down"""
        validate_velocity_component(v_north, "north")
        validate_velocity_component(v_east, "east")
        validate_velocity_component(v_down, "down")
        self._ned_velocity = (v_north, v_east, v_down)
    
    def get_velocity_wcs(self) -> Tuple[float, float, float]:
        """Get velocity in World Coordinate System (ECEF)"""
        if self._wcs_velocity is not None:
            return self._wcs_velocity
        elif self._ned_velocity is not None and self._reference_lla is not None:
            return self._ned_to_wcs_velocity()
        else:
            raise ValueError("Velocity not set or insufficient data for conversion")
    
    def get_velocity_ned(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Get velocity in North/East/Down
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if self._ned_velocity is not None:
            return self._ned_velocity
        elif self._wcs_velocity is not None:
            return self._wcs_to_ned_velocity(reference_lla)
        elif self._lla_rates is not None:
            return self._lla_rates_to_ned_velocity(reference_lla)
        else:
            raise ValueError("Velocity not set or insufficient data for conversion")
    
    def _wcs_to_ned_velocity(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Convert WCS velocity to NED velocity
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if reference_lla is not None:
            ref_lat, ref_lon, ref_alt = reference_lla
        elif self._reference_lla is not None:
            ref_lat, ref_lon, ref_alt = self._reference_lla
        else:
            raise ValueError("Reference point not set")
        
        vx, vy, vz = self._wcs_velocity
        
        # Convert using rotation matrix
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        v_north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
        v_east = -sin_lon * vx + cos_lon * vy
        v_down = -cos_lat * cos_lon * vx - cos_lat * sin_lon * vy - sin_lat * vz
        
        return (v_north, v_east, v_down)
    
    def _ned_to_wcs_velocity(self) -> Tuple[float, float, float]:
        """Convert NED velocity to WCS velocity"""
        if self._reference_lla is None:
            raise ValueError("Reference point not set")
        
        v_north, v_east, v_down = self._ned_velocity
        ref_lat, ref_lon, _ = self._reference_lla
        
        # Convert using inverse rotation matrix
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        vx = -sin_lat * cos_lon * v_north - sin_lon * v_east - cos_lat * cos_lon * v_down
        vy = -sin_lat * sin_lon * v_north + cos_lon * v_east - cos_lat * sin_lon * v_down
        vz = cos_lat * v_north - sin_lat * v_down
        
        return (vx, vy, vz)
    
    def _lla_rates_to_ned_velocity(self, reference_lla: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Convert LLA rates to NED velocity
        
        Args:
            reference_lla: Optional reference point (lat, lon, alt). If not provided, uses stored reference.
        """
        if reference_lla is not None:
            ref_lat, ref_lon, ref_alt = reference_lla
        elif self._reference_lla is not None:
            ref_lat, ref_lon, ref_alt = self._reference_lla
        else:
            raise ValueError("Reference point not set")
        
        lat_rate, lon_rate, alt_rate = self._lla_rates
        
        # Convert angular rates to linear velocities
        # Earth radius approximation
        R_earth = 6378137.0  # WGS84 equatorial radius in meters
        
        # Convert degrees to radians for calculation
        lat_rate_rad = math.radians(lat_rate)
        lon_rate_rad = math.radians(lon_rate)
        ref_lat_rad = math.radians(ref_lat)
        
        # Calculate NED velocities
        v_north = lat_rate_rad * R_earth
        v_east = lon_rate_rad * R_earth * math.cos(ref_lat_rad)
        v_down = -alt_rate  # Negative because down is positive in NED
        
        return (v_north, v_east, v_down)