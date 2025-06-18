"""Position tracking in various coordinate systems."""

import math
from typing import Tuple
from pyproj import Transformer, CRS


class PositionTracker:
    """Tracks position in various coordinate systems: WCS (ECEF), LLA, and NED"""
    
    def __init__(self):
        self._wcs_position = None  # ECEF coordinates (x, y, z) in meters
        self._lla_position = None  # (latitude, longitude, altitude) in degrees, degrees, meters
        self._reference_lla = None  # Reference point for NED coordinate system
        
        # Initialize coordinate transformers
        self._ecef_crs = CRS.from_epsg(4978)  # ECEF (Earth-Centered, Earth-Fixed)
        self._wgs84_crs = CRS.from_epsg(4326)  # WGS84 (lat/lon)
        self._ecef_to_wgs84 = Transformer.from_crs(self._ecef_crs, self._wgs84_crs, always_xy=True)
        self._wgs84_to_ecef = Transformer.from_crs(self._wgs84_crs, self._ecef_crs, always_xy=True)
    
    def set_reference_lla(self, lat: float, lon: float, alt: float) -> None:
        """Set the reference point for NED coordinate system"""
        self._reference_lla = (lat, lon, alt)
    
    def set_location_wcs(self, x: float, y: float, z: float) -> None:
        """Set position in World Coordinate System (ECEF)"""
        self._wcs_position = (x, y, z)
        # Convert to LLA for internal consistency
        lon, lat, alt = self._ecef_to_wgs84.transform(x, y, z)
        self._lla_position = (lat, lon, alt)
    
    def set_location_lla(self, lat: float, lon: float, alt: float) -> None:
        """Set position in Latitude/Longitude/Altitude"""
        self._lla_position = (lat, lon, alt)
        # Convert to WCS for internal consistency
        x, y, z = self._wgs84_to_ecef.transform(lon, lat, alt)
        self._wcs_position = (x, y, z)
    
    def set_location_ned(self, north: float, east: float, down: float) -> None:
        """Set position using North/East/Down coordinates relative to reference point"""
        if self._reference_lla is None:
            raise ValueError("Reference point not set for NED coordinates")
        
        ref_lat, ref_lon, ref_alt = self._reference_lla
        
        # Convert reference point to ECEF
        ref_x, ref_y, ref_z = self._wgs84_to_ecef.transform(ref_lon, ref_lat, ref_alt)
        
        # Convert NED to ECEF using inverse rotation matrix
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Rotation matrix from NED to ECEF (transpose of ECEF to NED)
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        dx = -sin_lat * cos_lon * north - sin_lon * east - cos_lat * cos_lon * down
        dy = -sin_lat * sin_lon * north + cos_lon * east - cos_lat * sin_lon * down
        dz = cos_lat * north - sin_lat * down
        
        # Calculate final ECEF position
        x = ref_x + dx
        y = ref_y + dy
        z = ref_z + dz
        
        self._wcs_position = (x, y, z)
        # Convert to LLA for internal consistency
        lon, lat, alt = self._ecef_to_wgs84.transform(x, y, z)
        self._lla_position = (lat, lon, alt)
    
    def get_location_wcs(self) -> Tuple[float, float, float]:
        """Get position in World Coordinate System (ECEF)"""
        if self._wcs_position is None:
            raise ValueError("Position not set")
        return self._wcs_position
    
    def get_location_lla(self) -> Tuple[float, float, float]:
        """Get position in Latitude/Longitude/Altitude"""
        if self._lla_position is None:
            raise ValueError("Position not set")
        return self._lla_position
    
    def get_location_ned(self) -> Tuple[float, float, float]:
        """Get position in North/East/Down relative to reference point"""
        if self._lla_position is None:
            raise ValueError("Position not set")
        if self._reference_lla is None:
            raise ValueError("Reference point not set for NED coordinates")
        
        lat, lon, alt = self._lla_position
        ref_lat, ref_lon, ref_alt = self._reference_lla
        
        # Convert both points to ECEF
        x, y, z = self._wgs84_to_ecef.transform(lon, lat, alt)
        ref_x, ref_y, ref_z = self._wgs84_to_ecef.transform(ref_lon, ref_lat, ref_alt)
        
        # Calculate ECEF differences
        dx = x - ref_x
        dy = y - ref_y
        dz = z - ref_z
        
        # Convert to NED using rotation matrix
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)
        
        # Rotation matrix from ECEF to NED
        sin_lat = math.sin(ref_lat_rad)
        cos_lat = math.cos(ref_lat_rad)
        sin_lon = math.sin(ref_lon_rad)
        cos_lon = math.cos(ref_lon_rad)
        
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        east = -sin_lon * dx + cos_lon * dy
        down = -cos_lat * cos_lon * dx - cos_lat * sin_lon * dy - sin_lat * dz
        
        return (north, east, down)