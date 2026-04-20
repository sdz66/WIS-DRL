"""Map configuration and management module."""

from typing import Any

from maps.map_a_afm import AFMOpenTrackEnv
from maps.map_b_apt import APTAlignmentEnv
from maps.map_c_azr import AZRReorientationEnv
from maps.tri_mode_composite_map import TriModeCompositeEnv


class MapManager:
    """
    Map manager - used for selecting maps and configuring parameters
    """

    _MAP_ALIASES = {}
    
    def __init__(self):
        """
        Initialize map manager
        """
        pass
    
    def create_map(self, map_type: str, **kwargs) -> Any:
        """
        Create map environment
        
        Args:
            map_type: Map type ('map_a', 'map_b', 'map_c', 'tri_mode_composite')
            **kwargs: Other parameters
                - start_point: Start point coordinates (x, y)
                - start_heading: Start heading (rad)
                - end_point: End point coordinates (x, y)
                - end_heading: End heading (rad)
        
        Returns:
            Map environment object
        """
        map_type = self._normalize_map_type(map_type)

        if map_type == 'map_a':
            env = AFMOpenTrackEnv(**kwargs)
        elif map_type == 'map_b':
            env = APTAlignmentEnv(**kwargs)
        elif map_type == 'map_c':
            env = AZRReorientationEnv(**kwargs)
        elif map_type == 'tri_mode_composite':
            env = TriModeCompositeEnv(**kwargs)
        else:
            raise ValueError(f"Unknown map type: {map_type}")
        
        # Configure start point
        if 'start_point' in kwargs:
            env.set_start_point(kwargs['start_point'])
        
        # Configure start heading
        if 'start_heading' in kwargs:
            env.set_start_heading(kwargs['start_heading'])
        
        # Configure end point
        if 'end_point' in kwargs:
            env.set_end_point(kwargs['end_point'])
        
        # Configure end heading
        if 'end_heading' in kwargs:
            env.set_end_heading(kwargs['end_heading'])
        
        return env

    @classmethod
    def _normalize_map_type(cls, map_type: str) -> str:
        """Map user-facing aliases to the internal canonical map name."""
        return cls._MAP_ALIASES.get(map_type, map_type)
    

    
    @staticmethod
    def get_available_maps() -> list:
        """
        Get available map types
        
        Returns:
            list: Map type list
        """
        return [
            'map_a',
            'map_b',
            'map_c',
            'tri_mode_composite',
        ]
    
    @staticmethod
    def validate_map_type(map_type: str) -> bool:
        """
        Validate if map type is valid
        
        Args:
            map_type: Map type
        
        Returns:
            bool: Whether valid
        """
        return MapManager._normalize_map_type(map_type) in {
            'map_a', 'map_b', 'map_c', 'tri_mode_composite'
        }
