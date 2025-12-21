"""
Trajectory generator for gen_full_traj_v1 strategy.
Generates 20 complete trajectory points (including observations and future).
"""

from typing import Any, Dict, List

import numpy as np

from .base import BaseTrajectoryGenerator
from .base import TrajectoryAlternatives

class CaseGenerator(BaseTrajectoryGenerator):
    """Generator for creating 20 complete trajectory points."""
    
    def __init__(self, model: str, temperature: float = 0.0, max_completion_tokens: int = 1024, 
                 system_version: str = "v1", user_version: str = "v1"):
        super().__init__(
            template_name="gen_full_traj",
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            system_version=system_version,
            user_version=user_version
        )
    
    def get_response_format(self) -> type:
        """Return the Pydantic model class for response parsing."""
        return TrajectoryAlternatives
    
    def prepare_template_data(self, traj: np.ndarray, anno_text: str) -> Dict[str, Any]:
        print(traj)
        """Prepare template variables for gen_full_traj_v1 strategy."""
        traj_int = self._convert_trajectory_to_int(traj)
        # Extract start point (first point of trajectory)
        start_point = traj_int[0]
        return {
            'traj': traj_int,  # Full original trajectory
            'anno_text': anno_text,
            'start_point': start_point,  # Starting point that model should continue from
        }
    
    def validate_points(self, points: List[List[float]], alt_idx: int, idx: int, traj_id: Any) -> bool:
        """Validate that we have exactly 19 points in correct format."""
        return self._validate_point_format(points, alt_idx, idx, traj_id)
    
    def create_full_trajectory(self, traj: np.ndarray, points: List[List[float]]) -> np.ndarray:
        """Create full trajectory by prepending start point to 19 predicted points."""
        # Get the start point (first point of original trajectory)
        start_point = traj[0:1]  # Shape: (1, 2)
        
        # Convert predicted points to array
        pred_array = np.array(points, dtype=np.float32)  # Shape: (19, 2)
        
        # Prepend start point to predictions
        full_traj = np.vstack([start_point, pred_array])  # Shape: (20, 2)
        
        # Ensure we have exactly 20 points (truncate if more)
        if len(full_traj) > 20:
            full_traj = full_traj[:20]
        
        return full_traj
    
    def get_expected_prediction_count(self) -> int:
        """Return the expected number of prediction points (19, excluding start point)."""
        return 19
