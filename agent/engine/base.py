"""
Base class for trajectory generation strategies.
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import cv2

from agent.utils.image_io import image_to_data_url
from agent.utils.debug import save_debug_image, save_debug_text
from agent.utils.prompt_template import render_template
from agent.utils.traj_draw import draw_trajectory_on_image


class Alternative(BaseModel):
    description: str = Field(description="The description of the trajectory.")
    points: List[List[int]] # = Field(description="The trajectory points, each point is a list of two floats, total number of points is 20.")

class TrajectoryAlternatives(BaseModel):
    alternatives: List[Alternative]

class BaseTrajectoryGenerator(ABC):
    """Base class for trajectory generation strategies."""
    
    def __init__(self, template_name: str, model: str, temperature: float = 0.2, max_completion_tokens: int = 1024, 
                 system_template_name: Optional[str] = None, user_template_name: Optional[str] = None,
                 system_version: Optional[str] = None, user_version: Optional[str] = None):
        """
        Initialize the trajectory generator.
        
        Args:
            template_name: Base template name (e.g., "gen_full_traj")
            model: Model name to use
            temperature: Temperature for generation
            max_completion_tokens: Maximum tokens for generation
            system_template_name: Explicit system template name (e.g., "gen_full_traj_system_v1")
            user_template_name: Explicit user template name (e.g., "gen_full_traj_user_v2") 
            system_version: System template version (e.g., "v1", "v2") - used if system_template_name not provided
            user_version: User template version (e.g., "v1", "v2") - used if user_template_name not provided
        """
        self.template_name = template_name
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.system_version = system_version
        self.user_version = user_version
        
        # Support flexible template naming with independent versioning
        if system_template_name is not None:
            self.system_template_name = system_template_name
        elif system_version is not None:
            self.system_template_name = f"{template_name}_system_{system_version}"
        else:
            self.system_template_name = f"{template_name}_system"
            
        if user_template_name is not None:
            self.user_template_name = user_template_name
        elif user_version is not None:
            self.user_template_name = f"{template_name}_user_{user_version}"
        else:
            self.user_template_name = f"{template_name}_user"
    
    @abstractmethod
    def get_response_format(self) -> type:
        """Return the Pydantic model class for response parsing."""
        pass
    
    @abstractmethod
    def prepare_template_data(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template variables for the specific generation strategy."""
        pass
    
    @abstractmethod
    def validate_points(self, points: List[List[float]], alt_idx: int, idx: int, traj_id: Any) -> bool:
        """Validate the predicted trajectory points."""
        pass
    
    @abstractmethod
    def create_full_trajectory(self, window_data: Dict[str, Any], points: List[List[float]]) -> np.ndarray:
        """Create the full trajectory array from original trajectory and predictions."""
        pass
    
    @abstractmethod
    def get_expected_prediction_count(self) -> int:
        """Return the expected number of prediction points."""
        pass
    
    def get_image_dimensions(self, img_any: Any) -> Tuple[int, int]:
        """Get image dimensions (width, height) from various image formats."""
        # numpy array (H, W, C) or (H, W)
        height, width = img_any.shape[:2]
        return width, height
        
    def add_image_dimensions(self, template_data: Dict[str, Any], img_any: Any) -> Dict[str, Any]:
        """Add image dimensions to template data."""
        width, height = self.get_image_dimensions(img_any)
        # template_data['image_width'] = width
        # template_data['image_height'] = height
        template_data['image_size'] = f"{width}x{height}"
        return template_data
    
    def load_scenario_mask(self, scenario: str, base_dir: Optional[str] = None, required: bool = True) -> Optional[np.ndarray]:
        """
        Load the scenario mask image.
        
        Args:
            scenario: Scenario name (e.g., "ETH", "HOTEL")
            base_dir: Base directory to search for mask files. If None, uses seg_shrink directory.
            required: If True, raise FileNotFoundError when mask is not found. Default is True.
            
        Returns:
            Mask image as numpy array
            
        Raises:
            FileNotFoundError: If mask file is not found and required=True
        """
        if base_dir is None:
            # Default to seg_shrink directory under project root
            # Using resolve().parents[3] for clarity: file -> generation -> pipelines -> visual_language -> project_root
            project_root = Path(__file__).resolve().parents[3]
            base_dir = project_root / "aux_data/seg_shrink"
        else:
            base_dir = Path(base_dir)
        
        # Try to find mask file with format: {scenario}_masked.jpg
        # First try with original scenario name, then with lowercase
        mask_path = base_dir / f"{scenario}_masked.jpg"
        
        if not mask_path.exists():
            # Try lowercase version
            scenario_lower = scenario.lower()
            mask_path = base_dir / f"{scenario_lower}_masked.jpg"
        
        if mask_path.exists():
            try:
                mask_img = cv2.imread(str(mask_path))
                if mask_img is not None:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                    print(f"[Info] Loaded scenario mask: {mask_path}")
                    return mask_img
                else:
                    error_msg = f"Failed to read mask image (cv2.imread returned None): {mask_path}"
                    if required:
                        raise FileNotFoundError(error_msg)
                    print(f"[Warning] {error_msg}")
                    return None
            except FileNotFoundError:
                raise
            except (IOError, cv2.error) as e:
                error_msg = f"Failed to load scenario mask {mask_path}: {e}"
                if required:
                    raise FileNotFoundError(error_msg) from e
                print(f"[Warning] {error_msg}")
                return None
        else:
            error_msg = (
                f"Scenario mask not found for '{scenario}'.\n"
                f"  Expected path: {base_dir / f'{scenario}_masked.jpg'}\n"
                f"  Please generate mask using SAM annotator tool:\n"
                f"    cd visual_language/sam_web_app && ./start_sam_annotator.sh\n"
                f"  Then save the mask to: {base_dir}/\n"
                f"  Optional: shrink the mask to save token usage. Refer to docs/DATA_PREPARATION.md for details."
            )
            if required:
                raise FileNotFoundError(error_msg)
            print(f"[Info] {error_msg}")
            return None
    
    async def run_generation(
        self,
        client: AsyncOpenAI,
        idx: int,
        traj_id: Any,
        window_data: Dict[str, Any], # Changed from traj: np.ndarray
        anno_text: str, # This is now mostly unused
        img_any: Any,
        filename: str,
        scenario: str,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        """Run trajectory generation for the given inputs."""
        
        # Prepare template data from the window's JSON object
        template_data = self.prepare_template_data(window_data)
        template_data = self.add_image_dimensions(template_data, img_any)
        
        # Render system and user prompts separately
        system_prompt = render_template(self.system_template_name, template_data)
        user_prompt = render_template(self.user_template_name, template_data)
        data_url = image_to_data_url(img_any)

        # Build user content with main image
        user_content = [
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "text", "text": user_prompt},
        ]
        
        # Load scenario mask (required by default - will raise if not found)
        scenario_mask = self.load_scenario_mask(scenario)
        mask_data_url = image_to_data_url(scenario_mask)
        user_content.insert(0, {"type": "image_url", "image_url": {"url": mask_data_url, "detail": "high"}})
        save_debug_image(f'{scenario}_{traj_id}_mask', scenario_mask)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # Make API call
        async with semaphore:
            try:
                resp = await client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                    response_format=self.get_response_format(),
                    reasoning_effort="minimal",
                    # Pass metadata for batch queue organization
                    scenario=scenario,
                    traj_id=str(traj_id),
                    idx=idx,
                    filename=filename,
                )
            except Exception as e:
                print(f"[Error] chat request failed for idx={idx}, id={traj_id}: {e}")
                return idx, None

        # Parse response
        parsed = self._parse_response(resp)
        
        # Save debug files
        save_debug_text(f'{scenario}_{traj_id}_system_prompt.md', system_prompt)
        save_debug_text(f'{scenario}_{traj_id}_user_prompt.md', user_prompt)
        save_debug_image(f'{scenario}_{traj_id}_img', img_any)
        save_debug_text(f'{scenario}_{traj_id}_traj.json', json.dumps(parsed, ensure_ascii=False, indent=2))

        # Validate and process trajectories
        valid_trajectories = self._process_alternatives(
            parsed, window_data, img_any, idx, traj_id, filename, scenario
        )

        if not valid_trajectories:
            print(f"[Validation] No valid trajectories generated for idx={idx}, id={traj_id}")
            return idx, None

        return idx, valid_trajectories
    
    def _parse_response(self, resp: Any) -> Optional[Dict[str, Any]]:
        """Parse the API response to extract the structured data."""
        parsed: Any = None
        try:
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                parsed = getattr(choice.message, "parsed", None)
        except Exception:
            pass

        if parsed is not None:
            try:
                parsed = parsed.model_dump()
            except Exception:
                try:
                    parsed = parsed.dict()
                except Exception:
                    parsed = None
        
        return parsed
    
    def _process_alternatives(
        self,
        parsed: Optional[Dict[str, Any]],
        window_data: Dict[str, Any], # Changed from traj: np.ndarray
        img_any: Any,
        idx: int,
        traj_id: Any,
        filename: str,
        scenario: str,
    ) -> List[Dict[str, Any]]:
        """Process and validate alternative trajectories from the parsed response."""
        if parsed is None or "alternatives" not in parsed:
            print(f"[Validation] No valid response for idx={idx}, id={traj_id}")
            return []

        valid_trajectories = []
        expected_count = self.get_expected_prediction_count()
        
        for alt_idx, alternative in enumerate(parsed["alternatives"]):
            if "points" not in alternative:
                print(f"[Validation] No points in alternative {alt_idx} for idx={idx}, id={traj_id}")
                continue
            
            points = alternative["points"]
            if len(points) < expected_count:
                print(f"[Validation] Expected {expected_count} points, got {len(points)} in alternative {alt_idx} for idx={idx}, id={traj_id} - discarding")
                continue
            elif len(points) > expected_count:
                print(f"[Validation] Expected {expected_count} points, got {len(points)} in alternative {alt_idx} for idx={idx}, id={traj_id} - truncating")
                points = points[:expected_count]
            
            # Validate points format
            if not self.validate_points(points, alt_idx, idx, traj_id):
                continue
            
            # Create full trajectory
            pred_array = np.array(points, dtype=np.float32)
            full_traj = self.create_full_trajectory(window_data, pred_array)
            
            # Draw trajectory on image for visualization
            img_with_traj = draw_trajectory_on_image(img_any, pred_array, color="green")
            
            # Save debug image with trajectory
            save_debug_image(f"{scenario}_{traj_id}_traj_{alt_idx}", img_with_traj)
            
            # Generate new unique traj_id
            new_traj_id = f"{traj_id}_gen_{alt_idx}"
            
            valid_trajectories.append({
                "traj_id": new_traj_id,
                "trajectory": full_traj,
                "traj_length": len(full_traj),
                "annotation": alternative.get("description", "Generated trajectory"),
                "filename": filename,
                "scenario": scenario,
                "img": img_with_traj
            })

        return valid_trajectories
    
    def _validate_point_format(self, points: List[List[int]], alt_idx: int, idx: int, traj_id: Any) -> bool:
        """Common validation for point format (list of [x, y] coordinates)."""
        for point_idx, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                print(f"[Validation] Invalid point format at index {point_idx} in alternative {alt_idx} for idx={idx}, id={traj_id}")
                return False
            try:
                int(point[0])
                int(point[1])
            except (ValueError, TypeError):
                print(f"[Validation] Non-numeric point at index {point_idx} in alternative {alt_idx} for idx={idx}, id={traj_id}")
                return False
        return True

    def _convert_trajectory_to_int(self, points: np.ndarray) -> List[List[int]]:
        """Convert trajectory points to integers."""
        return [[int(point[0]), int(point[1])] for point in points]
