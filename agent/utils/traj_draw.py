from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw
import numpy as np

from agent.utils.image_io import ndarray_to_pil_rgb

def draw_trajectory_on_image(img_any: Any, trajectory: np.ndarray, color: str = "green", radius: int = 3) -> Image.Image:
    """Draw trajectory points on image and return the modified PIL Image."""
    # Convert input to PIL Image
    if isinstance(img_any, np.ndarray):
        img = ndarray_to_pil_rgb(img_any)
    elif isinstance(img_any, Image.Image):
        img = img_any.copy()
    else:
        # Try to convert other types
        img = ndarray_to_pil_rgb(np.array(img_any))
    
    draw = ImageDraw.Draw(img)
    
    # Draw trajectory points
    for i, point in enumerate(trajectory):
        x, y = point[0], point[1]
        # Draw circle for each point
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color)
        
        # Draw line to next point
        if i < len(trajectory) - 1:
            next_x, next_y = trajectory[i+1][0], trajectory[i+1][1]
            draw.line([x, y, next_x, next_y], fill=color, width=2)
    
    return img
