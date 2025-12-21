import base64
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as _Image

from .image_io import ndarray_to_pil_rgb, image_to_data_url


try:
    import debugpy
    IS_DEBUG_ENV = debugpy.is_client_connected()
except Exception:
    IS_DEBUG_ENV = False


def ensure_cache_dirs():
    if IS_DEBUG_ENV:
        os.makedirs(".cache/debug_images", exist_ok=True)
        os.makedirs(".cache/debug_texts", exist_ok=True)

def save_debug_image(name: str, img_any: Any):
    if not IS_DEBUG_ENV:
        return
    
    img_path = f".cache/debug_images/{name}.jpg"
    
    try:
        if isinstance(img_any, np.ndarray):
            ndarray_to_pil_rgb(img_any).save(img_path)
        elif isinstance(img_any, _Image.Image):
            img_any.convert("RGB").save(img_path)
        elif isinstance(img_any, (str, Path)):
            _Image.open(img_any).convert("RGB").save(img_path)
        else:
            data_url = image_to_data_url(img_any)
            with open(img_path, "wb") as _f:
                _f.write(base64.b64decode(data_url.split(",", 1)[1]))
    except Exception as e:
        print(f"[Debug] save image failed for {name}: {e}")

def save_debug_text(name: str, prompt_text: str):
    if not IS_DEBUG_ENV:
        return
    
    prompt_path = f".cache/debug_texts/{name}"
    
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)
    except Exception as e:
        print(f"[Debug] save prompt failed for {name}: {e}")
        
