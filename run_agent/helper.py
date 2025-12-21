import os
import asyncio
import argparse
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from openai import AsyncOpenAI
from pathlib import Path
from agent.utils.openai_wrapper import create_wrapped_client

def image_to_base64_url(image_path: Path) -> str:
    """Reads an image and converts it to a base64 data URL."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image"):
        raise ValueError(f"Unsupported file type: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"