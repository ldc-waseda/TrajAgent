import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

def ndarray_to_pil_rgb(arr: np.ndarray):
    a = arr
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        mode = "L"
    else:
        if a.shape[2] > 3:
            a = a[:, :, :3]
        mode = "RGB"

    if a.dtype.kind == "f":
        a = a.copy()
        a = np.clip(a, 0, 1 if a.max() <= 1.0 else 255)
        if a.max() <= 1.0:
            a = (a * 255.0).round()
        a = a.astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)

    img = Image.fromarray(a, mode=mode)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def image_to_data_url(img_any: Any) -> str:
    mime = "image/jpeg"

    if isinstance(img_any, (str, Path)):
        p = Path(img_any)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        mime_guess, _ = mimetypes.guess_type(str(p))
        if mime_guess:
            mime = mime_guess
        raw = p.read_bytes()
    elif isinstance(img_any, np.ndarray):
        img = ndarray_to_pil_rgb(img_any)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()
    elif isinstance(img_any, Image.Image):
        buf = BytesIO()
        img_any.convert("RGB").save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()
    else:
        raise TypeError(f"Unsupported image type: {type(img_any)}")

    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"
