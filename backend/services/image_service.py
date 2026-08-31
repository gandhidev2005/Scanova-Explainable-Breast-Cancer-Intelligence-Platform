"""Medical image loading (JPEG/PNG/DICOM)."""

import io

import numpy as np
from PIL import Image


def load_image_from_upload(filename: str, contents: bytes, mimetype: str = "") -> Image.Image:
    fname = (filename or "").strip()
    low = fname.lower()
    ctype = (mimetype or "").lower()

    if low.endswith(".avif") or ctype == "image/avif":
        raise ValueError(
            "AVIF images are not supported. Please upload JPG/PNG, or DICOM (.dcm)."
        )

    if low.endswith(".dcm"):
        import cv2
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(contents))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)
        return Image.fromarray(arr).convert("RGB")

    return Image.open(io.BytesIO(contents)).convert("RGB")
