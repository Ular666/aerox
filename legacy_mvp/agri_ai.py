
import numpy as np
from PIL import Image


def analyze_agriculture(image, field_area_ha=None):
    """
    Estimate visible green vegetation coverage.

    This is an image-processing baseline, not a
    trained crop recognition or yield prediction model.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)

    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0

    # Excess Green vegetation index
    exg = 2 * g - r - b

    # Basic green vegetation mask
    mask = (
        (exg > 0.08)
        & (g > r * 1.08)
        & (g > b * 1.05)
        & (g > 0.18)
    )

    coverage = float(mask.mean() * 100)

    # Green overlay on detected vegetation
    overlay = rgb.copy()
    green = np.array([35, 220, 90], dtype=np.float32)
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * green
    )

    result_image = Image.fromarray(
    np.clip(overlay, 0, 255).astype(np.uint8)
)
    result = {
        "coverage_percent": coverage,
        "image": result_image,
        "vegetation_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
    }

    if field_area_ha is not None:
        result["vegetation_area_ha"] = (
            field_area_ha * coverage / 100
        )

    return result
