import io
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from guard import csv_bytes

YIELD_STATUS = "Yield Prediction: Not Yet Validated"


def analyze(data, grid):
    source = Image.open(io.BytesIO(data))
    if source.width * source.height > 30_000_000:
        raise ValueError("Use an RGB image of no more than 30 megapixels.")
    image = ImageOps.exif_transpose(source).convert("RGB")
    rgb = np.asarray(image)
    r, g, b = np.moveaxis(rgb, 2, 0).astype(np.float32) / 255
    # Preserve the ExG baseline and color gates from the original AGRI module.
    mask = ((2 * g - r - b > 0.08) & (g > r * 1.08) & (g > b * 1.05) & (g > 0.18))
    if min(mask.shape) < grid:
        raise ValueError("The image is too small for the selected grid.")
    height, width = mask.shape
    overlay_pixels = rgb.copy()
    overlay_pixels[mask] = rgb[mask] * 0.45 + np.array([35, 220, 90]) * 0.55
    overlay = Image.fromarray(overlay_pixels)
    grid_map = Image.new("RGB", image.size)
    draw = ImageDraw.Draw(grid_map)
    regions = []
    for row in range(grid):
        y0, y1 = row * height // grid, (row + 1) * height // grid
        for col in range(grid):
            x0, x1 = col * width // grid, (col + 1) * width // grid
            cell = mask[y0:y1, x0:x1]
            coverage = float(cell.mean())
            label = f"{chr(65 + row)}{col + 1}"
            regions.append(dict(region=label, coverage_percent=round(coverage * 100, 3),
                                vegetation_pixels=int(cell.sum()), total_pixels=cell.size))
            box = (x0, y0, x1 - 1, y1 - 1)
            draw.rectangle(box, fill=(round(215 - 190 * coverage), round(123 + 69 * coverage), 56),
                           outline="white")
            draw.text((x0 + 5, y0 + 5), f"{label} {coverage:.0%}", fill="black",
                      stroke_width=1, stroke_fill="white")
    coverage = float(mask.mean() * 100)
    low = [r["region"] for r in sorted(regions, key=lambda r: r["coverage_percent"])
           if r["coverage_percent"] <= coverage - 5][:3]
    return dict(image=image, mask=Image.fromarray(mask.astype("uint8") * 255),
                overlay=overlay, grid_map=grid_map, regions=regions, coverage=coverage,
                inspect=low, grid=grid)


def report(result, source):
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        archive.writestr("summary.txt",
                         f"AEROX AGRI - RGB vegetation screening\nInput: {source}\n"
                         f"Green vegetation coverage: {result['coverage']:.3f}%\n"
                         f"Lower-cover cells: {', '.join(result['inspect']) or 'None flagged'}\n"
                         "Review low-cover cells against the source image. Soil, shadows and lighting can affect the mask.\n"
                         f"{YIELD_STATUS}\n")
        archive.writestr("grid.csv", csv_bytes(result["regions"],
                         ["region", "coverage_percent", "vegetation_pixels", "total_pixels"]))
        for name in ("overlay", "mask", "grid_map"):
            png = io.BytesIO()
            result[name].save(png, format="PNG")
            archive.writestr(f"{name}.png", png.getvalue())
    return out.getvalue()

