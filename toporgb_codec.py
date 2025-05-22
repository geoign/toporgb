#!/usr/bin/env python3
"""
toporgb codec – encode GeoTIFF DEMs to toporgb PNG *or* decode back.
---------------------------------------------------------------------

* Encoding  : *.tif[f]  →  *.png   (24-bit toporgb, perceptual palette)
Run: python toporgb_codec.py "./dem_tiles/*.tif"

* Decoding  : *.png     →  *.tif   (Int16 DEM, 1 m resolution)
Run: python toporgb_codec.py "./png_tiles/*.png"

Geo-reference
-------------
The script does **not** copy any spatial reference when *encoding* because
PNG has no native geo-tags.  
When *decoding* we write a dummy affine     (pixelHeight = −1) so that the
raster is “north-up” in GIS.  Change `transform` / `crs` to suit your data.
"""

from __future__ import annotations
import sys, glob
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Iterable, Sequence

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
try:               # optional progress bar
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x      # type: ignore

# -------------------------------------------------------------------- #
# 1. toporgb palette definition (must be identical to the encoder!)
# -------------------------------------------------------------------- #
_SEGMENTS: Tuple[Tuple[int, str], ...] = (
    (-11_000, "#081d58"), (-6_000, "#225ea8"), (-1_000, "#41b6c4"),
    (      0, "#66c2a5"), (   500, "#238b45"), ( 2_000, "#fdae61"),
    (  4_500, "#a6611a"), ( 9_000, "#ffffff"),
)

try:
    from skimage import color     # perceptual Lab ⇄ sRGB conversion
except ImportError as e:
    sys.exit("scikit-image is required – install with:  pip install scikit-image")

# -------------------------------------------------------------------- #
# 2. Forward LUT: height16 → RGB   • size = 65 536 × 3  (192 KiB)
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def lut_forward() -> np.ndarray:
    """Return uint8[65 536, 3]   mapping unsigned 16-bit heights to sRGB."""
    h_pts = np.array([h for h, _ in _SEGMENTS], np.float32)
    rgb_pts = np.array(
        [tuple(int(c[i:i+2], 16) / 255 for i in (1, 3, 5)) for _, c in _SEGMENTS],
        np.float32,
    )
    lab_pts = color.rgb2lab(rgb_pts.reshape(-1, 1, 3)).reshape(-1, 3)

    h16 = np.arange(65_536, dtype=np.float32)
    lab  = np.empty((65_536, 3), np.float32)
    h_m  = h16 - 11_000.0                       # back-to-metres

    for i in range(len(_SEGMENTS) - 1):
        h0, h1 = h_pts[i : i + 2]
        mask = (h_m >= h0) & (h_m <= h1)
        t = (h_m[mask] - h0) / (h1 - h0)
        lab[mask] = (1 - t)[:, None] * lab_pts[i] + t[:, None] * lab_pts[i + 1]

    rgb = color.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)

# -------------------------------------------------------------------- #
# 3. Inverse LUT: RGB → height16   • size = 32 MiB, built from forward
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def lut_inverse() -> np.ndarray:
    """Return uint16[256,256,256]   mapping sRGB triplets to height16."""
    rgb = lut_forward()
    inv = np.full((256, 256, 256), 65_535, dtype=np.uint16)   # sentinel = clip
    inv[rgb[:, 0], rgb[:, 1], rgb[:, 2]] = np.arange(65_536, dtype=np.uint16)
    return inv

# -------------------------------------------------------------------- #
# 4. Core helpers
# -------------------------------------------------------------------- #
def height_to_h16(dem_m: np.ndarray) -> np.ndarray:
    """metres → uint16   (offset +11 000 m, 1 m resolution)"""
    return np.clip(np.round(dem_m + 11_000.0), 0, 65_535).astype(np.uint16)


def encode_dem_to_png(dem: np.ndarray) -> Image.Image:
    """DEM array → PIL.Image (RGB, toporgb)."""
    rgb = lut_forward()[height_to_h16(dem)]        # (H, W, 3) uint8
    return Image.fromarray(rgb, mode="RGB")


def decode_png_to_dem(rgb_img: Image.Image, *, dtype=np.int16) -> np.ndarray:
    """toporgb PNG → DEM array (dtype = int16 or float32)."""
    rgb = np.asarray(rgb_img.convert("RGB"), np.uint8)
    h16 = lut_inverse()[rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    if dtype == np.int16:
        return (h16.astype(np.int32) - 11_000).astype(np.int16)
    return h16.astype(np.float32) - 11_000.0

# -------------------------------------------------------------------- #
# 5. I/O helpers
# -------------------------------------------------------------------- #
def tif_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith((".tif", ".tiff"))]

def png_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith(".png")]

# -------------------------------------------------------------------- #
# 6. Batch routines
# -------------------------------------------------------------------- #
def batch_encode(tifs: Iterable[Path]) -> None:
    for tif in tqdm(list(tifs), desc="Encoding"):
        with rasterio.open(tif) as src:
            dem = src.read(1).astype(np.float32)   # first band
        img = encode_dem_to_png(dem)
        img.save(tif.with_suffix(".png"), optimize=True, compress_level=9)


def batch_decode(pngs: Iterable[Path]) -> None:
    for png in tqdm(list(pngs), desc="Decoding"):
        dem = decode_png_to_dem(Image.open(png), dtype=np.int16)
        h, w = dem.shape
        transform = Affine(1, 0, 0, 0, -1, h)      # north-up dummy GT
        with rasterio.open(
            png.with_suffix(".tif"), "w",
            driver="GTiff", width=w, height=h, count=1,
            dtype="int16", compress="LZW", predictor=2,
            transform=transform, crs=None
        ) as dst:
            dst.write(dem, 1)

# -------------------------------------------------------------------- #
# 7. Main
# -------------------------------------------------------------------- #
def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.tif"
    if pattern.lower().endswith((".tif", ".tiff")):
        files = tif_paths(pattern)
        if not files:
            sys.exit("No GeoTIFF found for encoding.")
        batch_encode(files)
    elif pattern.lower().endswith(".png"):
        files = png_paths(pattern)
        if not files:
            sys.exit("No PNG found for decoding.")
        batch_decode(files)
    else:
        sys.exit("Please supply a *.tif[f] pattern for encoding or *.png for decoding.")


if __name__ == "__main__":
    main()
