#!/usr/bin/env python3
"""
toporgb codec – encode GeoTIFF DEMs to toporgb PNG *and* JSON metadata, or decode back using JSON.
---------------------------------------------------------------------

* Encoding  : *.tif[f]  →  *.png + *.json   (24-bit toporgb, perceptual palette)
  Run: python toporgb_codec.py "./dem_tiles/*.tif"

* Decoding  : *.png + *.json     →  *.tif   (Int16 DEM, 1 m resolution)
  Run: python toporgb_codec.py "./png_tiles/*.png"

Geo-reference
-------------
When encoding, we extract and save the source affine transform and CRS into JSON.
When decoding, if JSON is available, we restore the transform/CRS; otherwise a dummy north-up GT is used.
"""

from __future__ import annotations
import sys, glob, json, zlib, base64, warnings, rasterio
from pathlib import Path
from functools import lru_cache
from typing import Sequence, Iterable, Tuple

import numpy as np
from PIL import Image
from rasterio.transform import Affine
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # type: ignore

# -------------------------------------------------------------------- #
# 1. toporgb palette definition (must be identical to the encoder!)
# -------------------------------------------------------------------- #
_SEGMENTS: Tuple[Tuple[int, str], ...] = (
    (-11_000, "#081d58"), (-6_000, "#225ea8"), (-1_000, "#41b6c4"),
    (      0, "#66c2a5"), (   500, "#238b45"), ( 2_000, "#fdae61"),
    (  4_500, "#a6611a"), ( 9_000, "#ffffff"),
)

try:
    from skimage import color  # perceptual Lab ⇄ sRGB conversion
except ImportError:
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
    h_m  = h16 - 11_000.0  # back-to-metres

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
    inv = np.full((256, 256, 256), 65_535, dtype=np.uint16)  # sentinel = clip
    inv[rgb[:, 0], rgb[:, 1], rgb[:, 2]] = np.arange(65_536, dtype=np.uint16)
    return inv

# -------------------------------------------------------------------- #
# 4. Base64 + zlib utilities for JSON embedding
# -------------------------------------------------------------------- #
def to_b64z(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 9)).decode()

def from_b64z(txt: str) -> bytes:
    return zlib.decompress(base64.b64decode(txt))

# -------------------------------------------------------------------- #
# 5. Core converters
# -------------------------------------------------------------------- #

def height_to_h16(dem_m: np.ndarray) -> np.ndarray:
    """metres → uint16   (offset +11 000 m, 1 m resolution)"""
    return np.clip(np.round(dem_m + 11_000.0), 0, 65_535).astype(np.uint16)


def encode_dem_to_png(dem: np.ndarray) -> Image.Image:
    """DEM array → PIL.Image (RGB, toporgb)."""
    rgb = lut_forward()[height_to_h16(dem)]  # (H, W, 3) uint8
    return Image.fromarray(rgb, mode="RGB")


def decode_png_to_dem(rgb_img: Image.Image, *, dtype=np.int16) -> np.ndarray:
    """toporgb PNG → DEM array (dtype = int16 or float32)."""
    rgb = np.asarray(rgb_img.convert("RGB"), np.uint8)
    h16 = lut_inverse()[rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    if dtype == np.int16:
        return (h16.astype(np.int32) - 11_000).astype(np.int16)
    return h16.astype(np.float32) - 11_000.0

# -------------------------------------------------------------------- #
# 6. I/O path helpers
# -------------------------------------------------------------------- #

def tif_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith((".tif", ".tiff"))]


def png_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith(".png")]

# -------------------------------------------------------------------- #
# 7. Batch routines with JSON I/O
# -------------------------------------------------------------------- #
def batch_encode(tifs: Iterable[Path]) -> None:
    for tif in tqdm(list(tifs), desc="Encoding"):
        with rasterio.open(tif) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        img = encode_dem_to_png(dem)
        png_path = tif.with_suffix(".png")
        img.save(png_path, optimize=True, compress_level=9)

        # Prepare JSON metadata
        h, w = dem.shape
        lut = lut_forward()
        nudged = np.array([], dtype=np.uint32)
        payload = {
            "width": w,
            "height": h,
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "crs_wkt": crs.to_wkt() if crs is not None else None,
            "lut_b64": to_b64z(lut.tobytes()),
            "nudged_b64": to_b64z(nudged.tobytes()),
        }
        json_path = tif.with_suffix(".json")
        json_path.write_text(json.dumps(payload, separators=(",", ":")))


def batch_decode(pngs: Iterable[Path]) -> None:
    for png in tqdm(list(pngs), desc="Decoding"):
        dem = decode_png_to_dem(Image.open(png), dtype=np.int16)
        # Attempt to read JSON metadata
        json_path = png.with_suffix('.json')
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            t = meta.get('transform', None)
            crs_wkt = meta.get('crs_wkt', None)
            if t:
                transform = Affine(*t)
            else:
                # fallback north-up dummy
                h, w = dem.shape
                transform = Affine(1, 0, 0, 0, -1, h)
            crs = rasterio.crs.CRS.from_wkt(crs_wkt) if crs_wkt else None
        else:
            warnings.warn(f"No JSON for {png.name}: using default north-up transform/CRS")
            # default dummy transform
            h, w = dem.shape
            transform = Affine(1, 0, 0, 0, -1, h)
            crs = None

        out_tif = png.with_suffix('.tif')
        h, w = dem.shape
        with rasterio.open(
            out_tif, 'w', driver='GTiff', width=w, height=h,
            count=1, dtype='int16', compress='LZW', predictor=2,
            transform=transform, crs=crs
        ) as dst:
            dst.write(dem, 1)

# -------------------------------------------------------------------- #
# 8. Main
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
