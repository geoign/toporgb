#!/usr/bin/env python3
"""
toporgb_codec.py – Encode GeoTIFF DEMs to TopoRGB PNG + JSON metadata, or decode back into GeoTIFF DEMs.
Features:
  • Forward and inverse look-up tables for height ↔ RGB mapping via perceptual Lab interpolation.
  • Optional "fuzzy" color matching mode for robust decoding with slight color variations (e.g., AI-generated images, JPEG compression).
  • JSON metadata includes affine transform and CRS WKT for full geo-referencing.

Usage:
  Encoding:
    python toporgb_codec.py "./dem_tiles/*.tif"
      - Input: GeoTIFF DEM (*.tif or *.tiff)
      - Output: TopoRGB PNG (*.png) + JSON metadata (*.json)

  Decoding:
    python toporgb_codec.py "./png_tiles/*.png" [--fuzzy]
      - Input: TopoRGB PNG (*.png) + JSON metadata (*.json)
      - Output: GeoTIFF DEM (*.tif, int16)
      - Use --fuzzy to enable nearest-neighbor color matching for non-exact LUT entries.

Requirements:
  • Python 3.7+
  • rasterio
  • Pillow
  • scikit-image
  • scipy (for KDTree in fuzzy mode)
  • tqdm (optional, for progress bars)

Geo-reference:
  When encoding, the source affine transform and CRS are saved into JSON. On decoding, these are restored if available; otherwise, a default north-up transform is used.
"""

from __future__ import annotations
import sys, glob, json, zlib, base64, warnings, rasterio
from pathlib import Path
from functools import lru_cache
from typing import Sequence, Iterable, Tuple
from scipy.spatial import cKDTree  # KDTree for fuzzy color matching
import numpy as np
from PIL import Image
from rasterio.transform import Affine
import argparse
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # fallback if tqdm is not installed

# -------------------------------------------------------------------- #
# 1. TopoRGB palette definition (must remain consistent across encoder/decoder)
# -------------------------------------------------------------------- #
_SEGMENTS: Tuple[Tuple[int, str], ...] = (
    (-11_000, "#081d58"), (-6_000, "#225ea8"), (-1_000, "#41b6c4"),
    (      0, "#66c2a5"), (   500, "#238b45"), ( 2_000, "#fdae61"),
    (  4_500, "#a6611a"), ( 9_000, "#ffffff"),
)

try:
    from skimage import color  # for perceptual Lab ⇄ sRGB conversion
except ImportError:
    sys.exit("scikit-image is required – install with:  pip install scikit-image")

# -------------------------------------------------------------------- #
# 2. Forward LUT: height16 → RGB   • size = 65 536 × 3 (uint8)
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def lut_forward() -> np.ndarray:
    """Return uint8[65536,3] mapping unsigned 16-bit heights to sRGB."""
    h_pts = np.array([h for h, _ in _SEGMENTS], np.float32)
    rgb_pts = np.array(
        [tuple(int(c[i:i+2], 16) / 255 for i in (1, 3, 5)) for _, c in _SEGMENTS],
        np.float32,
    )
    lab_pts = color.rgb2lab(rgb_pts.reshape(-1, 1, 3)).reshape(-1, 3)

    h16 = np.arange(65_536, dtype=np.float32)
    lab  = np.empty((65_536, 3), np.float32)
    h_m  = h16 - 11_000.0  # convert to metres

    for i in range(len(_SEGMENTS) - 1):
        h0, h1 = h_pts[i : i + 2]
        mask = (h_m >= h0) & (h_m <= h1)
        t = (h_m[mask] - h0) / (h1 - h0)
        lab[mask] = (1 - t)[:, None] * lab_pts[i] + t[:, None] * lab_pts[i + 1]

    rgb = color.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)

# -------------------------------------------------------------------- #
# 3. Inverse LUT: exact RGB → height16   • size = 256³ (uint16)
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def lut_inverse() -> np.ndarray:
    """Return uint16[256,256,256] mapping sRGB triplets to height16, with 65535 as sentinel."""
    rgb = lut_forward()
    inv = np.full((256, 256, 256), 65_535, dtype=np.uint16)
    inv[rgb[:, 0], rgb[:, 1], rgb[:, 2]] = np.arange(65_536, dtype=np.uint16)
    return inv

# -------------------------------------------------------------------- #
# 4. KDTree for fuzzy matching: nearest RGB neighbor → height16
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def lut_kdtree():
    """
    Return (tree, heights):
      - tree: cKDTree built over forward LUT RGB points
      - heights: uint16 array of corresponding height16 values
    """
    rgb = lut_forward().astype(np.float32)
    heights = np.arange(65_536, dtype=np.uint16)
    tree = cKDTree(rgb)
    return tree, heights

# -------------------------------------------------------------------- #
# 5. Base64 + zlib utilities for embedding LUT in JSON metadata
# -------------------------------------------------------------------- #
def to_b64z(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 9)).decode()

def from_b64z(txt: str) -> bytes:
    return zlib.decompress(base64.b64decode(txt))

# -------------------------------------------------------------------- #
# 6. Core converters: height ↔ PNG
# -------------------------------------------------------------------- #

def height_to_h16(dem_m: np.ndarray) -> np.ndarray:
    """Convert metres to uint16 with +11000m offset."""
    return np.clip(np.round(dem_m + 11_000.0), 0, 65_535).astype(np.uint16)


def encode_dem_to_png(dem: np.ndarray) -> Image.Image:
    """Encode DEM metres array to TopoRGB PIL Image."""
    rgb = lut_forward()[height_to_h16(dem)]
    return Image.fromarray(rgb, mode="RGB")


def decode_png_to_dem(
    rgb_img: Image.Image,
    *,
    dtype=np.int16,
    fuzzy: bool = False
) -> np.ndarray:
    """
    Decode TopoRGB PIL Image back to DEM.
    If fuzzy=True, perform nearest-neighbor search for colors not found exactly in LUT.
    """
    rgb = np.asarray(rgb_img.convert("RGB"), np.uint8)
    inv = lut_inverse()
    h16 = inv[rgb[..., 0], rgb[..., 1], rgb[..., 2]]

    if fuzzy:
        mask = (h16 == 65535)
        if mask.any():
            tree, heights = lut_kdtree()
            pts = rgb[mask].reshape(-1, 3).astype(np.float32)
            _, idxs = tree.query(pts)
            h16[mask] = heights[idxs]

    if dtype == np.int16:
        return (h16.astype(np.int32) - 11_000).astype(np.int16)
    return h16.astype(np.float32) - 11_000.0

# -------------------------------------------------------------------- #
# 7. I/O helpers and batch routines with JSON I/O
# -------------------------------------------------------------------- #

def tif_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith((".tif", ".tiff"))]

def png_paths(glob_pattern: str) -> Sequence[Path]:
    return [Path(p) for p in glob.glob(glob_pattern) if p.lower().endswith(".png")]

def batch_encode(tifs: Iterable[Path]) -> None:
    for tif in tqdm(list(tifs), desc="Encoding"):
        with rasterio.open(tif) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        img = encode_dem_to_png(dem)
        png_path = tif.with_suffix(".png")
        img.save(png_path, optimize=True, compress_level=9)

        payload = {
            "width": dem.shape[1],
            "height": dem.shape[0],
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "crs_wkt": crs.to_wkt() if crs is not None else None,
            "lut_b64": to_b64z(lut_forward().tobytes()),
            "nudged_b64": to_b64z(np.array([], dtype=np.uint32).tobytes()),
        }
        tif.with_suffix(".json").write_text(json.dumps(payload, separators=(",", ":")))

def batch_decode(pngs: Iterable[Path], *, fuzzy: bool = False) -> None:
    for png in tqdm(list(pngs), desc="Decoding"):
        dem = decode_png_to_dem(Image.open(png), dtype=np.int16, fuzzy=fuzzy)
        json_path = png.with_suffix('.json')
        if json_path.exists():
            meta = json.loads(json_path.read_text())
            t = meta.get('transform')
            crs_wkt = meta.get('crs_wkt')
            transform = Affine(*t) if t else Affine(1, 0, 0, 0, -1, dem.shape[0])
            crs = rasterio.crs.CRS.from_wkt(crs_wkt) if crs_wkt else None
        else:
            warnings.warn(f"No JSON for {png.name}: using default north-up transform/CRS")
            transform = Affine(1, 0, 0, 0, -1, dem.shape[0])
            crs = None

        out_tif = png.with_suffix('.tif')
        with rasterio.open(
            out_tif, 'w', driver='GTiff', width=dem.shape[1], height=dem.shape[0],
            count=1, dtype='int16', compress='LZW', predictor=2,
            transform=transform, crs=crs
        ) as dst:
            dst.write(dem, 1)

# -------------------------------------------------------------------- #
# 8. Argument parsing and main entry point
# -------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TopoRGB codec: encode GeoTIFFs or decode TopoRGB PNGs."
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern for input files (*.tif[f] for encode, *.png for decode)"
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy color matching on decode (nearest neighbor lookup)"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    pattern = args.pattern

    if pattern.lower().endswith((".tif", ".tiff")):
        files = tif_paths(pattern)
        if not files:
            sys.exit("No GeoTIFF found for encoding.")
        batch_encode(files)
    elif pattern.lower().endswith(".png"):
        files = png_paths(pattern)
        if not files:
            sys.exit("No PNG found for decoding.")
        batch_decode(files, fuzzy=args.fuzzy)
    else:
        sys.exit(
            "Please supply a *.tif[f] pattern for encoding or *.png for decoding."
        )

if __name__ == "__main__":
    main()
