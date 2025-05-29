#!/usr/bin/env python3
"""
TopoRGB Codec: Encode GeoTIFF DEMs to TopoRGB PNG + JSON metadata, or decode back into GeoTIFF DEMs.
Features:
  - Forward & inverse LUTs via perceptual Lab interpolation.
  - Optional fuzzy color matching (nearest‐neighbor) for slight color variations.
  - JSON metadata contains affine transform & CRS for full geo-referencing.

Usage:
  # Encode DEMs to PNG + metadata:
  python toporgb_codec.py encode "./dem_tiles/*.tif" [--palette NAME]

  # Decode PNG + metadata back to DEM GeoTIFFs:
  python toporgb_codec.py decode "./png_tiles/*.png" [--palette NAME] [--fuzzy]
"""

from __future__ import annotations
import sys
import glob
import json
import zlib
import base64
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Sequence, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import argparse

from color_palettes import get_palette, list_palettes

# ---------------------------------------------------------------------
# Utilities: Base64 + zlib for embedding LUT in metadata
# ---------------------------------------------------------------------
def _compress_bytes(data: bytes) -> str:
    """Compress and base64-encode bytes."""
    return base64.b64encode(zlib.compress(data, level=9)).decode()


def _decompress_bytes(txt: str) -> bytes:
    """Base64-decode and decompress bytes."""
    return zlib.decompress(base64.b64decode(txt))

# ---------------------------------------------------------------------
# LUT Selection: CLI override > embedded LUT > named palette > default
# ---------------------------------------------------------------------

def select_lut(cli_name: Optional[str], metadata: Optional[dict]) -> np.ndarray:
    """Return forward LUT (shape=(65536,3)) based on CLI, metadata, or default."""
    if cli_name:
        return get_palette(cli_name)

    if metadata:
        if (b64 := metadata.get("lut_b64")):
            raw = _decompress_bytes(b64)
            return np.frombuffer(raw, np.uint8).reshape(65536, 3)
        if (name := metadata.get("palette")):
            return get_palette(name)

    # default palette
    return get_palette("classic")

# ---------------------------------------------------------------------
# Forward LUT (height_index -> RGB)
# ---------------------------------------------------------------------

_SELECTED_PALETTE: str  # set in main()

@lru_cache(maxsize=1)
def lut_forward() -> np.ndarray:
    """Return the selected forward LUT array (uint8 RGB rows)."""
    return get_palette(_SELECTED_PALETTE)

# ---------------------------------------------------------------------
# Inverse LUT (exact color match) and KDTree (fuzzy match)
# ---------------------------------------------------------------------
# Cache full-cube inverse LUT for exact match (≈33 MB), keyed on bytes
@lru_cache(maxsize=None)
def _build_inverse_lut(lut_bytes: bytes) -> np.ndarray:
    """Return uint16[256,256,256] mapping sRGB triplets to height_index."""
    lut = np.frombuffer(lut_bytes, np.uint8).reshape(65536, 3)
    inv = np.full((256, 256, 256), 65535, dtype=np.uint16)
    indices = np.arange(lut.shape[0], dtype=np.uint16)
    inv[lut[:, 0], lut[:, 1], lut[:, 2]] = indices
    return inv

# Cache KDTree lookup for fuzzy matching, keyed on bytes
@lru_cache(maxsize=None)
def _build_kdtree(lut_bytes: bytes) -> tuple[cKDTree, np.ndarray]:
    """Return (KDTree, heights[]) for nearest-neighbor lookup."""
    arr = np.frombuffer(lut_bytes, np.uint8).reshape(65536, 3).astype(np.float32)
    heights = np.arange(arr.shape[0], dtype=np.uint16)
    tree = cKDTree(arr)
    return tree, heights

# ---------------------------------------------------------------------
# Core Conversions: height <-> uint16 <-> RGB
# ---------------------------------------------------------------------

def metres_to_index(dem_m: np.ndarray) -> np.ndarray:
    """Convert metre-array to uint16 index with +11,000m offset."""
    return np.clip(np.round(dem_m + 11000.0), 0, 65535).astype(np.uint16)


def index_to_metres(index: np.ndarray) -> np.ndarray:
    """Convert uint16 index back to metre-array (int16 or float32)."""
    return index.astype(np.float32) - 11000.0


def encode_dem_to_image(dem: np.ndarray) -> Image.Image:
    """Encode DEM (metres) to TopoRGB PIL Image."""
    lut = lut_forward()
    indices = metres_to_index(dem)
    rgb = lut[indices]
    return Image.fromarray(rgb, mode="RGB")


def decode_image_to_dem(
    img: Image.Image,
    *,
    dtype: type = np.int16,
    fuzzy: bool = False,
    lut: np.ndarray,
) -> np.ndarray:
    """Decode TopoRGB PIL Image to DEM metres array."""
    rgb = np.asarray(img.convert("RGB"), np.uint8)
    # exact match via cached full-cube LUT inverse
    index = _build_inverse_lut(lut.tobytes())[rgb[..., 0], rgb[..., 1], rgb[..., 2]]

    if fuzzy and (mask := (index == 65535)).any():
        # fuzzy match via cached KDTree
        tree, heights = _build_kdtree(lut.tobytes())
        pts = rgb[mask].reshape(-1, 3).astype(np.float32)
        _, idxs = tree.query(pts)
        index[mask] = heights[idxs]

    metres = index_to_metres(index)
    return metres.astype(dtype)

# ---------------------------------------------------------------------
# File I/O: glob helpers and batch routines
# ---------------------------------------------------------------------

def glob_paths(pattern: str, exts: Tuple[str, ...]) -> list[Path]:
    """Return sorted Paths matching glob pattern and extensions."""
    return sorted(
        Path(p) for p in glob.glob(pattern) if p.lower().endswith(exts)
    )


def batch_encode(pattern: str) -> None:
    """Encode all GeoTIFF DEMs matching pattern."""
    tifs = glob_paths(pattern, (".tif", ".tiff"))
    if not tifs:
        raise FileNotFoundError("No GeoTIFF files found for encoding.")

    for tif in tifs:
        with rasterio.open(tif) as src:
            dem = src.read(1).astype(np.float32)
            transform, crs = src.transform, src.crs

        img = encode_dem_to_image(dem)
        png_file = tif.with_suffix(".png")
        img.save(png_file, optimize=True, compress_level=9)

        metadata = {
            "width": dem.shape[1],
            "height": dem.shape[0],
            "transform": list(transform[:6]),
            "crs_wkt": crs.to_wkt() if crs else None,
            "lut_b64": _compress_bytes(lut_forward().tobytes()),
            "palette": _SELECTED_PALETTE,
        }
        json_file = tif.with_suffix(".json")
        json_file.write_text(json.dumps(metadata, separators=(",", ":")))


def batch_decode(pattern: str, fuzzy: bool) -> None:
    """Decode all TopoRGB PNGs matching pattern back to GeoTIFF DEMs."""
    pngs = glob_paths(pattern, (".png",))
    if not pngs:
        raise FileNotFoundError("No PNG files found for decoding.")

    for png in pngs:
        meta_file = png.with_suffix(".json")
        metadata = json.loads(meta_file.read_text()) if meta_file.exists() else None

        lut = select_lut(_SELECTED_PALETTE, metadata)
        dem = decode_image_to_dem(
            Image.open(png), dtype=np.int16, fuzzy=fuzzy, lut=lut
        )

        if metadata:
            t = metadata.get("transform")
            transform = Affine(*t) if t else Affine(1, 0, 0, 0, -1, dem.shape[0])
            crs = CRS.from_wkt(metadata.get("crs_wkt")) if metadata.get("crs_wkt") else None
        else:
            warnings.warn(f"Missing JSON for {png.name}, using default transform/CRS.")
            transform = Affine(1, 0, 0, 0, -1, dem.shape[0])
            crs = None

        out_tif = png.with_suffix(".tif")
        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            dtype="int16",
            compress="LZW",
            predictor=2,
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(dem, 1)

# ---------------------------------------------------------------------
# CLI: Argument parsing and main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TopoRGB codec: encode GeoTIFFs or decode TopoRGB PNGs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode GeoTIFF DEMs to TopoRGB PNGs")
    enc.add_argument("pattern", help="Glob for input .tif/.tiff files")
    enc.add_argument(
        "--palette",
        default="classic",
        choices=list_palettes(),
        help="Name of color palette",
    )

    dec = sub.add_parser("decode", help="Decode TopoRGB PNGs to GeoTIFF DEMs")
    dec.add_argument("pattern", help="Glob for input .png files")
    dec.add_argument(
        "--palette",
        default="cubehelix16x16",
        choices=list_palettes(),
        help="Name of color palette",
    )
    dec.add_argument(
        "--fuzzy",
        action="store_true",
        help="Enable fuzzy color matching for slight color variations",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global _SELECTED_PALETTE
    _SELECTED_PALETTE = args.palette

    if args.command == "encode":
        batch_encode(args.pattern)
    elif args.command == "decode":
        batch_decode(args.pattern, fuzzy=args.fuzzy)


if __name__ == "__main__":
    main()
