"""
Viridis perceptually-uniform palette (256 × 256 × 256 forward lookup table)

* At the lowest value (0) → near-black navy; at the highest value → yellow–green
* Automatically retrieves via Matplotlib if available. Otherwise it raises an informative error.
"""
import numpy as np
from functools import lru_cache
from skimage import color

try:
    from matplotlib import cm
except ImportError as e:
    # If Matplotlib isn't installed, inform the user how to install it
    raise ImportError(
        "The viridis palette requires matplotlib. "
        "Please install it with: pip install matplotlib"
    ) from e


@lru_cache(maxsize=1)
def forward_lut() -> np.ndarray:
    """
    Build a uint8 lookup table of shape [65_536, 3]:

    Steps:
    1. Sample the 256-color Viridis colormap.
    2. Convert from sRGB to CIE Lab, then linearly interpolate in Lab space
       to expand to 65,536 discrete levels.
    3. Convert back from Lab to sRGB, clip to [0,255], and cast to uint8.
    """
    # 1. Sample 256 colors from the Viridis colormap
    #    cm.get_cmap("viridis", 256) returns RGBA; we slice off the alpha channel.
    rgb256 = cm.get_cmap("viridis", 256)(np.linspace(0, 1, 256))[:, :3]

    #    Convert the 256 sRGB colors to Lab color space
    lab256 = color.rgb2lab(rgb256.reshape(-1, 1, 3)).reshape(-1, 3)

    # 2. Prepare to interpolate to 65,536 evenly spaced levels in Lab space
    h16 = np.arange(65_536, dtype=np.float32)

    #    Map to the range [0, 255], which corresponds to indices in our 256-color array
    t = (h16 / 65_535) * 255.0

    #    For each target level, find the lower and upper 256-color indices
    i0 = t.astype(np.int16)                  # lower index
    i1 = np.clip(i0 + 1, 0, 255)             # upper index, clamped to [0,255]

    #    Compute interpolation weights (fractional part of t)
    w1 = t - i0                              # weight for the upper color
    w0 = 1.0 - w1                            # weight for the lower color

    #    Linearly interpolate between lab256[i0] and lab256[i1]
    lab = lab256[i0] * w0[:, None] + lab256[i1] * w1[:, None]

    # 3. Convert interpolated Lab values back to sRGB
    rgb = color.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)

    #    Scale to [0,255], round, clip, and cast to uint8
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
