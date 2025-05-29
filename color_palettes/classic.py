"""
Classic blue-green-orange-white palette (TerrainRGB-like)
"""
import numpy as np
from functools import lru_cache
from skimage import color

_SEGMENTS = (
    (-11_000, "#081d58"), (-6_000, "#225ea8"), (-1_000, "#41b6c4"),
    (      0, "#66c2a5"), (   500, "#238b45"), ( 2_000, "#fdae61"),
    (  4_500, "#a6611a"), ( 9_000, "#ffffff"),
)

@lru_cache(maxsize=1)
def forward_lut() -> np.ndarray:
    h_pts = np.array([h for h, _ in _SEGMENTS], np.float32)
    rgb_pts = np.array(
        [tuple(int(c[i:i+2], 16)/255 for i in (1,3,5)) for _,c in _SEGMENTS],
        np.float32)
    lab_pts = color.rgb2lab(rgb_pts.reshape(-1,1,3)).reshape(-1,3)

    h16 = np.arange(65_536, dtype=np.float32)
    lab = np.empty((65_536,3), np.float32)
    h_m = h16 - 11_000.0
    for i in range(len(_SEGMENTS)-1):
        h0,h1 = h_pts[i:i+2]
        m = (h_m>=h0)&(h_m<=h1)
        t = (h_m[m]-h0)/(h1-h0)
        lab[m] = (1-t)[:,None]*lab_pts[i] + t[:,None]*lab_pts[i+1]
    rgb = color.lab2rgb(lab.reshape(-1,1,3)).reshape(-1,3)
    return np.clip(np.round(rgb*255),0,255).astype(np.uint8)
