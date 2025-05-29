"""
Cubehelix-16×16 palette  (JPEG-resilient, perceptually uniform)
"""
import numpy as np
from functools import lru_cache
from skimage import color

name = "cubehelix16x16"
description = "Cubehelix luminance + 16×16 a*b* grid (256×256×256 coverage)"

@lru_cache(maxsize=1)
def forward_lut() -> np.ndarray:
    gamma = 1.0          # 0.5～2 で暗部/明部の強調度
    rot   = -1.5         # 回転量。負なら時計回りの紫→黄系
    hue   = 1.2          # 彩度スケール（大きいほどカラフル）
    N = 65_536

    t = np.linspace(0, 1, N, dtype=np.float32)
    tgamma = t ** gamma

    # 1) L* のベース
    Lstar = tgamma * 100.0   # Lab L* 0–100

    # 2) 下位 8 bit を ab 格子にマッピング（±40）
    lo = (np.arange(N) & 0xFF).astype(np.int16)
    a_base = ((lo & 0x0F) - 8) * 5.0
    b_base = (((lo >> 4) - 8) * 5.0)

    # 3) Cubehelix 的な螺旋ノイズを ab に重畳
    phi = 2 * np.pi * (0.5 + rot * tgamma)
    a_star = a_base + hue * 20.0 * np.cos(phi)
    b_star = b_base + hue * 20.0 * np.sin(phi)

    lab = np.stack([Lstar, a_star, b_star], axis=1)
    rgb = color.lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
