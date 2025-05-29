"""
color_palettes package
----------------------
任意の *.py ファイルを置いておくだけで自動発見。
各モジュールは以下のどちらかを *必ず* 提供すること:

* forward_lut() -> np.ndarray (shape: [65_536, 3], dtype:uint8)
* LUT (np.ndarray 型で直接定義)

オプションで:
    name        : 画面表示用の名前（str）
    description : 説明文（str）
"""

from functools import lru_cache
import importlib
import pkgutil
from pathlib import Path
import numpy as np

def _discover():
    """color_palettes 配下の *.py を走査して {name: module} dict を返す"""
    modules = {}
    for _, modname, ispkg in pkgutil.iter_modules(__path__):
        if ispkg:
            continue
        modules[modname] = importlib.import_module(f"{__name__}.{modname}")
    return modules

_MODULES = _discover()

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def list_palettes():
    """利用可能なパレット名一覧"""
    return sorted(_MODULES.keys())


@lru_cache(maxsize=None)
def get_palette(name: str = "cubehelix16x16") -> np.ndarray:
    """
    指定名の forward LUT を返す。
    - まず module.forward_lut() を試みる
    - 次に module.LUT ndarray を探す
    """
    if name not in _MODULES:
        raise ValueError(f"Unknown palette '{name}'. Available: {list_palettes()}")
    mod = _MODULES[name]

    if hasattr(mod, "forward_lut") and callable(mod.forward_lut):
        return mod.forward_lut()
    if hasattr(mod, "LUT"):
        lut = mod.LUT
        # 軽く型チェック
        if isinstance(lut, np.ndarray) and lut.shape == (65_536, 3):
            return lut.astype(np.uint8, copy=False)
        raise TypeError(f"{name}.LUT must be uint8[65536,3] ndarray")
    raise AttributeError(f"{name} must expose forward_lut() or LUT ndarray.")
