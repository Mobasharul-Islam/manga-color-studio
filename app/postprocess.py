"""Local CPU post-processing that restores detail the diffusion pass tends to smear.

- Lineart blend: multiplies the original black & white lines over the colorized page,
  so lines and screentone edges stay as crisp as the scan.
- Bubble restore: finds speech bubbles in the original scan (white, enclosed regions
  holding several small dark glyphs) and pastes them back untouched, so text is never
  garbled or tinted.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


@dataclass
class PostSettings:
    lineart_strength: float = 0.5   # 0 = off, 1 = full multiply of original lines
    preserve_bubbles: bool = False

    @property
    def is_noop(self) -> bool:
        return self.lineart_strength <= 0 and not self.preserve_bubbles


def _line_map(gray: np.ndarray) -> np.ndarray:
    """1.0 where the scan is paper, falling towards 0 on ink. Light screentone barely darkens."""
    return np.clip(gray.astype(np.float32) / 200.0, 0.0, 1.0)


def _solidity(filled: np.ndarray) -> float:
    edge = filled & ~ndimage.binary_erosion(filled)
    ys, xs = np.nonzero(edge)
    if len(xs) < 3:
        return 0.0
    try:
        return float(filled.sum() / ConvexHull(np.c_[xs, ys]).volume)
    except QhullError:
        return 0.0


def detect_bubbles(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of speech bubbles (interior + text) in a grayscale scan."""
    h, w = gray.shape
    page_area = h * w
    white = gray > 225
    labels, n = ndimage.label(white)
    if n == 0:
        return np.zeros_like(white)

    # Regions touching the page edge are page margins or open panels, never bubbles.
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    areas = np.bincount(labels.ravel(), minlength=n + 1)

    mask = np.zeros_like(white)
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None or idx in edge_labels:
            continue
        if not (0.0015 * page_area <= areas[idx] <= 0.12 * page_area):
            continue
        region = labels[sl] == idx
        filled = ndimage.binary_fill_holes(region)
        filled_area = filled.sum()
        # Bubbles are compact ovals/boxes; long thin gutters and odd shapes are not.
        if filled_area / filled.size < 0.5:
            continue
        holes = filled & ~region
        hole_frac = holes.sum() / filled_area
        if not (0.01 <= hole_frac <= 0.4):
            continue
        # Text is many small glyphs; a drawn object is one big hole.
        hole_labels, n_holes = ndimage.label(holes)
        if n_holes < 3:
            continue
        if np.bincount(hole_labels.ravel())[1:].max() > 0.1 * filled_area:
            continue
        # Lettering is solid ink; faces and hair inside a white area carry gray shading.
        hole_px = gray[sl][holes]
        if hole_px.mean() > 140 or ((hole_px > 60) & (hole_px < 200)).mean() > 0.47:
            continue
        # Bubbles are near-convex; skin and clothing regions follow irregular outlines.
        if _solidity(filled) < 0.93:
            continue
        mask[sl] |= filled

    return ndimage.binary_dilation(mask, iterations=2)


def apply(color: Image.Image, original: Image.Image, settings: PostSettings) -> Image.Image:
    if settings.is_noop:
        return color
    color = color.convert("RGB")
    gray_img = original.convert("L")
    if gray_img.size != color.size:
        gray_img = gray_img.resize(color.size, Image.LANCZOS)
    gray = np.asarray(gray_img)
    out = np.asarray(color).astype(np.float32)

    if settings.lineart_strength > 0:
        s = float(min(settings.lineart_strength, 1.0))
        factor = 1.0 - s * (1.0 - _line_map(gray))
        out *= factor[..., None]

    if settings.preserve_bubbles:
        bubbles = detect_bubbles(gray)
        out[bubbles] = gray[bubbles][:, None]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
