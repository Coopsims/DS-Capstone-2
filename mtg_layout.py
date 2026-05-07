"""
Shared layout module for the MTG card identifier pipeline.

Centralizes:
  - Region crops for modern-frame cards (art, title, type line, set symbol)
  - Frame classification (works whether or not metadata has layout fields)
  - Hash functions used by both training and inference
  - Detection-validation heuristic
"""
import cv2
import numpy as np
from typing import Dict, Optional

# Canonical rectified card size. Aspect = 488/680 = 0.7176, matching MTG's
# 63mm/88mm = 0.7159.
CARD_W = 488
CARD_H = 680


# ---------------------------------------------------------------------------
# Region geometry for modern-frame cards
#
# All offsets are fractions of the canonical card. Values cover the M15+
# frame (2014 onwards). Old frames (1993/1997/2003) and Showcase/Borderless
# treatments don't fit these exactly - those go down the whole-card path.

REGIONS_MODERN = {
    'art':        ((0.06, 0.94), (0.12, 0.55)),
    'title':      ((0.06, 0.75), (0.04, 0.10)),
    'mana_cost':  ((0.75, 0.94), (0.04, 0.10)),
    'type_line':  ((0.06, 0.82), (0.56, 0.61)),
    'set_symbol': ((0.82, 0.94), (0.56, 0.61)),
    'text_box':   ((0.07, 0.93), (0.62, 0.90)),
    'pt_box':     ((0.78, 0.95), (0.88, 0.95)),
}


def crop_region(card_image: np.ndarray, region_name: str,
                regions: Dict = None) -> np.ndarray:
    regions = regions or REGIONS_MODERN
    (xs, xe), (ys, ye) = regions[region_name]
    h, w = card_image.shape[:2]
    return card_image[int(h*ys):int(h*ye), int(w*xs):int(w*xe)]


def extract_art_crop(card_image: np.ndarray) -> np.ndarray:
    return crop_region(card_image, 'art')


def extract_whole_card(card_image: np.ndarray, inset_frac: float = 0.03) -> np.ndarray:
    """
    Whole-card region for the fallback path. Slightly inset to avoid the
    rounded corners and any background bleed from imperfect rectification.
    """
    h, w = card_image.shape[:2]
    xi = int(w * inset_frac)
    yi = int(h * inset_frac)
    return card_image[yi:h-yi, xi:w-xi]


# ---------------------------------------------------------------------------
# Frame classification
#
# Maps a card to one of three processing strategies:
#   'modern'  : standard layout, multi-region extraction works
#   'fullart' : full-art / borderless / extended, only whole-card hash works
#   'special' : split / flip / saga / class / transform / etc.
#
# Training labels come from this function applied to Scryfall metadata.
# At inference we predict it from pixels with a small CNN.

FRAME_CLASSES = ['modern', 'fullart', 'special']
FRAME_CLASS_TO_IDX = {c: i for i, c in enumerate(FRAME_CLASSES)}

# Layout names from Scryfall that don't render as a normal single-face card
SPECIAL_LAYOUTS = {
    'split', 'flip', 'saga', 'class', 'planar', 'scheme',
    'leveler', 'meld', 'adventure', 'mutate', 'prototype',
    'case', 'battle', 'transform', 'modal_dfc', 'reversible_card',
    'double_faced_token', 'art_series', 'host', 'augment',
}


def classify_frame_from_metadata(card: dict, image_path: Optional[str] = None) -> str:
    """
    Map a Scryfall card record to a frame class label.

    Tries metadata fields first. If the relevant fields are missing
    (i.e. the metadata file was downloaded with a slim schema), falls back
    to image-based heuristics by inspecting the card image at image_path.

    image_path is optional - omit it for fast metadata-only classification.
    """
    # ---- Path 1: metadata-driven (preferred when fields exist)
    layout = card.get('layout')
    if layout in SPECIAL_LAYOUTS:
        return 'special'

    if card.get('full_art') is True:
        return 'fullart'

    if card.get('border_color') == 'borderless':
        return 'fullart'

    effects = card.get('frame_effects')
    if effects:
        if 'showcase' in effects or 'extendedart' in effects:
            return 'fullart'

    # If we got useful metadata signal that says "this is normal", trust it
    if layout in ('normal', 'token', 'emblem'):
        return 'modern'

    # ---- Path 2: image-based fallback for slim metadata
    # If we get here, the metadata didn't tell us much. Look at the image.
    if image_path is not None:
        cls = _classify_frame_from_image(image_path)
        if cls is not None:
            return cls

    # ---- Default: assume modern. Better to mis-classify a few full-art lands
    # as modern (they'll get a slightly wrong art crop but still embed) than
    # to mis-classify everything as fullart (we'd never use the art signal).
    return 'modern'


def _classify_frame_from_image(image_path: str) -> Optional[str]:
    """
    Look at a card image and decide whether it has a standard frame.

    Heuristic: check whether there's a roughly-uniform light-colored title
    bar at the top. Modern frame cards have one. Full-art lands and many
    showcase treatments don't.

    Returns 'modern' / 'fullart' / None (if we can't decide).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        if img.shape[0] != CARD_H or img.shape[1] != CARD_W:
            img = cv2.resize(img, (CARD_W, CARD_H))
    except Exception:
        return None

    # Title bar region: top 10% of the card, columns 6-75% (skipping mana cost)
    h, w = img.shape[:2]
    title = img[int(h*0.04):int(h*0.10), int(w*0.06):int(w*0.75)]

    # Modern frame title bars are nearly uniform in color (white-ish or
    # colored stripe). Full-art cards have varied content there (sky, water,
    # mountains, text overlay, etc).
    gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)
    title_std = float(gray.std())

    # Type-line region: similar idea, sits between art and text box
    type_line = img[int(h*0.55):int(h*0.61), int(w*0.06):int(w*0.82)]
    type_gray = cv2.cvtColor(type_line, cv2.COLOR_BGR2GRAY)
    type_std = float(type_gray.std())

    # Empirical thresholds tuned on a Scryfall sample.
    # Both regions low std -> classic modern frame
    # Both regions high std -> full-art (no frame elements there)
    if title_std < 35 and type_std < 35:
        return 'modern'
    if title_std > 55 and type_std > 55:
        return 'fullart'
    # Ambiguous - let the caller default
    return None


# ---------------------------------------------------------------------------
# Hashing - 64-bit pHash and dHash for fast retrieval

def phash_64(image: np.ndarray, hash_size: int = 8, dct_size: int = 32) -> np.uint64:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (dct_size, dct_size),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(resized)[:hash_size, :hash_size]
    flat = block.flatten()
    bits = (flat > np.median(flat[1:])).astype(np.uint64)
    h = np.uint64(0)
    for b in bits:
        h = (h << np.uint64(1)) | b
    return h


def dhash_64(image: np.ndarray, hash_size: int = 8) -> np.uint64:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size),
                         interpolation=cv2.INTER_AREA)
    diff = (resized[:, 1:] > resized[:, :-1]).astype(np.uint64).flatten()
    h = np.uint64(0)
    for b in diff:
        h = (h << np.uint64(1)) | b
    return h


def hamming_distance_vectorized(query: np.uint64, db: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(db, query)
    if hasattr(np, 'bit_count'):
        return np.bit_count(xor).astype(np.int32)
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


# ---------------------------------------------------------------------------
# Detection validation

def is_valid_card_detection(warped: np.ndarray, min_std: float = 25.0,
                            min_mean: float = 20.0, max_mean: float = 235.0) -> bool:
    """Reject obviously-wrong detector output."""
    if warped is None or warped.size == 0:
        return False
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY) if warped.ndim == 3 else warped
    if gray.std() < min_std:
        return False
    if gray.mean() < min_mean or gray.mean() > max_mean:
        return False
    return True


# ---------------------------------------------------------------------------
# Convenience: inspect what fields a metadata file actually has

def inspect_metadata(metadata_path: str, n_sample: int = 500) -> dict:
    """
    Sample the metadata and return a report of what frame-related fields
    are present and how often they take useful values. Use this to debug
    the "everything classifies as modern" failure mode.
    """
    import json
    from collections import Counter
    import random

    with open(metadata_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    sample = random.sample(cards, min(n_sample, len(cards)))

    report = {
        'total_cards': len(cards),
        'sampled': len(sample),
        'has_layout': sum(1 for c in sample if 'layout' in c),
        'has_full_art': sum(1 for c in sample if 'full_art' in c),
        'has_border_color': sum(1 for c in sample if 'border_color' in c),
        'has_frame_effects': sum(1 for c in sample if 'frame_effects' in c),
        'has_frame': sum(1 for c in sample if 'frame' in c),
        'layout_values': Counter(c.get('layout', '<missing>') for c in sample).most_common(15),
        'border_color_values': Counter(c.get('border_color', '<missing>') for c in sample).most_common(),
        'full_art_values': Counter(c.get('full_art', '<missing>') for c in sample).most_common(),
    }
    return report