"""
Train the identification components of the MTG pipeline.

Run train_detector.py first to get mtg_data/yolo_card_best.pt. This script
trains everything else:

  - Frame classifier: predicts modern / fullart / special from a rectified
    card image. Drives extraction strategy at inference time.
  - Art re-ranker: embedding model for the art crop (modern frame only).
  - Whole-card re-ranker: embedding model for the whole card (universal).
  - Set symbol classifier: predicts set code from the set symbol crop.
  - Hash database: pHash + dHash for both art crops AND whole cards.

Inputs:
  mtg_data/cards_metadata.json
  mtg_data/card_images/{id}.jpg
  mtg_data/yolo_card_best.pt (optional)

Outputs:
  mtg_data/phash_db.npz                 (art + whole_card hashes + frame class)
  mtg_data/frame_classifier.pth
  mtg_data/art_reranker.pth
  mtg_data/art_embeddings.npy
  mtg_data/whole_reranker.pth
  mtg_data/whole_embeddings.npy
  mtg_data/set_classifier.pth
  mtg_data/set_codes.json
  mtg_data/frame_classes_cache.json     (per-card frame class, computed once)

Usage:
  python train_identifier.py                    # full run
  python train_identifier.py --diagnose         # inspect metadata, then exit
  python train_identifier.py --skip frame art   # skip components
  python train_identifier.py --skip-train       # rebuild DB and embeddings only
  python train_identifier.py --no-image-fallback  # don't peek at images for
                                                   # frame classification (faster
                                                   # but worse if metadata is slim)
  python train_identifier.py --epochs 8         # shorter training
"""

import argparse
import json
import random
import sys
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mtg_layout import (
    CARD_W, CARD_H,
    REGIONS_MODERN, crop_region, extract_art_crop, extract_whole_card,
    classify_frame_from_metadata, FRAME_CLASSES, FRAME_CLASS_TO_IDX,
    phash_64, dhash_64, hamming_distance_vectorized,
    inspect_metadata,
)

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'

HASH_DB_PATH = DATA_DIR / 'phash_db.npz'
FRAME_CLF_PATH = DATA_DIR / 'frame_classifier.pth'
ART_RERANKER_PATH = DATA_DIR / 'art_reranker.pth'
ART_EMB_PATH = DATA_DIR / 'art_embeddings.npy'
WHOLE_RERANKER_PATH = DATA_DIR / 'whole_reranker.pth'
WHOLE_EMB_PATH = DATA_DIR / 'whole_embeddings.npy'
SET_CLF_PATH = DATA_DIR / 'set_classifier.pth'
SET_CODES_PATH = DATA_DIR / 'set_codes.json'
FRAME_CACHE_PATH = DATA_DIR / 'frame_classes_cache.json'

ART_INPUT = 160
WHOLE_INPUT = 224
FRAME_INPUT = 96
SETSYM_INPUT = 64
EMBEDDING_DIM = 256
DEFAULT_EPOCHS = 12
BATCH_SIZE = 128
SEED = 42


# ===========================================================================
# Loading helpers

def load_cards():
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    available = [i for i, c in enumerate(cards)
                 if (IMAGE_DIR / f"{c['id']}.jpg").exists()]
    return cards, available


def load_card_image(card_id: str) -> Optional[np.ndarray]:
    path = IMAGE_DIR / f"{card_id}.jpg"
    if not path.exists():
        return None
    img = np.array(Image.open(path).convert('RGB'))
    if img.shape[0] != CARD_H or img.shape[1] != CARD_W:
        img = cv2.resize(img, (CARD_W, CARD_H))
    return img


# ===========================================================================
# Frame class precomputation
#
# Classifying every card's frame can require reading the image (when
# metadata is slim). That's slow to do inline with training - so we do it
# once, cache the result to disk, and let every downstream component read
# from the cache.

def compute_frame_classes(cards, available, use_image_fallback: bool = True,
                          force_recompute: bool = False) -> dict:
    """
    Compute the frame class for every card. Cached on disk keyed by Scryfall ID.
    Returns: dict mapping card_id (str) -> frame class (str).
    """
    cache: dict = {}
    if FRAME_CACHE_PATH.exists() and not force_recompute:
        with open(FRAME_CACHE_PATH) as f:
            cache = json.load(f)
        # Only return cached values for cards still in available set
        avail_ids = {cards[i]['id'] for i in available}
        cache = {k: v for k, v in cache.items() if k in avail_ids}
        # If cache covers everything we need, we're done
        missing = [i for i in available if cards[i]['id'] not in cache]
        if not missing:
            print(f"  Loaded {len(cache)} cached frame classes")
            return cache
        print(f"  Cache covers {len(cache)} / {len(available)} cards, "
              f"computing {len(missing)} more...")
        to_compute = missing
    else:
        print(f"  Computing frame class for {len(available)} cards "
              f"(use_image_fallback={use_image_fallback})...")
        to_compute = available

    for idx in tqdm(to_compute, desc="  classifying"):
        card = cards[idx]
        img_path = str(IMAGE_DIR / f"{card['id']}.jpg") if use_image_fallback else None
        cache[card['id']] = classify_frame_from_metadata(card, image_path=img_path)

    with open(FRAME_CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    return cache


def report_frame_distribution(frame_classes: dict, label: str = "Frame distribution"):
    counts = Counter(frame_classes.values())
    total = sum(counts.values())
    print(f"  {label}:")
    for cls in FRAME_CLASSES:
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"    {cls:<10} {n:>7,}  ({pct:.1f}%)")
    return counts


# ===========================================================================
# Synthetic compositing

def make_random_background(size: int) -> np.ndarray:
    mode = random.random()
    if mode < 0.3:
        c = tuple(random.randint(0, 255) for _ in range(3))
        return np.full((size, size, 3), c, dtype=np.uint8)
    if mode < 0.55:
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        if random.random() < 0.5:
            for i in range(size):
                t = i / size
                bg[i, :] = (c1 * (1-t) + c2 * t).astype(np.uint8)
        else:
            for j in range(size):
                t = j / size
                bg[:, j] = (c1 * (1-t) + c2 * t).astype(np.uint8)
        return bg
    if mode < 0.8:
        bg = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return cv2.GaussianBlur(bg, (21, 21), 0)
    base = np.array([random.randint(60, 180) for _ in range(3)], dtype=np.float32)
    var = random.uniform(15, 40)
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    for j in range(size):
        wave = np.sin(j / random.uniform(8, 25)) * var
        bg[:, j] = np.clip(base + wave, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def synthesize_rectified_card(card_img: np.ndarray) -> np.ndarray:
    H_card, W_card = card_img.shape[:2]
    canvas = random.randint(500, 900)
    bg = make_random_background(canvas)

    scale = random.uniform(0.55, 0.85)
    new_h = int(canvas * scale)
    new_w = int(new_h * W_card / H_card)
    if new_w > canvas * 0.9:
        new_w = int(canvas * 0.9)
        new_h = int(new_w * H_card / W_card)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    angle = random.uniform(-15, 15)
    margin = int(max(new_w, new_h) * 0.6)
    margin = min(margin, canvas // 2 - 1)
    cx = random.randint(margin, max(margin + 1, canvas - margin))
    cy = random.randint(margin, max(margin + 1, canvas - margin))

    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    M[0, 2] += cx - new_w / 2
    M[1, 2] += cy - new_h / 2

    warped = cv2.warpAffine(card_resized, M, (canvas, canvas))
    mask = cv2.warpAffine(np.ones((new_h, new_w), dtype=np.uint8) * 255,
                          M, (canvas, canvas))
    scene = bg.copy()
    scene[mask > 0] = warped[mask > 0]

    if random.random() < 0.5:
        scene = scene.astype(np.float32)
        scene *= random.uniform(0.7, 1.3)
        scene += random.uniform(-20, 20)
        scene = np.clip(scene, 0, 255).astype(np.uint8)

    corners = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]],
                       dtype=np.float32)
    corners_warped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    jitter = random.uniform(0, 0.04) * max(new_w, new_h)
    jittered = corners_warped + np.random.uniform(
        -jitter, jitter, corners_warped.shape).astype(np.float32)

    dst = np.array([[0, 0], [CARD_W-1, 0], [CARD_W-1, CARD_H-1], [0, CARD_H-1]],
                   dtype=np.float32)
    try:
        M_rect = cv2.getPerspectiveTransform(jittered.astype(np.float32), dst)
        return cv2.warpPerspective(scene, M_rect, (CARD_W, CARD_H))
    except cv2.error:
        return cv2.resize(card_img, (CARD_W, CARD_H))


# ===========================================================================
# Frame classifier

class FrameClassifier(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class FrameDataset(Dataset):
    def __init__(self, cards, indices, frame_classes_by_id, training: bool = True):
        self.cards = cards
        self.indices = indices
        self.training = training
        self.labels = [
            FRAME_CLASS_TO_IDX[frame_classes_by_id[cards[i]['id']]]
            for i in indices
        ]
        self.tf_train = T.Compose([
            T.Resize((FRAME_INPUT, FRAME_INPUT)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = T.Compose([
            T.Resize((FRAME_INPUT, FRAME_INPUT)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = load_card_image(self.cards[idx]['id'])
        if img is None:
            img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        if self.training and random.random() < 0.3:
            img = synthesize_rectified_card(img)
        pil = Image.fromarray(img)
        tf = self.tf_train if self.training else self.tf_eval
        return tf(pil), self.labels[i]


def train_frame_classifier(cards, available, frame_classes_by_id, epochs: int):
    print("\n" + "=" * 70)
    print("Training frame classifier")
    print("=" * 70)

    counts = report_frame_distribution(
        {cards[i]['id']: frame_classes_by_id[cards[i]['id']] for i in available},
        label="Class distribution",
    )

    # Refuse to train if classes are too imbalanced - it's pointless
    n_present = sum(1 for c in counts.values() if c > 0)
    if n_present < 2:
        print(f"\n  ABORT: only {n_present} class(es) present in the data.")
        print("  Frame classification can't learn anything from one class.")
        print("  Run --diagnose to see what fields your metadata has.")
        print("  Re-download metadata with layout/full_art/border_color fields,")
        print("  OR re-run training with image-fallback enabled (default).")
        return

    minority = min(c for c in counts.values() if c > 0)
    if minority < 50:
        print(f"\n  WARNING: minority class has only {minority} examples.")
        print("  Classifier will likely overfit. Consider gathering more or")
        print("  deferring this component until you have richer metadata.")

    # Stratified split - keep all classes represented in val
    by_class = defaultdict(list)
    for i in available:
        by_class[frame_classes_by_id[cards[i]['id']]].append(i)
    tr_idx, va_idx = [], []
    for cls, ix in by_class.items():
        random.shuffle(ix)
        s = int(0.85 * len(ix))
        tr_idx.extend(ix[:s])
        va_idx.extend(ix[s:])
    random.shuffle(tr_idx)

    tr_set = FrameDataset(cards, tr_idx, frame_classes_by_id, training=True)
    va_set = FrameDataset(cards, va_idx, frame_classes_by_id, training=False)

    # Weighted sampler to balance classes during training
    label_counts = Counter(tr_set.labels)
    class_weights = {k: 1.0 / max(1, v) for k, v in label_counts.items()}
    sample_weights = [class_weights[lab] for lab in tr_set.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = FrameClassifier(n_classes=len(FRAME_CLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e3:.1f}K")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for x, y in tqdm(tr_loader, desc=f"  ep{epoch+1} train", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optim.step()
            tr_loss += loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        per_class = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                for p, t in zip(pred.tolist(), y.tolist()):
                    per_class[t][1] += 1
                    if p == t:
                        per_class[t][0] += 1
        acc = correct / total
        per_class_acc = {FRAME_CLASSES[k]: v[0] / max(1, v[1])
                         for k, v in per_class.items()}
        print(f"  ep{epoch+1}: tr_loss={tr_loss/len(tr_loader):.3f} "
              f"val_acc={acc:.3f} per_class={per_class_acc}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), FRAME_CLF_PATH)
            print(f"    saved best (acc={acc:.3f})")

    print(f"  Best val acc: {best_acc:.3f}")


# ===========================================================================
# MobileNet re-ranker (shared between art and whole-card)

class MobileNetReranker(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.projection(x), p=2, dim=1)


def triplet_loss(a, p, n, margin: float = 0.3):
    return F.relu(F.pairwise_distance(a, p) - F.pairwise_distance(a, n) + margin).mean()


class TripletRegionDataset(Dataset):
    def __init__(self, cards, indices, region_extractor, input_size: int,
                 p_synthetic: float = 0.5, training: bool = True):
        self.cards = cards
        self.region_extractor = region_extractor
        self.input_size = input_size
        self.training = training
        self.p_synthetic = p_synthetic if training else 0.0

        self.name_to_indices = defaultdict(list)
        self.valid = []
        for i in indices:
            self.name_to_indices[cards[i]['name']].append(i)
            self.valid.append(i)
        self.triplet_names = [n for n, ix in self.name_to_indices.items() if len(ix) >= 2]
        print(f"    {len(self.valid)} cards, {len(self.triplet_names)} names with 2+ printings")

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_tf = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            T.ToTensor(), self.norm,
            T.RandomErasing(p=0.2),
        ])
        self.eval_tf = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(), self.norm,
        ])
        self.synth_tf = T.Compose([T.ToTensor(), self.norm])

    def __len__(self):
        return len(self.valid)

    def _augment(self, idx):
        card_img = load_card_image(self.cards[idx]['id'])
        if card_img is None:
            card_img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        if self.training and random.random() < self.p_synthetic:
            rectified = synthesize_rectified_card(card_img)
            region = self.region_extractor(rectified)
            region = cv2.resize(region, (self.input_size, self.input_size))
            return self.synth_tf(region)
        region = self.region_extractor(card_img)
        pil = Image.fromarray(region)
        return self.train_tf(pil) if self.training else self.eval_tf(pil)

    def __getitem__(self, i):
        anchor = self.valid[i]
        name = self.cards[anchor]['name']
        pos_pool = [j for j in self.name_to_indices[name] if j != anchor]
        pos = random.choice(pos_pool) if pos_pool else anchor
        neg_name = random.choice([n for n in self.triplet_names if n != name])
        neg = random.choice(self.name_to_indices[neg_name])
        return self._augment(anchor), self._augment(pos), self._augment(neg)


def train_reranker(cards, indices, output_path: Path,
                   region_extractor, input_size: int,
                   label: str = "", epochs: int = DEFAULT_EPOCHS):
    """Train a re-ranker on a pre-filtered list of card indices."""
    print("\n" + "=" * 70)
    print(f"Training {label} re-ranker on {len(indices)} cards")
    print("=" * 70)

    if len(indices) < 100:
        print(f"  Too few cards ({len(indices)}). Skipping.")
        return

    shuffled = indices.copy()
    random.shuffle(shuffled)
    split = int(0.85 * len(shuffled))
    tr_idx, va_idx = shuffled[:split], shuffled[split:]

    print("  Building datasets:")
    tr_set = TripletRegionDataset(cards, tr_idx, region_extractor,
                                   input_size, p_synthetic=0.5, training=True)
    va_set = TripletRegionDataset(cards, va_idx, region_extractor,
                                   input_size, p_synthetic=0.0, training=False)
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best = float('inf')
    for ep in range(epochs):
        model.train()
        tot = 0
        for a, p, n in tqdm(tr_loader, desc=f"  ep{ep+1} tr", leave=False):
            a, p, n = a.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), n.to(DEVICE, non_blocking=True)
            optim.zero_grad()
            loss = triplet_loss(model(a), model(p), model(n))
            loss.backward()
            optim.step()
            tot += loss.item()

        model.eval()
        vtot = 0
        with torch.no_grad():
            for a, p, n in va_loader:
                a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                vtot += triplet_loss(model(a), model(p), model(n)).item()
        sched.step()

        tr_l = tot / len(tr_loader); va_l = vtot / len(va_loader)
        print(f"  ep{ep+1}: train={tr_l:.4f} val={va_l:.4f}")
        if va_l < best:
            best = va_l
            torch.save(model.state_dict(), output_path)
            print(f"    saved best (val={va_l:.4f})")

    print(f"  Best val: {best:.4f}")


def build_embeddings(cards, indices, model_path: Path,
                     region_extractor, input_size: int,
                     output_path: Path, label: str = ""):
    print(f"\nBuilding {label} embeddings -> {output_path.name}")
    if not model_path.exists():
        print(f"  Model {model_path} not found. Skipping embeddings.")
        return None

    model = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    eval_tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"  embed {label}"):
            img = load_card_image(cards[idx]['id'])
            if img is None:
                continue
            region = region_extractor(img)
            pil = Image.fromarray(region)
            t = eval_tf(pil).unsqueeze(0).to(DEVICE)
            embs.append(model(t).cpu().numpy())
    arr = np.vstack(embs).astype('float32')
    np.save(output_path, arr)
    print(f"  Saved: {arr.shape}")
    return arr


# ===========================================================================
# Set symbol classifier

class SetSymbolClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SetSymbolDataset(Dataset):
    def __init__(self, cards, indices, set_to_idx, training: bool = True):
        self.cards = cards
        self.indices = indices
        self.set_to_idx = set_to_idx
        self.training = training
        self.labels = [set_to_idx[cards[i]['set']] for i in indices]
        self.tf_train = T.Compose([
            T.Resize((SETSYM_INPUT, SETSYM_INPUT)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = T.Compose([
            T.Resize((SETSYM_INPUT, SETSYM_INPUT)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = load_card_image(self.cards[idx]['id'])
        if img is None:
            img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        if self.training and random.random() < 0.3:
            img = synthesize_rectified_card(img)
        sym = crop_region(img, 'set_symbol')
        pil = Image.fromarray(sym)
        return (self.tf_train(pil) if self.training else self.tf_eval(pil),
                self.labels[i])


def train_set_classifier(cards, modern_indices, epochs: int):
    print("\n" + "=" * 70)
    print("Training set symbol classifier")
    print("=" * 70)
    print(f"  Modern-frame cards available: {len(modern_indices)}")

    if len(modern_indices) < 1000:
        print(f"  Too few modern cards ({len(modern_indices)}). Skipping set classifier.")
        return

    set_counts = Counter(cards[i].get('set', '<unknown>') for i in modern_indices)
    valid_sets = {s for s, c in set_counts.items() if c >= 20 and s != '<unknown>'}
    keep = [i for i in modern_indices if cards[i].get('set') in valid_sets]
    print(f"  After dropping rare sets (<20 cards): {len(keep)} cards across {len(valid_sets)} sets")

    if len(valid_sets) < 5:
        print(f"  Only {len(valid_sets)} usable sets. Skipping.")
        return

    set_codes = sorted(valid_sets)
    set_to_idx = {s: i for i, s in enumerate(set_codes)}
    with open(SET_CODES_PATH, 'w') as f:
        json.dump(set_codes, f)

    shuffled = keep.copy()
    random.shuffle(shuffled)
    split = int(0.85 * len(shuffled))
    tr_idx, va_idx = shuffled[:split], shuffled[split:]

    tr_set = SetSymbolDataset(cards, tr_idx, set_to_idx, training=True)
    va_set = SetSymbolDataset(cards, va_idx, set_to_idx, training=False)

    label_counts = Counter(tr_set.labels)
    class_weights = {k: 1.0 / max(1, v) for k, v in label_counts.items()}
    sample_weights = [class_weights[lab] for lab in tr_set.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = SetSymbolClassifier(n_classes=len(set_codes)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e3:.0f}K, classes: {len(set_codes)}")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        model.train()
        tot = 0
        for x, y in tqdm(tr_loader, desc=f"  ep{ep+1} tr", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optim.step()
            tot += loss.item()
        sched.step()

        model.eval()
        c1 = c5 = total = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                c1 += (logits.argmax(1) == y).sum().item()
                top5 = logits.topk(5, dim=1).indices
                c5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
                total += y.size(0)
        a1 = c1 / total; a5 = c5 / total
        print(f"  ep{ep+1}: tr_loss={tot/len(tr_loader):.3f} top1={a1:.3f} top5={a5:.3f}")
        if a1 > best:
            best = a1
            torch.save(model.state_dict(), SET_CLF_PATH)
            print(f"    saved best (top1={a1:.3f})")
    print(f"  Best top-1: {best:.3f}")


# ===========================================================================
# Hash database

def build_hash_db(cards, available, frame_classes_by_id):
    print("\n" + "=" * 70)
    print("Building hash database (art + whole-card)")
    print("=" * 70)

    indices, art_p, art_d, whole_p, whole_d = [], [], [], [], []
    frame_class_arr = []

    for idx in tqdm(available, desc="  hashing"):
        img = load_card_image(cards[idx]['id'])
        if img is None:
            continue
        art = extract_art_crop(img)
        whole = extract_whole_card(img)
        indices.append(idx)
        art_p.append(phash_64(art));     art_d.append(dhash_64(art))
        whole_p.append(phash_64(whole)); whole_d.append(dhash_64(whole))
        cls = frame_classes_by_id.get(cards[idx]['id'], 'modern')
        frame_class_arr.append(FRAME_CLASS_TO_IDX[cls])

    np.savez(
        HASH_DB_PATH,
        indices=np.array(indices, dtype=np.int32),
        art_phash=np.array(art_p, dtype=np.uint64),
        art_dhash=np.array(art_d, dtype=np.uint64),
        whole_phash=np.array(whole_p, dtype=np.uint64),
        whole_dhash=np.array(whole_d, dtype=np.uint64),
        frame_class=np.array(frame_class_arr, dtype=np.int8),
    )
    print(f"  Saved {len(indices)} entries to {HASH_DB_PATH.name}")


# ===========================================================================
# Sanity check

def sanity_check(cards, available, n: int = 30):
    print("\n" + "=" * 70)
    print("Sanity check")
    print("=" * 70)

    if not (HASH_DB_PATH.exists() and ART_EMB_PATH.exists() and ART_RERANKER_PATH.exists()):
        print("  Missing artifacts, skipping.")
        return

    db = np.load(HASH_DB_PATH)
    db_indices = db['indices']
    art_emb = np.load(ART_EMB_PATH)

    if len(art_emb) != len(db_indices):
        print(f"  Embedding count {len(art_emb)} != hash DB count {len(db_indices)}")
        print("  Skipping sanity check.")
        return

    reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    reranker.load_state_dict(torch.load(ART_RERANKER_PATH, map_location=DEVICE))
    reranker.eval()

    eval_tf = T.Compose([
        T.Resize((ART_INPUT, ART_INPUT)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ix = random.sample(available, min(n, len(available)))
    c1 = c5 = 0
    for idx in test_ix:
        card = cards[idx]
        img = load_card_image(card['id'])
        if img is None:
            continue
        art = extract_art_crop(img)
        q_p = phash_64(art); q_d = dhash_64(art)
        d_p = hamming_distance_vectorized(q_p, db['art_phash'])
        d_d = hamming_distance_vectorized(q_d, db['art_dhash'])
        combined = d_p + d_d
        cand = np.argsort(combined)[:50]

        with torch.no_grad():
            t = eval_tf(Image.fromarray(art)).unsqueeze(0).to(DEVICE)
            q_emb = reranker(t).cpu().numpy()
        sims = (q_emb @ art_emb[cand].T).flatten()
        order = cand[np.argsort(-sims)]
        top5 = [cards[int(db_indices[o])]['name'] for o in order[:5]]
        if top5 and top5[0] == card['name']:
            c1 += 1
        if card['name'] in top5:
            c5 += 1

    n_actual = len(test_ix)
    print(f"  On {n_actual} cards: Top-1 {c1/n_actual*100:.1f}%  Top-5 {c5/n_actual*100:.1f}%")


# ===========================================================================
# Main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--diagnose', action='store_true',
                   help="Inspect metadata fields and frame distribution, then exit")
    p.add_argument('--skip', nargs='+', default=[],
                   choices=['frame', 'art', 'whole', 'set'],
                   help="Skip training of specific components")
    p.add_argument('--skip-train', action='store_true',
                   help="Don't train anything; rebuild hash DB and embeddings only")
    p.add_argument('--no-image-fallback', action='store_true',
                   help="Don't peek at card images for frame classification")
    p.add_argument('--rebuild-frame-cache', action='store_true',
                   help="Recompute frame classes from scratch (slow if image fallback on)")
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--no-sanity', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 70)
    print("MTG Identifier - Multi-Region Trainer")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------------
    # Diagnose mode: tell the user what's in their metadata, then exit
    if args.diagnose:
        print("\n" + "=" * 70)
        print("Metadata diagnosis")
        print("=" * 70)
        report = inspect_metadata(str(METADATA_PATH))
        print(f"Total cards in metadata: {report['total_cards']:,}")
        print(f"Sampled: {report['sampled']}")
        print(f"\nFields present (count of cards in sample with each field):")
        for k in ['has_layout', 'has_full_art', 'has_border_color',
                  'has_frame_effects', 'has_frame']:
            field = k.replace('has_', '')
            present = report[k]
            pct = present / report['sampled'] * 100
            status = 'OK' if pct > 80 else 'PARTIAL' if pct > 0 else 'MISSING'
            print(f"  {field:<18} {present:>4}/{report['sampled']} ({pct:5.1f}%)  [{status}]")
        print(f"\nlayout values:        {report['layout_values']}")
        print(f"border_color values:  {report['border_color_values']}")
        print(f"full_art values:      {report['full_art_values']}")

        # Test classification on a sample
        cards, available = load_cards()
        sample = random.sample(available, min(200, len(available)))
        print(f"\nClassifying {len(sample)} sample cards (metadata only)...")
        meta_only = Counter(classify_frame_from_metadata(cards[i]) for i in sample)
        print(f"  Result: {dict(meta_only)}")

        print(f"\nClassifying same sample (with image fallback)...")
        with_image = Counter(
            classify_frame_from_metadata(
                cards[i], image_path=str(IMAGE_DIR / f"{cards[i]['id']}.jpg")
            ) for i in sample
        )
        print(f"  Result: {dict(with_image)}")
        if meta_only.get('modern', 0) == len(sample):
            print("\n  -> Metadata alone classifies everything as modern.")
            print("  -> Image fallback is needed - keep --no-image-fallback OFF.")
        if with_image.get('modern', 0) == len(sample):
            print("\n  -> Even with image fallback, everything is modern.")
            print("  -> Either your sample is all modern frame, or the heuristic")
            print("     thresholds need tuning. Try a larger sample.")
        return

    # ----------------------------------------------------------------------
    # Normal training flow
    cards, available = load_cards()
    print(f"\nDataset: {len(cards)} cards, {len(available)} on disk")

    # Step 0: precompute frame classes for every card (cached on disk)
    print("\n[0/5] Frame classification (precompute)")
    frame_classes_by_id = compute_frame_classes(
        cards, available,
        use_image_fallback=not args.no_image_fallback,
        force_recompute=args.rebuild_frame_cache,
    )
    report_frame_distribution(frame_classes_by_id, label="Final distribution")

    # Build the indices we need by frame class
    modern_indices = [i for i in available
                      if frame_classes_by_id[cards[i]['id']] == 'modern']
    fullart_indices = [i for i in available
                       if frame_classes_by_id[cards[i]['id']] == 'fullart']
    special_indices = [i for i in available
                       if frame_classes_by_id[cards[i]['id']] == 'special']

    print(f"\n  -> modern:  {len(modern_indices):,} (used for art re-ranker + set classifier)")
    print(f"  -> fullart: {len(fullart_indices):,} (whole-card path only)")
    print(f"  -> special: {len(special_indices):,} (whole-card path only)")

    skip = set(args.skip)
    if args.skip_train:
        skip = {'frame', 'art', 'whole', 'set'}

    # Step 1: frame classifier
    if 'frame' not in skip:
        train_frame_classifier(cards, available, frame_classes_by_id, args.epochs)
    else:
        print("\nSkipping frame classifier")

    # Step 2: art re-ranker (modern only)
    if 'art' not in skip:
        if len(modern_indices) >= 100:
            train_reranker(
                cards, modern_indices,
                output_path=ART_RERANKER_PATH,
                region_extractor=extract_art_crop,
                input_size=ART_INPUT,
                label="art-crop",
                epochs=args.epochs,
            )
        else:
            print(f"\nSkipping art re-ranker (only {len(modern_indices)} modern cards)")
    else:
        print("\nSkipping art re-ranker")

    # Step 3: whole-card re-ranker (all cards)
    if 'whole' not in skip:
        train_reranker(
            cards, available,
            output_path=WHOLE_RERANKER_PATH,
            region_extractor=extract_whole_card,
            input_size=WHOLE_INPUT,
            label="whole-card",
            epochs=args.epochs,
        )
    else:
        print("\nSkipping whole-card re-ranker")

    # Step 4: set symbol classifier (modern only)
    if 'set' not in skip:
        train_set_classifier(cards, modern_indices, args.epochs)
    else:
        print("\nSkipping set symbol classifier")

    # Step 5: hash DB + embeddings (always rebuilt)
    build_hash_db(cards, available, frame_classes_by_id)
    if ART_RERANKER_PATH.exists():
        build_embeddings(cards, available, ART_RERANKER_PATH,
                         extract_art_crop, ART_INPUT,
                         ART_EMB_PATH, label="art")
    if WHOLE_RERANKER_PATH.exists():
        build_embeddings(cards, available, WHOLE_RERANKER_PATH,
                         extract_whole_card, WHOLE_INPUT,
                         WHOLE_EMB_PATH, label="whole-card")

    if not args.no_sanity:
        sanity_check(cards, available, n=30)

    print("\n" + "=" * 70)
    print("Identifier training complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()