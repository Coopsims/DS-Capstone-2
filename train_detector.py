"""
Train a YOLOv11n-OBB detector to find Magic cards in real-world scenes.

Why this exists:
  The pretrained YOLOv11n-OBB checkpoint was trained on DOTA aerial imagery
  (planes, ships, vehicles). On photos of MTG cards it locks onto strong
  edges inside the card art (P/T boxes, contrast streaks) instead of the
  card's outer rectangle. The result is detector output that warps a
  random sub-region of the card to 488x680 and breaks every downstream
  hash lookup.

  Fine-tuning on ~5000 synthetic scenes - cards composited onto random
  backgrounds with known corner coordinates - is enough to fix this. We
  generate the data, write YOLO-OBB labels, and let Ultralytics handle
  the training.

Inputs:
  mtg_data/cards_metadata.json
  mtg_data/card_images/{id}.jpg

Outputs:
  mtg_data/yolo_card/                       - synthetic dataset (images + labels)
  mtg_data/yolo_card/cards_obb.yaml         - dataset config
  mtg_data/yolo_card_runs/<run>/weights/best.pt
  mtg_data/yolo_card_best.pt                - copy of best weights for inference

Usage:
  python train_detector.py                   # generate + train
  python train_detector.py --gen-only        # just generate data
  python train_detector.py --train-only      # train on existing data
  python train_detector.py --n-train 5000    # change scene count
  python train_detector.py --epochs 50       # change training epochs
  python train_detector.py --imgsz 640       # change input resolution
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'

YOLO_DIR = DATA_DIR / 'yolo_card'
YOLO_CFG = YOLO_DIR / 'cards_obb.yaml'
YOLO_RUNS = DATA_DIR / 'yolo_card_runs'
YOLO_BEST_OUT = DATA_DIR / 'yolo_card_best.pt'

CARD_W = 488
CARD_H = 680


# ---------------------------------------------------------------------------
# Synthetic scene generation
#
# Each training image is a card composited onto a random background at a
# random scale, position, and rotation. We save the actual rotated quad
# coordinates as the YOLO-OBB label.

def random_background(size: int, mode: str = None) -> np.ndarray:
    """Generate a random background. Mix matched to benchmark difficulty."""
    if mode is None:
        mode = random.choice(['solid', 'gradient', 'noise', 'wood', 'cluttered'])

    if mode == 'solid':
        c = tuple(random.randint(0, 255) for _ in range(3))
        return np.full((size, size, 3), c, dtype=np.uint8)

    if mode == 'gradient':
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

    if mode == 'noise':
        bg = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return cv2.GaussianBlur(bg, (21, 21), 0)

    if mode == 'wood':
        base = np.array([random.randint(70, 160), random.randint(50, 130),
                         random.randint(30, 90)], dtype=np.float32)
        var = random.uniform(20, 50)
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        for j in range(size):
            wave = (np.sin(j / random.uniform(8, 25)) * var
                    + np.sin(j / random.uniform(40, 80)) * var * 0.5)
            bg[:, j] = np.clip(base + wave, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(bg, (5, 5), 0)

    # cluttered
    bg = np.random.randint(40, 200, (size, size, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(0, size), random.randint(0, size)
        radius = random.randint(size // 8, size // 3)
        color = tuple(random.randint(20, 230) for _ in range(3))
        cv2.circle(bg, (cx, cy), radius, color, -1)
    return cv2.GaussianBlur(bg, (51, 51), 0)


def add_glare(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    cx = random.randint(int(w*0.2), int(w*0.8))
    cy = random.randint(int(h*0.2), int(h*0.8))
    radius = random.randint(min(h, w) // 6, min(h, w) // 3)
    glare = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glare, (cx, cy), radius, 1.0, -1)
    glare = cv2.GaussianBlur(glare, (51, 51), 0)
    glare = (glare / glare.max()) * 255 * strength
    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + glare, 0, 255)
    return out.astype(np.uint8)


def synthesize_scene(card_img: np.ndarray, output_size: int = 640) -> tuple:
    """
    Composite a card onto a random background. Returns:
      scene: HxWx3 uint8
      corners_normalized: 8 floats in [0,1] - the YOLO-OBB label
                          ordered TL, TR, BR, BL
    """
    H_card, W_card = card_img.shape[:2]
    canvas = output_size
    bg = random_background(canvas)

    # Card scale - want a wide range so the detector handles small AND large cards
    scale = random.uniform(0.25, 0.85)
    new_h = int(canvas * scale)
    new_w = int(new_h * W_card / H_card)
    if new_w > canvas * 0.95:
        new_w = int(canvas * 0.95)
        new_h = int(new_w * H_card / W_card)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    # Card-level lighting jitter
    if random.random() < 0.7:
        card_f = card_resized.astype(np.float32)
        card_f *= random.uniform(0.6, 1.4)
        card_f += random.uniform(-30, 30)
        card_resized = np.clip(card_f, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        card_resized = add_glare(card_resized, strength=random.uniform(0.3, 0.6))

    # Random rotation - up to 60 degrees so the OBB component actually matters
    angle = random.uniform(-60, 60)

    # Position with margin so the rotated card stays mostly on canvas
    margin = int(max(new_w, new_h) * 0.7)
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

    # Mild scene-level blur and noise
    if random.random() < 0.3:
        scene = cv2.GaussianBlur(scene, (random.choice([3, 5, 7]),) * 2, 0)
    if random.random() < 0.4:
        noise = np.random.randint(-8, 9, scene.shape, dtype=np.int16)
        scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Compute the actual rotated corners of the card in scene coordinates.
    # Order: TL, TR, BR, BL of the original (pre-rotation) card.
    corners = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]],
                       dtype=np.float32)
    corners_warped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
    corners_normalized = corners_warped / np.array([canvas, canvas], dtype=np.float32)
    # Clip in case rotation pushes a corner slightly off-canvas
    corners_normalized = np.clip(corners_normalized, 0.0, 1.0)
    return scene, corners_normalized.flatten()


def write_yolo_obb_label(label_path: Path, corners_normalized: np.ndarray,
                          class_id: int = 0):
    """
    YOLO-OBB label format: one line per object,
      class_id x1 y1 x2 y2 x3 y3 x4 y4
    All coords normalized to [0,1].
    """
    parts = [str(class_id)] + [f"{c:.6f}" for c in corners_normalized.tolist()]
    label_path.write_text(" ".join(parts) + "\n")


def generate_dataset(cards: list, available_indices: list,
                      n_train: int, n_val: int, imgsz: int = 640):
    """Generate synthetic train + val splits and write YOLO directory layout."""
    YOLO_DIR.mkdir(exist_ok=True)
    for split in ['train', 'val']:
        (YOLO_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_train} training scenes + {n_val} val scenes at {imgsz}px...")

    for split, n in [('train', n_train), ('val', n_val)]:
        for i in tqdm(range(n), desc=f"  {split}"):
            card_idx = random.choice(available_indices)
            card = cards[card_idx]
            src = IMAGE_DIR / f"{card['id']}.jpg"
            try:
                card_img = np.array(Image.open(src).convert('RGB'))
                if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
                    card_img = cv2.resize(card_img, (CARD_W, CARD_H))
            except Exception:
                continue

            scene, corners = synthesize_scene(card_img, output_size=imgsz)

            stem = f"{i:06d}"
            img_path = YOLO_DIR / split / 'images' / f"{stem}.jpg"
            lbl_path = YOLO_DIR / split / 'labels' / f"{stem}.txt"
            Image.fromarray(scene).save(img_path, quality=88)
            write_yolo_obb_label(lbl_path, corners)

    # Write the dataset config YAML
    config = (
        f"path: {YOLO_DIR.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 1\n"
        "names: ['card']\n"
    )
    YOLO_CFG.write_text(config)
    print(f"Wrote {YOLO_CFG}")


# ---------------------------------------------------------------------------
# Training

def train_detector(epochs: int, imgsz: int, batch: int, device: str):
    """Fine-tune yolo11n-obb.pt on the synthetic card dataset."""
    from ultralytics import YOLO

    model = YOLO('yolo11n-obb.pt')

    print(f"\nTraining for {epochs} epochs at {imgsz}px (batch={batch})...")
    results = model.train(
        data=str(YOLO_CFG),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(YOLO_RUNS),
        name='cards',
        exist_ok=True,
        # OBB-specific knobs that help small datasets
        patience=10,
        cos_lr=True,
        close_mosaic=10,
        # We do our own augmentation, so soften the built-in stuff
        mosaic=0.3,
        mixup=0.0,
        degrees=0.0,    # rotation already in synthetic data
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0001,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
    )

    # Replace the existing best/copy section with:
    run_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
    if run_dir is None:
        # Fallback: search for the most recent best.pt in either location
        candidates = [
            YOLO_RUNS / 'cards' / 'weights' / 'best.pt',
            Path('runs/obb') / YOLO_RUNS / 'cards' / 'weights' / 'best.pt',
        ]
        best = next((p for p in candidates if p.exists()), None)
    else:
        best = run_dir / 'weights' / 'best.pt'

    if best and best.exists():
        shutil.copy(best, YOLO_BEST_OUT)
        print(f"\nBest weights copied to: {YOLO_BEST_OUT}")
    else:
        print(f"\nWARNING: best.pt not found. Check {run_dir or YOLO_RUNS}")

    return results


# ---------------------------------------------------------------------------
# Validation - eyeball the dataset before training

def show_examples(n: int = 6):
    """Render a grid of synthetic scenes with corner overlays as a sanity check."""
    import matplotlib.pyplot as plt
    train_imgs = sorted((YOLO_DIR / 'train' / 'images').glob('*.jpg'))
    if not train_imgs:
        print("No training images found - run with --gen-only first.")
        return
    sample = random.sample(train_imgs, min(n, len(train_imgs)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, img_path in zip(axes.flat, sample):
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]
        lbl = (YOLO_DIR / 'train' / 'labels' / f"{img_path.stem}.txt").read_text().split()
        coords = np.array(lbl[1:], dtype=np.float32).reshape(4, 2)
        coords[:, 0] *= w; coords[:, 1] *= h
        coords = coords.astype(np.int32)
        cv2.polylines(img, [coords], True, (0, 255, 0), 3)
        ax.imshow(img); ax.axis('off')
        ax.set_title(img_path.name)
    plt.tight_layout()
    out = DATA_DIR / 'detector_dataset_preview.png'
    plt.savefig(out, dpi=80, bbox_inches='tight')
    plt.show()
    print(f"Saved preview to {out}")


# ---------------------------------------------------------------------------
# Main

def main():
    p = argparse.ArgumentParser(description="Train YOLO-OBB card detector")
    p.add_argument('--gen-only', action='store_true', help="Generate dataset, skip training")
    p.add_argument('--train-only', action='store_true', help="Train on existing dataset")
    p.add_argument('--n-train', type=int, default=5000)
    p.add_argument('--n-val', type=int, default=500)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--show', action='store_true', help="Show example scenes after generation")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    print("=" * 70)
    print("YOLO Card Detector - Training")
    print("=" * 70)

    if not args.train_only:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            cards = json.load(f)
        available = [i for i, c in enumerate(cards)
                     if (IMAGE_DIR / f"{c['id']}.jpg").exists()]
        print(f"Source cards: {len(available)} on disk")
        generate_dataset(cards, available, args.n_train, args.n_val, imgsz=args.imgsz)
        if args.show:
            show_examples()

    if args.gen_only:
        print("\n--gen-only: stopping after dataset generation.")
        print("Inspect mtg_data/yolo_card/ then re-run with --train-only.")
        return

    train_detector(epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device)

    print("\n" + "=" * 70)
    print("Detector training complete.")
    print(f"Use {YOLO_BEST_OUT} in the identifier instead of yolo11n-obb.pt")
    print("=" * 70)


if __name__ == '__main__':
    main()