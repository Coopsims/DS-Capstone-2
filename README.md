# Fast MTG Card Identifier

A multi-stage pipeline that identifies Magic: The Gathering cards from
photos. Designed to match or beat Claude's vision accuracy at 30-50× the
speed by combining cheap classical signals (perceptual hashing, OCR) with
small specialist neural networks (~3M params each).

The pipeline understands that not all MTG cards have the same layout —
modern bordered cards get multi-region extraction (art crop, title, set
symbol), while full-art lands and Showcase frames go down a whole-card
fallback path.

## Pipeline architecture

```
Photo
  ↓
[YOLO detector]  → rectified 488×680 card
  ↓
[Frame classifier]  → modern / fullart / special
  ↓
  ├─ modern path:  art crop → pHash retrieval → MobileNet re-rank → set symbol → name
  └─ fallback:     whole card → pHash retrieval → MobileNet re-rank → name
```

Five trained models, each tiny:

| Component | Size | Purpose |
|-----------|------|---------|
| YOLO detector (fine-tuned) | 2.7M params | Find card in photo, get oriented bounding box |
| Frame classifier | 98K params | Choose extraction strategy from rectified card |
| Art re-ranker | 3M params | Embedding for modern-frame art crops |
| Whole-card re-ranker | 3M params | Universal fallback embedding |
| Set symbol classifier | 500K params | Disambiguate reprints by set |

## File layout

```
project/
├── mtg_layout.py              # shared constants & helpers (DO NOT run)
├── train_detector.py          # train the YOLO card detector
├── train_identifier.py        # train everything else
├── identify_card.py           # interactive identifier (run this to use it)
├── benchmark_simulated.py     # benchmark vs Claude on synthetic photos
├── benchmark_sanity_check.py  # diagnostic for the benchmark
└── mtg_data/
    ├── cards_metadata.json        # Scryfall metadata (input)
    ├── card_images/{id}.jpg       # card images (input)
    ├── yolo_card_best.pt          # trained detector (output)
    ├── frame_classifier.pth       # trained frame CNN (output)
    ├── art_reranker.pth           # trained art embedding model (output)
    ├── whole_reranker.pth         # trained whole-card embedding model (output)
    ├── set_classifier.pth         # trained set symbol CNN (output)
    ├── phash_db.npz               # hash database (output)
    ├── art_embeddings.npy         # precomputed art embeddings (output)
    ├── whole_embeddings.npy       # precomputed whole-card embeddings (output)
    ├── set_codes.json             # set classifier label mapping (output)
    └── frame_classes_cache.json   # per-card frame class cache (output)
```

## Setup

### Requirements

```
torch >= 2.0
torchvision
opencv-python
numpy >= 2.0  (for fast bit_count; older numpy works but slower)
Pillow
matplotlib
tqdm
ultralytics
easyocr        (optional, for OCR confirmation)
symspellpy     (optional, for fuzzy name matching)
anthropic      (only for benchmarking against Claude)
```

### Metadata

Your `mtg_data/cards_metadata.json` must contain Scryfall records with at
least these fields per card:

- `id` (string, the Scryfall UUID)
- `name` (string)
- `set` (string, set code)
- `layout` (string, e.g. "normal", "split", "saga")
- `full_art` (bool)
- `border_color` (string)

If your metadata is missing the layout-related fields, see "Re-downloading
metadata" at the bottom of this document.

### Card images

`mtg_data/card_images/` should contain one JPG per Scryfall ID. The
trainers will resize to 488×680 if the source isn't already that size, but
clean Scryfall `large` or `normal` images at native resolution work best.

## Training

Two phases. Run them in order.

### Phase 1: train the detector (~2-3 hours)

```bash
python train_detector.py
```

What this does:
1. Generates 5,000 synthetic training scenes + 500 validation scenes by
   compositing your card images onto random backgrounds with rotation,
   lighting jitter, glare, and noise. Saves them to `mtg_data/yolo_card/`
   with YOLO-OBB labels.
2. Fine-tunes `yolo11n-obb.pt` on those scenes for 50 epochs. The
   pretrained checkpoint was trained on aerial imagery — fine-tuning
   teaches it what an MTG card looks like.
3. Copies the best weights to `mtg_data/yolo_card_best.pt`.

Useful flags:

```bash
python train_detector.py --gen-only --show    # just inspect the dataset
python train_detector.py --train-only         # data already generated
python train_detector.py --n-train 10000      # bigger dataset
python train_detector.py --epochs 30          # shorter training
python train_detector.py --imgsz 800          # higher input resolution
```

What to watch:
- `mAP50(OBB)` should reach 0.85+ within 30 epochs. This is the metric
  that matters — bbox-only mAP can be high while rotation is wrong.
- `mtg_data/yolo_card_runs/cards/val_batch0_pred.jpg` is the visual
  sanity check. Open it after training and confirm boxes hug the cards
  at correct angles.

### Phase 2: train the identifier components (~4-6 hours)

```bash
# First run the diagnosis to confirm metadata is healthy
python train_identifier.py --diagnose
```

Look for: 100% present for `layout`, `full_art`, `border_color`, `frame`.
Final classification should produce all three classes (e.g.
`{modern: 168, fullart: 28, special: 4}`). If everything classifies as
one class, see "Re-downloading metadata" below before proceeding.

```bash
python train_identifier.py
```

What this does:
1. Precomputes frame classes for every card (cached to disk).
2. Trains the frame classifier (~10 min).
3. Trains the art re-ranker on modern-frame cards (~1 hour).
4. Trains the whole-card re-ranker on all cards (~1.5 hours).
5. Trains the set symbol classifier on modern cards (~30 min).
6. Builds the hash database with both art and whole-card hashes.
7. Builds embedding databases for both re-rankers.
8. Runs an end-to-end sanity check (Top-1 should be ≥99%).

Useful flags:

```bash
python train_identifier.py --skip set frame   # skip specific components
python train_identifier.py --skip-train       # rebuild DB and embeddings only
python train_identifier.py --epochs 8         # shorter training
python train_identifier.py --rebuild-frame-cache   # discard cached labels
```

If you re-download metadata or change classification logic, always pass
`--rebuild-frame-cache`. Otherwise the trainer reuses old labels.

## Using the identifier

### Interactive mode

```bash
python identify_card.py
```

Opens a file picker. Pick an image, see the prediction, pick another.

### One-off mode

```bash
python identify_card.py path/to/card.jpg
```

Prints the prediction and shows a 2×3 figure: query image, after-detection
crop, art crop, and the top-3 matches from the database.

### Other flags

```bash
python identify_card.py --no-gui         # type paths at stdin (good over SSH)
python identify_card.py --once           # quit after one image
```

### Reading the output

The identifier prints something like:

```
PREDICTION: Lightning Bolt
  Set:         Magic 2010 (M10)
  Collector:   #146
  Type:        Instant
  Mana cost:   {R}
  Confidence:  similarity 0.847
  Route:       rerank
  OCR title:   'Lightning Bolt'
```

`Route` is which path the pipeline took:
- `fast_hash` — pHash alone was decisive, skipped the re-ranker (fastest)
- `rerank` — needed the neural re-ranker to pick between similar candidates

`OCR title` only appears when OCR was triggered (close calls between top
candidates).

## Benchmarking

### Against Claude on synthetic photos

```bash
python benchmark_simulated.py
```

Generates 90 synthetic photos (30 each at easy / medium / hard difficulty)
and runs both the hash pipeline and Claude on them. Prints accuracy and
latency per difficulty, saves results to `mtg_data/benchmark_results.json`
and a comparison plot to `mtg_data/benchmark_comparison.png`.

Costs about $0.45 in Claude API calls.

```bash
python benchmark_simulated.py --no-claude          # hash only
python benchmark_simulated.py --n-per-difficulty 50  # bigger run
python benchmark_simulated.py --reuse-images       # don't regenerate scenes
```

You'll need either `ANTHROPIC_API_KEY` in your environment or in
`mtg_data/.env`, or it'll prompt you.

### Diagnosing failed benchmarks

If benchmark accuracy is unexpectedly low:

```bash
python benchmark_sanity_check.py
```

Picks one image per difficulty from your benchmark and runs it through
the pipeline manually with verbose output. Tells you:
- Whether the hash DB is intact (self-consistency check)
- The Hamming distance from the query to the *correct* card's DB entry
- Where the correct card lives in the candidate list
- What the detector is producing (visualization)

The truth-check distance is the key number:
- **0-5** → hashing path works; the re-ranker is the problem
- **10-30** → art crop is misaligned; detector or color order is the problem
- **30+** → essentially random; detector is mangling the input

## Re-downloading metadata

If `train_identifier.py --diagnose` shows missing layout fields, you need
to re-fetch metadata from Scryfall. Save this as `download_metadata.py`:

```python
import requests, json
from pathlib import Path

# Get the bulk-data manifest to find the current default-cards URL
manifest = requests.get('https://api.scryfall.com/bulk-data').json()
default_cards = next(b for b in manifest['data'] if b['type'] == 'default_cards')
print(f"Downloading {default_cards['size'] / 1e6:.0f} MB...")

resp = requests.get(default_cards['download_uri'])
all_cards = resp.json()
print(f"Got {len(all_cards):,} cards")

# Merge layout fields into existing metadata
with open('mtg_data/cards_metadata.json') as f:
    existing = json.load(f)

by_id = {c['id']: c for c in all_cards}
fields_to_add = ['layout', 'full_art', 'border_color', 'frame_effects',
                 'frame', 'type_line', 'mana_cost', 'set_name',
                 'collector_number']

for card in existing:
    if card['id'] in by_id:
        for field in fields_to_add:
            if field in by_id[card['id']]:
                card[field] = by_id[card['id']][field]

Path('mtg_data/cards_metadata.json').rename('mtg_data/cards_metadata_old.json')
with open('mtg_data/cards_metadata.json', 'w') as f:
    json.dump(existing, f)
print("Done.")
```

Run it once, then re-run `train_identifier.py --diagnose` to confirm
fields are populated, then proceed with training (passing
`--rebuild-frame-cache` to discard any stale frame labels).

## Updating the database when new cards release

When a new MTG set drops or you add more printings:

1. Update `mtg_data/cards_metadata.json` with the new cards
2. Add the new card images to `mtg_data/card_images/`
3. Run:
   ```bash
   python train_identifier.py --skip-train
   ```

This rebuilds the hash database and embeddings to cover the new cards
without retraining any models. Runs in 15-30 minutes depending on how
many cards were added.

You only need a full retrain (without `--skip-train`) when:
- You add 10%+ new cards relative to the existing dataset
- Cards are added in a stylistically novel category (Universes Beyond
  crossovers, experimental Secret Lair drops, new frame revisions)
- The sanity check Top-1 drops below 95% on a held-out sample of new cards

For typical incremental additions, the embedding space generalizes well
enough to skip training entirely.

## Troubleshooting

### `ModuleNotFoundError: No module named 'mtg_layout'`

`mtg_layout.py` must be in the same directory as the script you're running.
The trainers and identifier import from it.

### "0% accuracy" on the benchmark

Run `python benchmark_sanity_check.py`. It diagnoses the four most common
causes (DB out of sync, detector mangling, channel order, re-ranker
failure) and tells you exactly which one is happening.

### Frame classifier achieves 100% accuracy on one class

This means metadata classification is degenerate (all cards labeled as
one class). Run `python train_identifier.py --diagnose` to see which
metadata fields are missing, then re-download metadata as described above.

### Detector finds objects inside the card art instead of the card itself

This happens when using the pretrained `yolo11n-obb.pt` directly without
fine-tuning. The pretrained model was trained on aerial imagery and
doesn't know what a card looks like. Run `train_detector.py` to fine-tune
it on synthetic card scenes.

### EasyOCR or symspellpy import errors

Both are optional. The pipeline runs without them, just without OCR-based
disambiguation on close calls. Install with `pip install easyocr symspellpy`
if you want the extra signal.

## Performance notes

Typical numbers on a 5070 Ti with the full trained pipeline:

- Detection: ~20ms (YOLO inference)
- Hash retrieval: <1ms (vectorized over ~92k entries)
- Re-ranker: ~5ms on GPU, ~30ms on CPU
- OCR (when triggered): ~15ms
- Total: 30-50ms typical, 80ms with OCR

Compare to ~1.5-2.0 seconds per Claude API call. The accuracy gap
narrows as image difficulty increases (Claude wins by more on hard
photos), but for clean scans and decent phone photos the hash pipeline
matches Claude at roughly 30-50× the speed.

The hash database is ~3MB total for 92k cards. Embeddings add ~95MB
(50MB art + 45MB whole-card). The trained models together are ~25MB.
The whole runtime footprint fits comfortably on a phone.