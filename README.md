# MTG Card Identifier: Deep Learning Capstone

A deep learning system for identifying Magic: The Gathering cards from real-world phone photos. It handles rotation, perspective warp, color shifts, glare, blur, and partial occlusion.

---

## Problem

MTG has over 90,000 unique card printings. Players, collectors, and traders need to identify cards quickly during gameplay or trading, but manual lookup is slow and existing apps only work with clean, flat scans. This project builds a model that holds up on messy, hand-held phone shots.

A few things make this genuinely difficult:
- 90k+ classes with only one official image per card
- Many cards share the same name but have different art across printings
- Real photos introduce rotation, perspective distortion, lighting variation, motion blur, glare, and occlusion

---

## Notebook

### `mtg_card_identifier.ipynb`
| Item | Detail |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Loss | Triplet loss with random negatives |
| Search | Brute-force cosine similarity |
| Top-1 accuracy | ~89.7% |
| Top-5 accuracy | ~94.8% |
| Query speed | ~50 ms |

Sets up a working baseline using standard transfer learning. One key finding: pretrained ImageNet features alone score 0%, so task-specific fine-tuning is required.

### Augmentation Pipeline

Each transform targets a specific real-world photo condition:

| Transform | Simulates |
|---|---|
| `RandomResizedCrop` (70-100%) | Variable zoom / framing |
| `RandomHorizontalFlip` / `RandomVerticalFlip` | Any card orientation |
| `RandomRotation` (+-35 degrees) | Tilted hand-held shots |
| `RandomPerspective` (45%) | Off-axis camera angle |
| `ColorJitter` | Varied lighting colour temperature |
| `RandomAutocontrast` | Phone auto-exposure correction |
| `RandomEqualize` | Uneven / harsh lighting |
| `RandomSolarize` | Foil glare / flash reflection |
| `GaussianBlur` | Motion blur / low shutter speed |
| `GaussianNoise` | Camera sensor noise in low light |
| `RandomAdjustSharpness` | Out-of-focus or over-sharpened shots |
| `RandomGrayscale` | Desaturated / B&W phone modes |
| `RandomErasing` x 2 | Fingers, card sleeves, partial occlusion |

---

## How It Works

```
Phone photo
    |
    v
simulate_photo()          <- rotation, perspective, jitter, blur, noise
    |
    v
EfficientNet-B4 + projection head  <- produces L2-normalised embedding
    |
    v
Cosine similarity search  <- nearest-neighbour over all card embeddings
    |
    v
Top-K ranked predictions  <- card name, set, similarity score
```

Training uses a `TripletCardDataset`, which builds *(anchor, positive, negative)* triplets each batch:
- **Anchor and Positive** are the same card image run through two different random augmentations
- **Negative** is a randomly selected different card

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy matplotlib pillow tqdm scikit-learn requests
```

### 2. Run the notebook

```bash
jupyter notebook mtg_card_identifier.ipynb
```

Run cells top-to-bottom. The first run downloads roughly 13 GB of card images from the [Scryfall API](https://scryfall.com); after that it uses the local cache.

### 3. Identify a card interactively

In the final cells of `mtg_card_identifier.ipynb`, set:

```python
CARD_NAME_TO_TEST = "Lightning Bolt"
```

The notebook applies random photo-simulation transforms, runs the trained model, and displays the top-5 predictions with similarity scores.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Scryfall API](https://api.scryfall.com/bulk-data) |
| Cards | ~91,000 unique printings |
| Excluded | Digital-only, tokens, art series, "Un-set" joke cards |
| Image size | 488 x 680 px (official scans) |
| Disk space | ~13 GB |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM | 16 GB VRAM |
| CUDA | 11.8 | 13.0 |
| RAM | 16 GB | 32 GB |
| Storage | 15 GB | 20 GB |

Training takes around 2 hours on an RTX 5070 Ti (2 epochs, 91k cards). Reduce `NUM_EPOCHS` or `max_cards` in the config cells to experiment faster.

---

## Results

| Metric | Value |
|---|---|
| Top-1 accuracy | 89.7% |
| Top-5 accuracy | 94.8% |
| Query speed | ~50 ms |

---

## Key Takeaways

1. Pretrained features alone do not work. ImageNet features score 0%; the model needs fine-tuning on card data.
2. Augmentation variety is critical. Stacking 13 transforms closes the gap between clean scans and real phone photos.
3. The open-set design means new cards can be added without retraining. Only their embeddings need to be computed.

---

## Acknowledgements

- [Scryfall](https://scryfall.com) for card data and images
- [PyTorch / torchvision](https://pytorch.org) for EfficientNet pretrained weights
