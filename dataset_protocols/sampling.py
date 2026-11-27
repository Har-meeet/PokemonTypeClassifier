"""Family-balanced training draws and species-weighted evaluation."""
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageCms, ImageOps


def training_draws(rows, epoch, draws_per_family=12, seed=42):
    """Return (row, augmentation seed) pairs for one training epoch."""
    if draws_per_family < 1:
        raise ValueError("draws_per_family must be positive")
    rng = random.Random(f"{seed}:{epoch}")
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] == "train":
            families[row["evolution_chain_id"]][row["species_id"]].append(row)

    draws = []
    for family, members in sorted(families.items()):
        species = sorted(members)
        random.Random(f"{seed}:{family}").shuffle(species)
        offset = (epoch * draws_per_family) % len(species)
        queues = {}
        for species_id in species:
            styles = defaultdict(list)
            for row in members[species_id]:
                styles[row["image_style"]].append(row)
            for images in styles.values():
                rng.shuffle(images)
            queues[species_id] = list(styles.values())
            rng.shuffle(queues[species_id])

        for index in range(draws_per_family):
            species_id = species[(offset + index) % len(species)]
            styles = queues[species_id]
            if styles:
                images = styles.pop(0)
                row = images.pop()
                if images:
                    styles.append(images)
                augmentation_seed = None
            else:
                row = rng.choice(members[species_id])
                augmentation_seed = rng.getrandbits(64)
            draws.append((row, augmentation_seed))
    rng.shuffle(draws)
    return draws


@lru_cache(maxsize=1)
def _lab_transforms():
    rgb = ImageCms.createProfile("sRGB")
    lab = ImageCms.createProfile("LAB")
    return (
        ImageCms.buildTransform(rgb, lab, "RGB", "LAB"),
        ImageCms.buildTransform(lab, rgb, "LAB", "RGB"),
    )


def _augment_repeat(image, seed):
    rng = random.Random(seed)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    offsets = (*range(-20, -9), *range(10, 21))
    delta_a, delta_b = rng.choice(offsets), rng.choice(offsets)

    to_lab, to_rgb = _lab_transforms()
    lab_pixels = bytearray(ImageCms.applyTransform(image, to_lab).tobytes())
    foreground = bytearray()
    for index, rgb in enumerate(image.getdata()):
        is_foreground = rgb != (255, 255, 255)
        foreground.append(255 if is_foreground else 0)
        if is_foreground:
            a = 3 * index + 1
            b = a + 1
            lab_pixels[a] = max(0, min(255, lab_pixels[a] + delta_a))
            lab_pixels[b] = max(0, min(255, lab_pixels[b] + delta_b))

    shifted_lab = Image.frombytes("LAB", image.size, bytes(lab_pixels))
    shifted_rgb = ImageCms.applyTransform(shifted_lab, to_rgb)
    mask = Image.frombytes("L", image.size, bytes(foreground))
    return Image.composite(shifted_rgb, image, mask)


def load_draw(draw, repo_root=None):
    """Load an RGB image and label, with optional repeat-draw augmentation."""
    row, augmentation_seed = draw
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    with Image.open(root / row["image_path"]) as source:
        image = source.convert("RGB")
    if augmentation_seed is not None:
        image = _augment_repeat(image, augmentation_seed)
    return image, int(row["label_id"])


def species_weights(rows):
    """Return per-image weights so every species contributes one unit."""
    counts = Counter(row["species_id"] for row in rows)
    return [1 / counts[row["species_id"]] for row in rows]
