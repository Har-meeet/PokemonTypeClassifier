"""Build the three split protocols from the fixed 14-type image pool."""
import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)
HASH_DISTANCE = 4
# Equal-size swaps within species fill the three missing source-style cells.
IMAGE_LEVEL_STYLE_SWAPS = {
    "104-5": ("train", "test"),
    "104-17": ("test", "train"),
    "104-13": ("train", "validation"),
    "104-16": ("validation", "train"),
    "24-10": ("train", "test"),
    "24-17": ("test", "train"),
}


def difference_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        bounds = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        cropped = rgb.crop(bounds) if bounds else rgb
        pixels = list(cropped.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    return sum(
        (pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
        for y in range(8) for x in range(8)
    )


def near_duplicate_groups(rows):
    hashes = [difference_hash(ROOT / row["image_path"]) for row in rows]
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for i, j in combinations(range(len(rows)), 2):
        if bin(hashes[i] ^ hashes[j]).count("1") <= HASH_DISTANCE:
            parents[find(j)] = find(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def image_level_rows(rows):
    by_species = defaultdict(list)
    for row in rows:
        by_species[row["species_id"]].append(row)
    rng = random.Random(42)
    for species_id, images in sorted(by_species.items(), key=lambda item: int(item[0])):
        groups = near_duplicate_groups(images)
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        holdouts = ["validation", "test"]
        rng.shuffle(holdouts)
        required = ["train", *holdouts]
        totals = dict.fromkeys(SPLITS, 0)
        targets = dict(zip(SPLITS, [ratio * len(images) for ratio in RATIOS]))
        for index, group in enumerate(groups):
            if index < 3:
                split = required[index]
            else:
                split = max(SPLITS, key=lambda name: targets[name] - totals[name])
            totals[split] += len(group)
            group_id = f"{species_id}-{index + 1}"
            assigned_split = split
            if group_id in IMAGE_LEVEL_STYLE_SWAPS:
                expected_split, assigned_split = IMAGE_LEVEL_STYLE_SWAPS[group_id]
                if split != expected_split or len(group) != 1:
                    raise ValueError(f"Unexpected assignment for style swap {group_id}")
            for row in group:
                yield dict(row, split=assigned_split, similarity_group=group_id)


def main():
    source = ROOT / "pokemon_type_dataset_focused/splits/monotype_14"
    chains = json.loads((HERE / "evolution_chains.json").read_text())
    rows = []
    for split in SPLITS:
        with (source / f"{split}.csv").open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    for row in rows:
        row["evolution_chain_id"] = chains["species"][row["species_id"]]
    variants = {
        "species_disjoint": rows,
        "evolution_disjoint": [
            dict(row, split=chains["family_splits"][str(row["evolution_chain_id"])])
            for row in rows
        ],
        "image_level": list(image_level_rows(rows)),
    }
    for name, manifest in variants.items():
        target = HERE / f"{name}.csv"
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(manifest)


if __name__ == "__main__":
    main()
