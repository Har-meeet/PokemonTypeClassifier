# Pokémon Type Classifier

## 14-type training dataset

Use the [fixed split manifests](pokemon_type_dataset_focused/splits/monotype_14/) with the processed images included in this repository.

| Manifest | Species | Images |
| --- | ---: | ---: |
| [Training](pokemon_type_dataset_focused/splits/monotype_14/train.csv) | 326 | 4,242 |
| [Validation](pokemon_type_dataset_focused/splits/monotype_14/validation.csv) | 67 | 927 |
| [Test](pokemon_type_dataset_focused/splits/monotype_14/test.csv) | 67 | 845 |

Resolve each `image_path` relative to the repository root. The path already includes `pokemon_type_dataset_focused/`. The images are 96×96 RGB PNGs. Use `label_id` as the integer target, with the same [class mapping](pokemon_type_dataset_focused/splits/monotype_14/class_to_idx.json) for every split.

The manifests reuse the seed-42 species assignments and exclude Dark, Dragon, Flying, and Steel. Each species belongs to one split. Evolutionary relatives can appear in different splits. The manifests contain original processed images, and each row includes its PokéAPI source URL.

See [dataset protocols](dataset_protocols/README.md) for the evolution-disjoint, species-disjoint, and image-level variants.
