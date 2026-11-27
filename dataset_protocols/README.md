# 14-type dataset protocols

All three manifests reference the same 6,014 processed PokéAPI images. Filter the `split` column for `train`, `validation`, or `test`. Resolve `image_path` from the repository root and use `label_id` as the integer target.

| Protocol | Group kept together | Training images | Validation images | Test images |
| --- | --- | ---: | ---: | ---: |
| [Evolution-disjoint](evolution_disjoint.csv) | PokéAPI evolution chain | 4,386 | 783 | 845 |
| [Species-disjoint](species_disjoint.csv) | Species | 4,242 | 927 | 845 |
| [Image-level](image_level.csv) | Groups of similar images within species | 4,122 | 973 | 919 |

The species-disjoint protocol reuses the original split. The evolution-disjoint assignments target 70/15/15 by species while retaining family coverage for every type. Evolution chain IDs come from [PokéAPI species metadata](https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/pokemon_species.csv).

The image-level protocol groups each species' cropped grayscale difference hashes by connecting pairs at Hamming distance four or below. Each connected group goes to one split, and each of the 460 species appears in all three splits. Three equal-size swaps within Cubone and Arbok give each type all three source styles in each split. The hash threshold is a heuristic and can miss visually similar pairs or group distinct views together. Similar images within a group remain available in its assigned split.

## Training and evaluation

For the two grouped protocols, call `training_draws(rows, epoch)` from [sampling.py](sampling.py) once per epoch. Pass the returned draws to `load_draw` in your dataset or batch loader. The default gives each training family 12 draws, distributes those draws as evenly as possible among its species, and cycles through available styles.

A species cycles through its original training images before repeat draws receive Lab colour jitter and an independent 50% chance of a horizontal flip. For each repeat, the sampler independently selects one integer offset per chromatic channel, uniformly from the inclusive bands `[-20, -10]` and `[10, 20]`. It applies the same offset pair to all non-white pixels, keeps L* fixed in Lab, and preserves white pixels in the RGB output. Horizontal flips reorder pixels without resampling. Augmentation runs on demand using [Pillow ImageCms](https://pillow.readthedocs.io/en/stable/reference/ImageCms.html).

The [CIELAB definition](https://cie.co.at/eilv/157) provides the lightness and chromatic coordinates. [Krizhevsky, Sutskever, and Hinton (2012, §4.1)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) provides precedent for horizontal reflections and colour perturbation in image classification and uses RGB PCA for its colour method. The Lab offset bands are project-specific settings chosen after visually comparing augmentation previews.

The protocols have different numbers of training families and therefore different numbers of draws per epoch. For a controlled comparison, match training updates across protocols and record the sampling policy for each run.

For validation and test, use the original images. Apply `species_weights(rows)` separately to each split if reporting a metric where every species should contribute equally. Each image receives a weight of one divided by the number of images of that species in the split.

To rebuild the CSVs from the fixed assignments, run `python3 dataset_protocols/build_splits.py` with Pillow installed.
