# Napari-Hydra

A [napari](https://napari.org) plugin for **single-shot nested instance segmentation** of biomedical objects using [HydraStarDist](https://arxiv.org/abs/2504.12078) — a branched deep learning architecture built on [StarDist](https://github.com/stardist/stardist).

Napari-Hydra enables GUI-based detection, segmentation, and quantification of spatially correlated, star-convex objects (e.g. viral plaques nested within tissue culture wells) directly from digital photographs — no microscopy required.

## How to Cite Us

If you use this plugin or the underlying methods in your research, please cite the following:

> De, T., Thangamani, S., Urbański, A., & Yakimovich, A. (2025). A digital photography dataset for Vaccinia Virus plaque quantification using Deep Learning. *Scientific Data*, *12*, 719. [https://doi.org/10.1038/s41597-025-05030-8](https://doi.org/10.1038/s41597-025-05030-8)

> De, T., Urbanski, A., & Yakimovich, A. (2025). Single-shot Star-convex Polygon-based Instance Segmentation for Spatially-correlated Biomedical Objects. *arXiv preprint arXiv:2504.12078*. [https://doi.org/10.48550/arXiv.2504.12078](https://doi.org/10.48550/arXiv.2504.12078)

## Overview

Traditional virological plaque assays rely on manual counting — a process that is laborious, subjective, and error-prone. Napari-Hydra addresses this by providing an end-to-end deep learning pipeline accessible through napari's graphical interface.

### What is HydraStarDist?

HydraStarDist (HSD) extends the StarDist architecture with a **joint encoder and branched decoders**, enabling simultaneous detection of two categories of nested objects in a single forward pass. This is in contrast to conventional approaches that require independent models for each object type (e.g. one for wells, one for plaques).

The shared encoder implicitly captures spatial correlations between nested objects (e.g. plaques can only appear within wells), resulting in more meaningful representations and improved joint detection accuracy.

### Key Features

- **Single-shot prediction** — detect and segment both wells and plaques simultaneously
- **Interactive thresholding** — tune probability and NMS thresholds per object class
- **Per-well plaque counting** — automatic quantification with per-well breakdown
- **Model fine-tuning** — adapt pre-trained models to your own data directly from the GUI
- **Stack support** — process time-lapse image stacks with frame-by-frame results
- **Export** — save prediction summaries including plaque counts and morphometrics

## Installation

```bash
pip install napari-hydra
```

Or install in development mode:

```bash
git clone https://github.com/plaque2/napari-hydra.git
cd napari-hydra
pip install -e ".[test]"
```

## Usage

Launch napari with the plugin:

```bash
napari -w napari-hydra
```

1. **Load an image** — drag and drop or use `File > Open` to load a plaque assay photograph.
2. **Run Prediction** — select the image layer and model, then click **Run Prediction**. Wells and plaques are detected simultaneously and displayed as label overlays.
3. **Count** — click **Count** to compute per-well plaque counts.
4. **Tune** — fine-tune the model on your own annotated data using the **Tune Model** button. Fine-tuned models are saved as new timestamped copies, preserving the original.
5. **Export** — save a prediction summary with plaque counts and areas.

## License

BSD-3-Clause
