# NMGAT

[English](README.md) | [简体中文](README.zh-CN.md)

Official implementation of **Graph attention network for nuclear mass predictions with interpretable attention patterns**.

NMGAT connects neighbouring nuclides in a graph and learns corrections to WS4, DZ, and LDM binding energies with a graph attention network. This repository provides ready-to-use data, training code, and prediction tables for both small and global graphs.

## ⚙️ Installation

Python 3.12 is recommended. From the project directory, run:

```bash
pip install -r requirements.txt
```

## 📂 Data

The datasets are included and ready for training.

| Dataset | Location |
|---|---|
| Small graph: 2,456 training, 23 test, and 1,085 unlabelled nodes | `data/processed/` |
| Global graph: 7,879 nodes for WS4/LDM; 7,775 for DZ | `data/global/` |
| Compiled datasets and predictions | `data/tables/` |
| Original Excel tables | `data/tables_original/` |

A uses all training labels. B masks the 5% with the largest experimental uncertainties (123 labels), keeping every graph node and the same training settings. The compiled global predictions are provided directly; the additional 4,211 nuclides were predicted on an expanded graph with frozen weights.

## 🚀 Training

To train WS4:

```bash
python scripts/train.py --model WS4 --variant A --graph small --output runs/WS4-A
```

Choose `--model DZ` or `--model LDM` for another base model, `--variant B` for setting B, or `--graph global` for the global graph.

The default run trains 200 seeds for 12,000 epochs each. For a quick example:

```bash
python scripts/train.py --model WS4 --variant B --graph global --epochs 2 --num-seeds 1 --device cpu --output runs/demo
```

Weights, predictions, and training histories are saved to the output directory. Use a new directory for each experiment.

## ▶️ Prediction

Pass a trained checkpoint to predict. The model and graph settings are read automatically from its saved metadata.

```bash
python scripts/predict.py --checkpoint runs/demo/artifacts/seed_2026013000/final.pt --output prediction.csv
```

## 📊 Reproducing results

Pretrained weights are provided separately as `nmgat-checkpoints.zip`. Extract it next to the project directory, then run:

```bash
python scripts/reproduce.py --checkpoints ../nmgat-checkpoints
```

This evaluates all three models with setting A. Add `--variant B` for B or `--model WS4` to run just one model. Predictions are saved to `runs/reproduced/` and checked automatically against the reference results.

| Model | Baseline RMSD | NMGAT RMSD |
|---|---:|---:|
| WS4 | 0.752 | 0.392 |
| DZ | 1.058 | 0.418 |
| LDM | 3.275 | 0.523 |

Values are in MeV for the 23 test nuclides in setting A. Result reproduction uses the supplied weights; new training uses the shared A/B configuration.

## 🧩 Code structure

```text
configs/       Training settings
data/          Datasets and prediction tables
src/nmgat/     Data processing, model, and training
scripts/       Training, prediction, and reproduction commands
results/       Reference results
tests/         Data and code checks
docs/          Additional documentation
```

To change the model, start with `src/nmgat/model.py`. Training parameters live in `configs/aligned.json`. See the [code guide](docs/ARCHITECTURE.md), [data dictionary](docs/DATA_DICTIONARY.md), and [reproducibility notes](docs/REPRODUCIBILITY.md) for more details.
