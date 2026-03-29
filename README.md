# The Rashomon Effect for Visualizing High-Dimensional Data

This repository contains the implementation code for the paper: **The Rashomon Effect for Visualizing High-Dimensional Data** [[arXiv(Release Soon)]](TODO)

## Overview

Dimension reduction (DR) is inherently non-unique: multiple embeddings can preserve the structure of high-dimensional data equally well while differing in layout or geometry. This paper formally defines the **Rashomon set for DR** -- the collection of "good" embeddings -- and shows how embracing this multiplicity leads to more powerful and trustworthy representations.

We pursue three goals:
1. **PCA-informed alignment** -- steer embeddings toward principal components, making axes interpretable without distorting local neighborhoods.
2. **Concept-alignment regularization** -- align an embedding dimension with external knowledge such as class labels or user-defined concepts.
3. **Common knowledge extraction** -- identify trustworthy and persistent nearest-neighbor relationships across the Rashomon set, and use them to construct refined embeddings with improved structure.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── code/
│   ├── contrastive-ne/          # Modified contrastive neighbor embedding package
│   │   ├── src/cne/             # Core CNE implementation
│   │   ├── tests/               # Tests and example notebooks
│   │   └── pyproject.toml
│   ├── paramrepulsor/           # Modified parametric PaCMAP package
│   │   ├── source/parampacmap/  # Core ParamRepulsor implementation
│   │   ├── setup.cfg
│   │   └── requirements.txt
│   └── scripts/                 # Experiment scripts
│       ├── data.py              # Data loading utilities
│       ├── evaluation.py        # Evaluation metrics
│       ├── experiment_1.py      # Experiment 1: embedding generation
│       ├── experiment_1_metrics.py
│       ├── experiment_1_metrics_missing_ratios.py
│       ├── experiment_1_metrics_summarization.py
│       ├── experiment_1_metrics_visualizations.py
│       ├── experiment_1_pca.py
│       ├── experiment_2.py      # Experiment 2: common knowledge extraction
│       ├── experiment_2_comparison.py
│       ├── experiment_2_metrics.py
│       └── cutoff_template.csv   # Template for Rashomon set cutoff values
```

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
- [PaCMAP](https://github.com/YingfanWang/PaCMAP)

### Install packages

```bash
# Install the modified contrastive-ne package
pip install -e ./code/contrastive-ne

# Install the modified paramrepulsor package
pip install -e ./code/paramrepulsor
```

### Additional dependencies

```bash
pip install numpy scikit-learn matplotlib pandas annoy scipy numba
```

## Datasets

All datasets used in the paper are described in Appendix D. Due to size constraints, data files are not included in this repository. Please refer to the cited works for dataset downloads:

- **MNIST** ([LeCun et al., 2010](http://yann.lecun.com/exdb/mnist/))
- **Fashion-MNIST** ([Xiao et al., 2017](https://github.com/zalandoresearch/fashion-mnist))
- **USPS** ([Hull, 1994](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/))
- **COIL-20** ([Nene et al., 1996](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php))
- **20 Newsgroups** (scikit-learn)
- **Mammoth** ([Smithsonian 3D](https://3d.si.edu/))
- **Airplane** point cloud dataset
- **scRNA-seq datasets**: Kang et al. (2018), Stuart et al. (2019), Human Cortex, NeurIPS 2021

After downloading, place data files in a `data/` directory (or set the `DATA_DIR` environment variable).

## Running Experiments

All experiment scripts are in `code/scripts/`. They use environment variables for path configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Directory containing dataset files |
| `RESULTS_DIR` | `./results` | Directory for experiment outputs |
| `OUTPUT_DIR` | `./output` | Directory for evaluation output (used by `evaluation.py`) |
| `TEST_RESULTS_DIR` | `./test_results` | Directory for test results (used by `evaluation.py`) |

### Cutoff file

The summarization and comparison scripts (`experiment_1_metrics_summarization.py`, `experiment_2_comparison.py`) require a `cutoff.csv` file that specifies the label weight threshold defining the Rashomon set boundary for each method/dataset/task combination. A template is provided in `cutoff_template.csv`. Copy and edit it:

```bash
cp cutoff_template.csv cutoff.csv
# Edit cutoff.csv to set appropriate cutoff values for your experiments
```

The CSV has columns: `method`, `dataset`, `task`, `cutoff` (label weight value).

### Experiment 1: Embedding generation with alignment

```bash
cd code/scripts

# Generate embeddings for a dataset with PCA alignment
python experiment_1.py --dataset MNIST --method pacmap --task_type pca

# Compute evaluation metrics
python experiment_1_metrics.py --dataset MNIST --method pacmap --task_type pca --metric Jaccard

# Visualize metrics across label weights
python experiment_1_metrics_visualizations.py --dataset MNIST --method pacmap --task_type pca --metric Jaccard

# Summarize metrics
python experiment_1_metrics_summarization.py --dataset MNIST --task_type pca
```

### Experiment 2: Common knowledge extraction

```bash
cd code/scripts

# Extract common knowledge embeddings from the Rashomon set
python experiment_2.py --dataset MNIST --method pacmap --task_type pca

# Compute metrics for common knowledge embeddings
python experiment_2_metrics.py --dataset MNIST --method pacmap --task_type pca --metric FastKNN

# Compare with original embeddings
python experiment_2_comparison.py --dataset MNIST --task_type pca
```

### Supported methods

- `pacmap` -- Parametric PaCMAP
- `umap` -- Parametric UMAP
- `infonce` -- Parametric Info-NC-t-SNE
- `negtsne` -- Parametric Neg-t-SNE
- `ncvis` -- Parametric NCVis

### Supported datasets

`MNIST`, `FMNIST`, `USPS`, `20NG`, `COIL20`, `AirPlane`, `Mammoth`, `kang`, `stuart`, `human_cortex`, `neurips2021`, `seurat`

## Modified Packages

This repository includes modified versions of two open-source packages:

- **[contrastive-ne](https://github.com/berenslab/contrastive-ne)** -- Contrastive Neighbor Embeddings (Damrich et al., ICLR 2023). Modified to support PCA-alignment and concept-alignment regularization terms.
- **[ParamRepulsor](https://github.com/hyhuang00/ParamRepulsor)** -- Parametric PaCMAP. Modified to support alignment regularization and batch training with custom neighbor graphs.

## Citation

```bibtex
@inproceedings{rashomon_dr_2026,
  title={The Rashomon Effect for Visualizing High-Dimensional Data},
  author={Yiyang Sun, Haiyang Huang,  Gaurav Rajesh Parikh, Cynthia Rudin},
  booktitle={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
