# ACE-Dynamics

**Official implementation of _Adaptive Confidence-Weighted Expansion for Trustworthy Multi-omics Multimodal Fusion_ (ICPR 2026).**

[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://link.springer.com/chapter/10.1007/978-3-032-31404-8_22)
[![arXiv](https://img.shields.io/badge/arXiv-2607.20742-b31b1b.svg)](https://arxiv.org/abs/2607.20742)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

ACE (**Adaptive Confidence-weighted Expansion**) is a trustworthy multimodal learning framework for multi-omics classification. It expands the multimodal representation using correlation-derived views, estimates feature- and modality-level reliability, adaptively weights modalities according to confidence, and introduces a global confidence head for the final fused prediction.

> **Paper:** Mohammad Raahemi, Ali Sekhavati, Alireza Maleki, and Hamid Nasiri,  
> “Adaptive Confidence-Weighted Expansion for Trustworthy Multi-omics Multimodal Fusion,”  
> *Pattern Recognition – ICPR 2026*, LNCS 16819, pp. 322–338, Springer, 2026.  
> DOI: [10.1007/978-3-032-31404-8_22](https://doi.org/10.1007/978-3-032-31404-8_22)

---

## Overview

Multimodal and multi-omics models can benefit from complementary information across data sources, but their predictions may become unreliable when some modalities are noisy, missing, or weakly informative.

ACE addresses this problem through four main ideas:

1. **Correlation-driven modality expansion**  
   Each original omics view is augmented with a complementary view generated from intra-modality feature correlations.

2. **Feature informativeness estimation**  
   A learnable gating mechanism suppresses weak or noisy features before modality-specific encoding.

3. **Confidence-aware multimodal fusion**  
   Each view receives a modality-specific classification head and confidence estimator. The learned confidence is used to scale the encoded representation and adaptively weight its training contribution.

4. **Global confidence estimation**  
   After confidence-weighted features are fused, ACE predicts both the final class and a global trust/confidence score for the fused decision.

The framework also uses input corruption and reconstruction as regularization to improve robustness on limited and noisy biological data.

---

## ACE Pipeline

Conceptually, the implementation follows:

```text
Original Omics Views
        │
        ├──► Correlation-based View Generation
        │
        ▼
Original + Derived Views
        │
        ▼
Noise / Dropout Corruption
        │
        ▼
Feature Informativeness Gating
        │
        ▼
Modality-specific Encoders
        │
        ├──► Modality Classifier
        ├──► Modality Confidence / TCP
        └──► Input Reconstruction
        │
        ▼
Confidence-weighted Features
        │
        ▼
Feature Concatenation / Fusion
        │
        ├──► Final Classifier
        └──► Global TCP / Confidence
```

For an original modality \(X_i\), the additional correlation-derived view is generated as

\[
X'_i = X_i \cdot \mathrm{corr}(X_i),
\]

where the correlation matrix is computed **only from the training data** and is then applied to both training and test samples.

---

## Supported Datasets

The repository includes data directories for the four datasets evaluated in the paper:

| Dataset | Task | Classes | Metrics used by the code |
|---|---|---:|---|
| **BRCA** | Breast cancer subtype classification | 5 | Accuracy, Weighted F1, Macro F1 |
| **KIPAN** | Kidney cancer classification | 3 | Accuracy, Weighted F1, Macro F1 |
| **LGG** | Lower-grade glioma classification | 2 | Accuracy, F1, ROC-AUC |
| **ROSMAP** | Aging / Alzheimer's-related classification | 2 | Accuracy, F1, ROC-AUC |

The code expects six views for every dataset. Views `1`–`3` are the original data views and views `4`–`6` are generated from their intra-modality correlations.

---

## Repository Structure

```text
ACE-Dynamics/
├── BRCA/                 # BRCA train/test views and labels
├── KIPAN/                # KIPAN train/test views and labels
├── LGG/                  # LGG train/test views and labels
├── ROSMAP/               # ROSMAP train/test views and labels
│
├── create_modals.py      # Generates correlation-derived views 4–6
├── main.py               # Main experiment entry point
├── model.py              # ACE model and loss components
└── train_test.py         # Data loading, training, testing, and metrics
```

Each dataset directory is expected to follow this layout:

```text
DATASET/
├── labels_tr.csv
├── labels_te.csv
├── 1_tr.csv
├── 1_te.csv
├── 2_tr.csv
├── 2_te.csv
├── 3_tr.csv
├── 3_te.csv
├── 4_tr.csv
├── 4_te.csv
├── 5_tr.csv
├── 5_te.csv
├── 6_tr.csv
└── 6_te.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AliGreen0/ACE-Dynamics.git
cd ACE-Dynamics
```

Creating a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or on Windows:

```bash
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn torch
```

### GPU requirement

The current training implementation directly calls `.cuda()` for the model and labels. Therefore, **the repository as currently written expects a CUDA-enabled PyTorch installation and an NVIDIA GPU**.

For CPU-only execution, replace the direct `.cuda()` calls in `train_test.py` with device-aware `.to(device)` logic.

The experiments reported in the paper were conducted on Ubuntu 20.04.1 with NVIDIA RTX 3080 Ti GPUs.

---

## Generating the Correlation-Derived Modalities

The repository contains `create_modals.py`, which generates views `4`, `5`, and `6` from original views `1`, `2`, and `3`.

Set the target dataset inside `create_modals.py`:

```python
dataset_name = 'KIPAN'  # 'BRCA', 'ROSMAP', 'LGG', or 'KIPAN'
```

Then run:

```bash
python create_modals.py
```

For each original view, the script:

1. loads the training and test features;
2. computes the feature-wise Pearson correlation matrix from the **training set**;
3. projects the training data through that correlation matrix;
4. applies the same training-derived transformation to the test set;
5. saves the resulting complementary view as `4_*`, `5_*`, or `6_*`.

This keeps test information out of the correlation estimation step.

> If the derived files already exist in the dataset folders, this preprocessing step does not need to be repeated.

---

## Training and Evaluation

Open `main.py` and select one of the supported datasets:

```python
data_folder = 'LGG'  # 'BRCA', 'ROSMAP', 'LGG', or 'KIPAN'
testonly = False
```

Then run:

```bash
python main.py
```

By default, `main.py` repeats the experiment **20 times** and reports the mean and standard deviation of the final metrics.

For binary datasets (`LGG`, `ROSMAP`), the output format is:

```text
Final Results:
Accuracy: <mean> ± <std>
F1:       <mean> ± <std>
AUC:      <mean> ± <std>
```

For multiclass datasets (`BRCA`, `KIPAN`), the output format is:

```text
Final Results:
Accuracy:    <mean> ± <std>
F1 weighted: <mean> ± <std>
F1 macro:    <mean> ± <std>
```

Model checkpoints are stored under:

```text
./model/<DATASET>/checkpoint.pt
```

---

## Dataset-Specific Defaults in the Current Code

The current `train_test.py` uses the following dataset-specific settings:

| Dataset | Epochs | Learning rate | Classes | Adaptive-weight coefficient |
|---|---:|---:|---:|---:|
| BRCA | 2500 | `1e-4` | 5 | `1e-4` |
| KIPAN | 1000 | `1e-4` | 3 | `1` |
| LGG | 2000 | `1e-4` | 2 | `1` |
| ROSMAP | 1000 | `3e-4` | 2 | `1` |

Other defaults include:

```text
Hidden dimension : 1000
Dropout          : 0.5
Optimizer        : Adam
Weight decay     : 1e-4
LR scheduler     : StepLR
Step size        : 500
Gamma            : 0.2
Test interval    : 50 epochs
```

### Reproducibility note

The published paper describes experiments run for **1200 epochs**, while the current public implementation contains the dataset-specific epoch values shown above. If you need to reproduce a specific paper protocol exactly, verify and set the desired epoch count in `train_test.py` before running experiments.

---

## Main Model Components

The ACE implementation in `model.py` contains the following major components.

### 1. Feature-level gating

Each corrupted input view is passed through a sigmoid feature-information layer:

```text
corrupted input
      ↓
feature-information gate
      ↓
element-wise feature weighting
```

This allows ACE to suppress less informative features before encoding.

### 2. Modality-specific encoders

Every view has an independent encoder that maps the gated input to a shared latent dimensionality.

### 3. Modality-specific confidence estimation

Each encoded view produces:

- classification logits;
- a modality-level confidence score.

The confidence score scales the modality representation before fusion and is also used in the adaptive loss.

### 4. Reconstruction regularization

A reconstruction head predicts the original modality input from its latent representation. This self-supervised objective encourages information-preserving representations under corruption.

### 5. Multimodal fusion

Confidence-weighted modality representations are concatenated and passed to the final multimodal classifier.

### 6. Global TCP

ACE adds a global confidence head after fusion. It estimates the reliability of the final multimodal prediction rather than only estimating confidence independently for each modality.

---

## Loss Function

Training combines several objectives:

```text
Total Loss
├── Final multimodal classification loss
├── Global confidence calibration loss
├── Modality-specific classification/confidence losses
├── Adaptive confidence-weighted classification loss
├── Input reconstruction loss
├── Feature informativeness penalty
├── L1 regularization
└── L2 regularization
```

This combination is designed to jointly optimize predictive performance, confidence calibration, modality robustness, and compact feature representations.

---

## Paper

**Adaptive Confidence-Weighted Expansion for Trustworthy Multi-omics Multimodal Fusion**

Mohammad Raahemi, Ali Sekhavati, Alireza Maleki, Hamid Nasiri

**28th International Conference on Pattern Recognition (ICPR 2026)**  
*Pattern Recognition, Lecture Notes in Computer Science, Vol. 16819*  
Springer, pp. 322–338, 2026.

- **Springer:** https://link.springer.com/chapter/10.1007/978-3-032-31404-8_22
- **DOI:** https://doi.org/10.1007/978-3-032-31404-8_22
- **arXiv:** https://arxiv.org/abs/2607.20742

---

## Citation

If you use this repository or ACE in your research, please cite the published paper:

```bibtex
@inproceedings{raahemi2026ace,
  author    = {Mohammad Raahemi and Ali Sekhavati and Alireza Maleki and Hamid Nasiri},
  title     = {Adaptive Confidence-Weighted Expansion for Trustworthy Multi-omics Multimodal Fusion},
  booktitle = {Pattern Recognition: 28th International Conference, ICPR 2026, Lyon, France, August 17--22, 2026, Proceedings, Part VIII},
  series    = {Lecture Notes in Computer Science},
  volume    = {16819},
  pages     = {322--338},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-31404-8_22}
}
```

---

## Authors

- **Mohammad Raahemi** — University of Ottawa
- **Ali Sekhavati** — University of Ottawa
- **Alireza Maleki** — RIV Lab, Bu-Ali Sina University
- **Hamid Nasiri** — Lancaster University

---

## Acknowledgment

ACE builds on the broader line of trustworthy multimodal learning and extends multimodal dynamics with correlation-based modality expansion, adaptive confidence weighting, reconstruction regularization, and global confidence estimation.

For methodological details, equations, experimental comparisons, and ablation studies, please refer to the published paper.
