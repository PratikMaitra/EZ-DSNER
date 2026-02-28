# EZ-DSNER: An End-to-End Framework for Distant Annotation
Repository for EZ-DSNER annotation framework


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

**EZ-DSNER** is a comprehensive annotation framework that unifies distant labeling, training, inference, and post-processing for Distantly Supervised Named Entity Recognition (DS-NER). The framework is designed for users of all technical levels — from domain experts who need quick distant annotation to researchers running full DS-NER training pipelines.


[Demo Video](https://youtu.be/TN5ApcsnKZk) · [Live Demo](https://pratikmaitra.github.io/EZ-DSNER/)

---

## Overview

EZ-DSNER has two main modules:

1. **Web Application** (`ez_dsner.html`) — A lightweight, browser-based tool for distant labeling using dictionaries. No installation required.
2. **Jupyter Notebook** (`EZ-DSNER.ipynb`) — An interactive notebook for training, inference, evaluation, and post-processing of DS-NER methods.

---

## Supported DS-NER Methods

Please download the method repositories from their respective GitHub repositories. Some of the methods might require patching to work. For datasets, please download commonly used DS-NER datasets like QTL etc

| Method | 
|--------|
| [BOND] |
| [ATSEN] |
| [DeSERT] |
---

## Repository Structure

```
EZ-DSNER/
├── ez_dsner.html              # Web application (open in browser)
├── EZ-DSNER.ipynb             # Jupyter notebook for training & evaluation
│
├── dsner_wrapper.py           # Unified wrapper for all 8 DS-NER methods
├── dsner_data.py              # Data format converter (8 formats)
├── dsner_postprocess.py       # Rule-based post-processing & evaluation
├── eval.py                    # Standalone evaluation (strict + relaxed)
│
├── requirements.txt           # Python 3.7 compatible dependencies
│
├── data/
│   └── QTL/                   # QTL dataset
│       ├── train.txt          # Distantly labeled training data
│       ├── test.txt           # Gold-standard test data
│       ├── valid.txt          # Validation data
│       └── types.txt          # Entity type definitions
│
├── dictionaries/              # Domain dictionaries for distant labeling
├── format_converter/          # Format conversion utilities
│
├── ATSEN/                     # ATSEN method source code
├── BOND/                      # BOND method source code
├── CuPuL/                     # CuPuL method source code
├── DeSERT/                    # DeSERT method source code           
└── SCDL/                      # SCDL method source code
```

---

## Installation

We recommend creating a conda/Python environment based on Python 3.7, which is the most compatible with the methods.

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA 11.6 (for training)

### Setup

**1. Clone the repository:**

```bash
git clone https://github.com/PratikMaitra/EZ-DSNER.git
cd EZ-DSNER
```

**2. Create the conda environment and install dependencies:**

```bash
conda create -n dsner python=3.7 -y
conda activate dsner
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

**3. Register the Jupyter kernel:**

```bash
pip install ipykernel
python -m ipykernel install --user --name dsner --display-name "Python (dsner)"
```

**4. Launch Jupyter and select the kernel:**

```bash
jupyter notebook EZ-DSNER.ipynb
```

Then select **Kernel → Change kernel → Python (dsner)**.

## Quick Start

### Web Application (No Installation)

Open `ez_dsner.html` in any modern browser. The web app runs entirely client-side — no server, login, or installation required.

1. Enter a PubMed ID or upload a `.txt` file
2. Upload a dictionary file (`.txt`, one term per line)
3. Click **Apply Dictionary Match** to annotate
4. Export annotations in Text, CoNLL, or BioC XML format

**Features:** dictionary customization (add/remove/merge terms), manual annotation, dark mode, dyslexia-friendly font, large text, and high contrast accessibility options.

### Jupyter Notebook (Training & Evaluation)

Open `EZ-DSNER.ipynb` and run cells sequentially:

```python
from dsner_wrapper import DSNERWrapper

# Train CuPuL on QTL dataset
cupul = DSNERWrapper(
    method="CuPuL",
    project_root=".",
    dataset="qtl",
    gpu_ids="0",
    pretrained_model="roberta-base",
    learning_rate=5e-7,
    num_train_epochs=1,
    drop_other=0.5,
    loss_type="MAE",
    m=20,
    curriculum_train_lr=2e-7,
    curriculum_train_epochs=5,
    self_train_lr=5e-7,
    self_train_epochs=5,
    max_seq_length=300,
)

cupul.train()
cupul.predict()
cupul.evaluate()
```

The wrapper automatically resolves dataset names, builds CLI commands, and manages method-specific configurations.

---

## Post-Processing

EZ-DSNER includes a rule-based post-processing module that refines DS-NER predictions. The rules target common errors in biomedical and animal science domain annotations.

| Rule | Description |
|------|-------------|
| `span_consistency` | If a multi-word span is tagged as an entity somewhere in the document, all case-insensitive matches are also tagged |
| `prep_bridging` | Bridges prepositions between adjacent entity spans (e.g., *[body weight] of [cattle]* → full span) |
| `abbrev_resolution` | Propagates labels between full forms and abbreviations (e.g., *Main Stem Node Number (MSNN)*) |
| `pos_filtering` | Removes singleton false positives that are not nouns based on POS tagging |

=

## Helper Scripts

In addition to the notebook, individual scripts can be used from the command line.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


## Acknowledgments

This work was developed in collaboration with the [Animal QTLdb](https://www.animalgenome.org/cgi-bin/QTLdb/index) group at Iowa State University, involving animal science domain experts and computer scientists.



