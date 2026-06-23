# CloMA
*Clo*nogenic *M*orphometric *A*nalysis
CloMA is a tool for segment and extract features from images of colonies in circular plates.
---

## Features
- Plate image preprocessing
- Segmentation (threshold-based morphological analysis)
- Feature extraction

## Installation

### Requirements
- Python 3.8 or higher
- pip or conda

### Quick Install

1. Clone the repository:
```bash
git clone https://github.com/labsinal/CloMA.git
cd CloMA
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. For GUI (napari plugin):
```bash
pip install napari[all]
```

### Using Conda (Optional)

```bash
conda create -n cloma python=3.10
conda activate cloma
git clone https://github.com/labsinal/CloMA.git
cd CloMA
pip install -e .
pip install napari[all]
```

## Usage (GUI)

Open napari with:
```bash
napari
```

On the top bar, select Plugins > CloMA Tools.

## Usage (CLI)

On the clonned project folder run:

```bash
python3 -m CloMA
```

It will open the help image