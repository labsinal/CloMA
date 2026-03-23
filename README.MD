# CloMA
*Clo*nogenic *M*orphometric *A*nalysis
CloMA is a tool for segment and extract features from images of colonies in circular plates.
---

## Features
- Plate image preprocessing
- Segmentation (uses a hybrid approach with threshold + Cellpose)
- Feature extraction

## Installation

Install or update [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda create -n CloMA
conda activate CloMA
conda install pip
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