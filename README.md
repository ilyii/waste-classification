# Waste Classification

Authors: [Gabriel](https://github.com/Gabriel9753), [Ilyi](https://github.com/ilyii)

## Overview

This project focuses on Waste Classification, a machine learning task where we aim to classify different types of waste materials into specific categories. The primary goal is to build an efficient waste sorting system using deep learning techniques. The project involves training a model to recognize and classify waste items, such as glass, organic, paper, restmuell, (elektromuell,) and wertstoff, based on images of these materials.

## Prerequisites

To run this project, you'll need the following prerequisites:

- Python 3.11 (Lower versions may work, but are untested.)
- Other project-specific dependencies (please refer to the project's `requirements.txt` file)
- A dataset of waste images (please refer to the "Data" section for more information)
- A GPU and some time

You can install the necessary Python packages by running:

`pip install -r requirements.txt`

## Data
The project expects the following data structure for waste classification:
    
```bash
dataset
├── glas
|   ├── image_1.jpg
|   ├── image_2.jpg
|   └── ...
├── organic
|   ├── image_1.jpg
|   ├── image_2.jpg
|   └── ...
├── paper
├── restmuell
├── elektromuell
└── wertstoff
```
The dataset should be organized into directories, with each directory corresponding to a waste category (e.g., glas, organic, paper, etc.). Inside each category directory, there should be image files (e.g., image_1.jpg, image_2.jpg) representing different waste items.

Please ensure that the dataset is structured according to this format for proper training and evaluation of the waste classification model.

## Getting Started
### (1) Clone the repository
```bash
git clone https://github.com/Gabriel9753/waste-classification.git
```	

### (2) Install the required dependencies:
```bash
cd waste-classification-ml
pip install -r requirements.txt
```	
## Training
```bash
python train.py --data_path DATA_PATH --output_dir OUTPUT_DIR 
```

## Hyperparameter Tuning
```bash
python hpo.py --data_path DATA_PATH --output_dir OUTPUT_DIR 
```
