# Dataset — Brain Tumor MRI Dataset (Kaggle)

This project uses the **Brain Tumor MRI Dataset** by Masoud Nickparvar:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset is not included in this repository (Kaggle's terms require
downloading it through Kaggle directly, and it's too large to vendor here).

## Setup

1. Download the dataset from the link above (requires a free Kaggle account),
   or via the Kaggle CLI:
   ```bash
   pip install kaggle
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
   unzip brain-tumor-mri-dataset.zip -d .
   ```
2. Make sure the folder layout under `dataset/` looks exactly like this
   (this is the layout the dataset ships with, and what `CNN_Project.ipynb`
   expects):

   ```
   dataset/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

3. Open `CNN_Project.ipynb` and run all cells top to bottom. The final cells
   export `model_weights.pth` and `model_config.json` into
   `backend/model_artifacts/`, which the FastAPI backend loads at startup.

## Class definitions

| Class        | Description                                   |
|--------------|------------------------------------------------|
| `glioma`     | Tumor arising from glial cells                 |
| `meningioma` | Tumor arising from the meninges                |
| `notumor`    | No tumor present                               |
| `pituitary`  | Tumor of the pituitary gland                   |
