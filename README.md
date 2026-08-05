# Neuroscan — Brain Tumor MRI Classification

A complete, portfolio-ready deep learning application: a custom CNN trained
on brain MRI scans to classify **glioma**, **meningioma**, **pituitary
tumor**, or **no tumor**, served through a FastAPI backend and a dark,
scanner-themed single-page frontend, with calibrated confidence scores and
Grad-CAM explainability.

> **Disclaimer**: This is an educational / portfolio project demonstrating an
> applied deep learning pipeline. It is **not** a certified medical device
> and must never be used for real clinical decisions.

---

## Features

- Custom CNN (4 conv blocks + BatchNorm + Dropout + Global Average Pooling)
- Full anti-overfitting / anti-underfitting toolkit: augmentation, weight
  decay, LR scheduling, early stopping, best-checkpointing
- Automatic class-imbalance handling (weighted loss + `WeightedRandomSampler`)
- Full evaluation suite: accuracy, precision, recall, F1, macro-F1, confusion
  matrix, classification report, per-class accuracy, ROC/AUC
- **Confidence calibration** via temperature scaling — confidence scores
  reflect real-world reliability, not raw (overconfident) softmax output
- **Grad-CAM explainability** — visual heatmap of what the model looked at
- FastAPI backend with `/predict`, `/gradcam`, `/model-info`, `/health`
- Modern, responsive, dark-mode frontend with drag-and-drop upload

---

## Folder Structure

```
Project/
├── frontend/
│   └── index.html            # Single-file UI (HTML + CSS + JS)
├── backend/
│   ├── main.py                # FastAPI app (predict, gradcam, model-info, health)
│   ├── requirements.txt
│   ├── .env.example
│   └── model_artifacts/       # Created by the notebook's export step
├── dataset/
│   ├── README.md              # How to obtain and structure the Kaggle dataset
│   ├── Training/               # (you add this — see dataset/README.md)
│   └── Testing/                # (you add this — see dataset/README.md)
├── CNN_Project.ipynb          # Full ML pipeline, start to finish
└── README.md
```

---

## Dataset

**Brain Tumor MRI Dataset** (Kaggle, by Masoud Nickparvar) — 4 classes:
`glioma`, `meningioma`, `notumor`, `pituitary`. See
[`dataset/README.md`](dataset/README.md) for download and folder-layout
instructions.

---

## Installation

### 1. Clone and set up a Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
pip install jupyter scikit-learn seaborn pandas
```

### 2. Get the dataset

Follow [`dataset/README.md`](dataset/README.md).

---

## Training the Model

1. Launch Jupyter:
   ```bash
   jupyter notebook CNN_Project.ipynb
   ```
2. Run all cells top to bottom. The notebook will:
   - Explore and visualize the dataset
   - Preprocess, augment, and split it
   - Train the CNN with early stopping and checkpointing
   - Evaluate on the held-out test set
   - Calibrate confidence with temperature scaling
   - Generate Grad-CAM visualizations
   - Export `model_weights.pth` + `model_config.json` to
     `backend/model_artifacts/`

Training time depends on hardware; a GPU is recommended but not required.

---

## Running the Backend

```bash
cd backend
cp .env.example .env      # adjust if needed
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs (via
FastAPI's built-in Swagger UI) are at `http://localhost:8000/docs`.

### API Reference

| Method | Endpoint       | Description                                             |
|--------|----------------|-----------------------------------------------------------|
| GET    | `/health`      | Liveness check + whether the model is loaded              |
| GET    | `/model-info`  | Architecture, classes, test metrics, calibration info      |
| POST   | `/predict`     | Upload an image (`multipart/form-data`, field `file`) → predicted class + calibrated confidence + per-class probabilities |
| POST   | `/gradcam`     | Upload an image → predicted class + confidence + base64-encoded Grad-CAM overlay PNG |

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_scan.jpg"
```

Configuration is entirely via environment variables — see
[`backend/.env.example`](backend/.env.example).

---

## Running the Frontend

The frontend is a single static HTML file with no build step.

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500` in a browser. Make sure the backend is
running at the URL configured by `API_BASE_URL` in `index.html` (defaults to
`http://localhost:8000`).

---

## Model Architecture

A custom CNN (`BrainTumorCNN`), not a pretrained backbone, so the whole
pipeline is transparent and explainable end to end:

```
Input (3×224×224)
 → ConvBlock(3→32)   [Conv-BN-ReLU ×2, MaxPool, Dropout 0.20]
 → ConvBlock(32→64)  [Conv-BN-ReLU ×2, MaxPool, Dropout 0.25]
 → ConvBlock(64→128) [Conv-BN-ReLU ×2, MaxPool, Dropout 0.30]
 → ConvBlock(128→256)[Conv-BN-ReLU ×2, MaxPool, Dropout 0.30]
 → Global Average Pooling
 → Linear(256→128) → BatchNorm → ReLU → Dropout 0.5
 → Linear(128→num_classes)
```

See Section 5 of the notebook for the full rationale behind each design
choice.

---

## Performance Metrics

Populated automatically after training — see `backend/model_artifacts/model_config.json`
and the evaluation section of the notebook for the exact numbers on your
trained run (accuracy, macro-F1, per-class precision/recall/F1, confusion
matrix, ROC/AUC).

---

## Explainability

Grad-CAM highlights the image regions most responsible for a prediction by
weighting the last convolutional layer's feature maps by their gradient with
respect to the predicted class. This is exposed through the `/gradcam`
endpoint and rendered inline in the frontend, so predictions are never a
black box.

Confidence scores shown to the user are **calibrated** via temperature
scaling (fit on the validation set), not raw softmax — see Section 10 of the
notebook.

---

## Screenshots

_Add screenshots here after running the app locally:_

- `docs/screenshot-upload.png` — landing page / upload state
- `docs/screenshot-prediction.png` — prediction + confidence + probabilities
- `docs/screenshot-gradcam.png` — Grad-CAM overlay

---

## Future Improvements

- Transfer-learning baseline (e.g. EfficientNet-B0) for comparison
- k-fold cross-validation for tighter generalization estimates
- Dockerize backend + frontend for one-command deployment
- Add authentication/rate-limiting for a public deployment
- Batch prediction endpoint for multiple scans at once

---

## License

MIT — see `LICENSE` (add one if distributing publicly). The dataset itself
is subject to its own license on Kaggle; check the dataset page for terms
before redistribution.
