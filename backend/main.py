"""
Brain Tumor MRI Classifier — FastAPI Backend
==============================================

Serves the CNN trained in ../CNN_Project.ipynb. Loads model weights + config
once at startup, then exposes:

    GET  /health        -> liveness / model-loaded check
    GET  /model-info     -> architecture, classes, metrics, calibration info
    POST /predict         -> class probabilities + calibrated confidence
    POST /gradcam          -> Grad-CAM overlay for an uploaded image (base64 PNG)

Configuration is read entirely from environment variables (see .env.example)
so the same image can be deployed to different environments without code
changes.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# ---------------------------------------------------------------------------
# Configuration (environment variables, with sensible local defaults)
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model_artifacts"))
MAX_UPLOAD_SIZE_MB = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("brain-tumor-api")

# ---------------------------------------------------------------------------
# Model definition (must match the architecture trained in the notebook)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_head: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.2),
            ConvBlock(32, 64, dropout=0.25),
            ConvBlock(64, 128, dropout=0.3),
            ConvBlock(128, 256, dropout=0.3),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(128, num_classes),
        )
        self.gradcam_target_layer = self.features[-1].block[3]

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class GradCAM:
    """Grad-CAM hooked on the model's last convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]


class ModelInfoResponse(BaseModel):
    architecture: str
    class_names: list
    image_size: int
    test_accuracy: float
    test_macro_f1: float
    calibration_temperature: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# App state — loaded once at startup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="CNN-based brain tumor MRI classification with Grad-CAM explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: Dict = {"model": None, "config": None, "transform": None, "gradcam": None, "device": None}


@app.on_event("startup")
def load_model() -> None:
    """Loads model weights + config once, so every request reuses the same
    in-memory model instead of paying disk/deserialization cost per call."""
    import json

    config_path = MODEL_DIR / "model_config.json"
    weights_path = MODEL_DIR / "model_weights.pth"

    if not config_path.exists() or not weights_path.exists():
        logger.error(
            "Model artifacts not found in %s. Run the notebook's export cell first.", MODEL_DIR
        )
        state["model"] = None
        return

    with open(config_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorCNN(num_classes=len(config["class_names"]))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config["resize_size"], config["resize_size"])),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(config["normalize_mean"], config["normalize_std"]),
    ])

    state.update({
        "model": model,
        "config": config,
        "transform": transform,
        "gradcam": GradCAM(model, model.gradcam_target_layer),
        "device": device,
    })
    logger.info("Model loaded successfully on device=%s, classes=%s", device, config["class_names"])


def _ensure_model_loaded() -> None:
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the notebook and export artifacts to model_artifacts/ first.",
        )


def _validate_and_open_image(raw_bytes: bytes) -> Image.Image:
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f}MB > {MAX_UPLOAD_SIZE_MB}MB limit).")
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return image


def _predict(image: Image.Image):
    config = state["config"]
    device = state["device"]
    input_tensor = state["transform"](image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = state["model"](input_tensor)
        calibrated = logits / config["temperature"]
        probs = F.softmax(calibrated, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    return input_tensor, pred_idx, probs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=state["model"] is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _ensure_model_loaded()
    config = state["config"]
    return ModelInfoResponse(
        architecture=config["architecture"],
        class_names=config["class_names"],
        image_size=config["image_size"],
        test_accuracy=config["test_accuracy"],
        test_macro_f1=config["test_macro_f1"],
        calibration_temperature=config["temperature"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    _ensure_model_loaded()
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPEG, PNG, or WEBP image.")

    raw_bytes = await file.read()
    image = _validate_and_open_image(raw_bytes)

    try:
        _, pred_idx, probs = _predict(image)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed on the server.") from exc

    class_names = state["config"]["class_names"]
    return PredictionResponse(
        predicted_class=class_names[pred_idx],
        confidence=float(probs[pred_idx]),
        class_probabilities={cls: float(p) for cls, p in zip(class_names, probs)},
    )


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)) -> JSONResponse:
    _ensure_model_loaded()
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPEG, PNG, or WEBP image.")

    raw_bytes = await file.read()
    image = _validate_and_open_image(raw_bytes)

    try:
        input_tensor, pred_idx, probs = _predict(image)
        cam = state["gradcam"].generate(input_tensor, class_idx=pred_idx)
        overlay_b64 = _render_overlay_png(input_tensor, cam)
    except Exception as exc:
        logger.exception("Grad-CAM generation failed")
        raise HTTPException(status_code=500, detail="Grad-CAM generation failed on the server.") from exc

    class_names = state["config"]["class_names"]
    return JSONResponse({
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "gradcam_overlay_png_base64": overlay_b64,
    })


def _render_overlay_png(input_tensor: torch.Tensor, cam: np.ndarray) -> str:
    """Denormalizes the input tensor, applies a jet colormap to the Grad-CAM
    heatmap, blends the two, and returns a base64-encoded PNG string that the
    frontend can drop directly into an <img> tag."""
    import matplotlib.cm as cm

    config = state["config"]
    mean = np.array(config["normalize_mean"]).reshape(3, 1, 1)
    std = np.array(config["normalize_std"]).reshape(3, 1, 1)
    img = input_tensor.squeeze(0).cpu().numpy() * std + mean
    img = np.clip(img.transpose(1, 2, 0), 0, 1)

    heatmap = cm.get_cmap("jet")(cam)[:, :, :3]
    overlay = np.clip(0.55 * img + 0.45 * heatmap, 0, 1)

    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
