import io
import os
import base64
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# =========================
# Configurações principais
# =========================
STATIC_DIR = "static"
# Usa exatamente o caminho e nome do seu arquivo .pth dentro de model/
MODEL_WEIGHTS = "model/v1_u_net__model_1024x512_b3.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Conforme classes.csv (0..23)
NUM_CLASSES = 24

# Transparência da sobreposição
ALPHA = float(os.getenv("MASK_ALPHA", "0.4"))

# Paleta simples (classe 0 fundo preto; demais geradas/repetidas)
# Pode ser substituída por uma paleta clínica estável.
BASE_PALETTE = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (64, 64, 64),
    (192, 64, 0),
    (0, 192, 64),
    (64, 0, 192),
    (192, 0, 64),
    (64, 192, 0),
    (0, 64, 192),
    (192, 192, 0),
    (192, 0, 192),
    (0, 192, 192),
    (128, 128, 128),
]
def get_palette(n):
    if n <= len(BASE_PALETTE):
        return BASE_PALETTE[:n]
    pal = list(BASE_PALETTE)
    rng = np.random.default_rng(123)
    while len(pal) < n:
        pal.append(tuple(int(c) for c in rng.integers(0, 255, size=3)))
    return pal

PALETTE = get_palette(NUM_CLASSES)

# =========================
# U‑Net simples (troque se tiver sua classe real)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=24, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        rev = list(reversed(features))
        ch = features[-1] * 2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch, f))
            ch = f
        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape[-2:] != s.shape[-2:]:
                x = T.functional.resize(x, s.shape[-2:], antialias=True)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        return self.head(x)

# =========================
# Carregar modelo
# =========================
def load_model(weights_path: str, num_classes: int) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Modelo não encontrado em {weights_path}")
    model = UNet(in_channels=3, num_classes=num_classes)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    # strict=False para tolerar pequenas diferenças de chaves
    model.load_state_dict(new_state, strict=False)  # PyTorch permite isso em casos práticos[20][9][12]
    model.to(DEVICE).eval()
    return model

# =========================
# Pré/pós-processamento
# =========================
# Ajuste mean/std e resize para refletir seu treino do notebook.
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD = [0.229, 0.224, 0.225]

# Opcional: se treinou com tamanho fixo (ex.: 1024x512), defina aqui:
TARGET_SIZE = (512, 1024)  # (H, W) — ajuste conforme seu notebook

def preprocess(pil_img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    orig_hw = (h, w)
    if TARGET_SIZE is not None:
        pil_img_res = pil_img.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
    else:
        pil_img_res = pil_img
    res_hw = (pil_img_res.size[1], pil_img_res.size[0])
    img = T.ToTensor()(pil_img_res)
    img = T.Normalize(IM_MEAN, IM_STD)(img)
    img = img.unsqueeze(0).to(DEVICE)
    return img, orig_hw, res_hw

def postprocess_mask(logits: torch.Tensor, orig_hw: Tuple[int, int], res_hw: Tuple[int, int]) -> np.ndarray:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)[0]  # [C,h,w]
        mask = torch.argmax(probs, dim=0).float()  # [h,w]
        mask_np = mask.cpu().numpy()
        # Primeiro volta para tamanho original da imagem
        mask_np = cv2.resize(mask_np, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)
        return mask_np

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    ids = mask.astype(np.uint8)
    for cls_id in np.unique(ids):
        if cls_id < len(PALETTE):
            color[ids == cls_id] = PALETTE[cls_id]
        else:
            # fallback
            rng = np.random.default_rng(int(cls_id))
            color[ids == cls_id] = tuple(int(c) for c in rng.integers(0, 255, size=3))
    return color  # BGR-like tuple order used consistently abaixo

def overlay_mask_on_image(pil_img: Image.Image, color_mask_bgr: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    # Transparência com addWeighted (alpha + beta = 1)[10][13][21]
    overlay = cv2.addWeighted(img_bgr, 1.0, color_mask_bgr, alpha, 0)
    return overlay

# =========================
# FastAPI
# =========================
app = FastAPI(title="radiologIA U‑Net API", version="1.0")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL: Optional[nn.Module] = None

@app.on_event("startup")
def on_startup():
    global MODEL
    MODEL = load_model(MODEL_WEIGHTS, NUM_CLASSES)

@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"status": "ok", "message": "radiologIA U‑Net API up"})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    try:
        content = await file.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem")

    tensor, orig_hw, res_hw = preprocess(pil)

    with torch.no_grad():
        logits = MODEL(tensor)  # [1,C,h,w]

    mask = postprocess_mask(logits, orig_hw, res_hw)  # [H,W] ids 0..C-1

    color_mask = colorize_mask(mask)  # BGR
    overlay = overlay_mask_on_image(pil, color_mask, alpha=ALPHA)  # BGR

    ok1, mask_png = cv2.imencode(".png", color_mask)
    ok2, overlay_png = cv2.imencode(".png", overlay)
    if not ok1 or not ok2:
        raise HTTPException(status_code=500, detail="Falha ao codificar PNGs")

    return {
        "width": orig_hw[1],
        "height": orig_hw[0],
        "num_classes": NUM_CLASSES,
        "mask_png_base64": base64.b64encode(mask_png.tobytes()).decode("utf-8"),  # retorno JSON com base64[4][14]
        "overlay_png_base64": base64.b64encode(overlay_png.tobytes()).decode("utf-8"),
    }
