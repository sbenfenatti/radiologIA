# app.py - Versão com Pipeline Híbrido YOLO + U-Net
import os
import io
import asyncio
import functools 
import traceback 
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# -----------------------------------------------------------------------------
# 1. ARQUITETURA U-NET PARA REFINAMENTO
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# -----------------------------------------------------------------------------
# 2. DEFINIÇÃO DOS MODELOS DE DADOS (PYDANTIC)
# -----------------------------------------------------------------------------
class Finding(BaseModel):
    id: str
    label: str
    confidence: float
    segmentation: List[List[float]]

class AnalysisResponse(BaseModel):
    findings: List[Finding]

class ChatPart(BaseModel):
    text: str

class ChatContent(BaseModel):
    role: str
    parts: List[ChatPart]

class ChatHistory(BaseModel):
    history: List[ChatContent]

# -----------------------------------------------------------------------------
# 3. CONFIGURAÇÃO INICIAL E CICLO DE VIDA DA APLICAÇÃO
# -----------------------------------------------------------------------------
lifespan_storage = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando a aplicação...")
    lifespan_storage['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
    if not lifespan_storage['gemini_api_key']:
        print("❌ AVISO: A variável de ambiente 'GEMINI_API_KEY' não foi encontrada.")
    
    try:
        # Carregar modelo YOLO
        lifespan_storage['yolo_model'] = YOLO('models/best.pt')
        print("✅ Modelo YOLO carregado com sucesso.")
        
        # Carregar modelo U-Net para refinamento
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {device}")
        
        unet_model = UNet(n_channels=3, n_classes=5)
        model_path = 'models/radiologia_5classes_fold_1_best.pth'
        
        if os.path.exists(model_path):
            unet_model.load_state_dict(torch.load(model_path, map_location=device))
            unet_model.to(device)
            unet_model.eval()
            lifespan_storage['unet_model'] = unet_model
            lifespan_storage['device'] = device
            print("✅ Modelo U-Net carregado com sucesso.")
        else:
            print("⚠️ Modelo U-Net não encontrado. Usando apenas YOLO.")
            lifespan_storage['unet_model'] = None
            
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        lifespan_storage['yolo_model'] = None
        lifespan_storage['unet_model'] = None
    
    lifespan_storage['http_client'] = httpx.AsyncClient()
    print("✅ Cliente HTTP assíncrono criado.")
    yield
    print("👋 Encerrando a aplicação...")
    await lifespan_storage['http_client'].aclose()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MAPEAMENTO DE LABELS ---
LABEL_MAP = {
    "lesao_periapical": "Lesão Periapical",
    "carie": "Cárie",
    "fratura_radicular": "Fratura Radicular",
    "calculo_dental": "Cálculo Dental",
    "restauracao": "Restauração",
    "implante": "Implante",
    "dente_incluso": "Dente Incluso",
    "pre_molar_inf": "Pré-Molar Inferior",
    "jaw": "Mandíbula",
    "incisivo_lateral_inf": "Incisivo Lateral Inferior",
    "incisivo_central_sup": "Incisivo Central Superior",
    "molar_inf": "Molar Inferior",
    "maxila": "Maxila",
    "incisivo_lateral_sup": "Incisivo Lateral Superior",
    "incisivo_central_inf": "Incisivo Central Inferior",
    "molar_sup": "Molar Superior",
    "pre_molar_sup": "Pré-Molar Superior",
    "canino_inf": "Canino Inferior",
    "canino_sup": "Canino Superior",
}

# -----------------------------------------------------------------------------
# 4. FUNÇÕES DE REFINAMENTO U-NET
# -----------------------------------------------------------------------------
def preprocess_for_unet(image_pil, target_size=(1052, 512)):
    """Preprocessa imagem para entrada no U-Net"""
    # Redimensionar para o tamanho do treinamento
    image_resized = image_pil.resize(target_size, Image.LANCZOS)
    
    # Converter para tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
    return tensor, target_size

def refine_masks_with_unet(image_pil, yolo_masks, unet_model, device):
    """
    Refina máscaras do YOLO usando U-Net
    Foca em mandíbula, maxila e dentes (ignora canal)
    """
    if unet_model is None:
        return yolo_masks
    
    try:
        # Preprocessar imagem
        input_tensor, unet_size = preprocess_for_unet(image_pil)
        input_tensor = input_tensor.to(device)
        
        # Predição U-Net
        with torch.no_grad():
            unet_output = unet_model(input_tensor)
            unet_probs = torch.softmax(unet_output, dim=1)
            unet_masks = torch.argmax(unet_probs, dim=1).squeeze(0).cpu().numpy()
        
        # Mapear classes U-Net (assumindo: 0=fundo, 1=mandíbula, 2=maxila, 3=dente, 4=canal)
        unet_class_map = {
            1: "jaw",      # mandíbula
            2: "maxila",   # maxila  
            3: "dente",    # dente
            # 4: canal ignorado por enquanto
        }
        
        refined_masks = []
        original_size = image_pil.size
        
        # Gerar máscaras refinadas das classes principais
        for class_id, class_name in unet_class_map.items():
            class_mask = (unet_masks == class_id).astype(np.uint8)
            
            if class_mask.sum() > 0:  # Se há pixels desta classe
                # Redimensionar de volta para tamanho original
                class_mask_resized = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) > 100:  # Filtrar contornos muito pequenos
                        # Converter contorno para formato esperado (normalizado)
                        contour_normalized = contour.reshape(-1, 2).astype(float)
                        contour_normalized[:, 0] /= original_size[0]  # normalizar x
                        contour_normalized[:, 1] /= original_size[1]  # normalizar y
                        
                        refined_masks.append({
                            "class_name": class_name,
                            "contour": contour_normalized * 100,  # formato esperado pelo frontend
                            "confidence": 0.85  # confiança alta para refinamento
                        })
        
        return refined_masks
        
    except Exception as e:
        print(f"⚠️ Erro no refinamento U-Net: {e}")
        return yolo_masks

async def run_model_prediction(model, image):
    loop = asyncio.get_event_loop()
    predict_with_args = functools.partial(model.predict, source=image, conf=0.5)
    results = await loop.run_in_executor(None, predict_with_args)
    return results

# -----------------------------------------------------------------------------
# 5. ENDPOINT PRINCIPAL COM PIPELINE HÍBRIDO
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    print("\n📡 Rota /analyze acessada!")
    yolo_model = lifespan_storage.get('yolo_model')
    unet_model = lifespan_storage.get('unet_model')
    device = lifespan_storage.get('device', 'cpu')
    
    if not yolo_model:
        raise HTTPException(status_code=500, detail="Modelo YOLO não está carregado.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ETAPA 1: Detecção inicial com YOLO
        results_generator = await run_model_prediction(yolo_model, image)
        results = list(results_generator)
        
        all_findings = []
        
        if not results:
            print("⚠️ Nenhum resultado do YOLO.")
            return {"findings": []}

        prediction = results[0]
        class_names = prediction.names

        # Processar detecções YOLO
        yolo_findings = []
        if prediction.masks:
            for i, box in enumerate(prediction.boxes):
                if i < len(prediction.masks.xyn):
                    technical_label = class_names.get(int(box.cls[0].item()), "Desconhecido")
                    friendly_label = LABEL_MAP.get(technical_label, technical_label.replace("_", " ").title())
                    
                    mask_points = prediction.masks.xyn[i]
                    
                    yolo_findings.append({
                        "id": f"yolo_finding_{i}",
                        "label": friendly_label,
                        "confidence": box.conf[0].item(),
                        "segmentation": (mask_points * 100).tolist()
                    })

        # ETAPA 2: Refinamento com U-Net (se disponível)
        if unet_model:
            print("🔬 Aplicando refinamento U-Net...")
            refined_masks = refine_masks_with_unet(image, yolo_findings, unet_model, device)
            
            # Adicionar máscaras refinadas
            for i, refined_mask in enumerate(refined_masks):
                friendly_label = LABEL_MAP.get(refined_mask["class_name"], 
                                             refined_mask["class_name"].replace("_", " ").title())
                
                all_findings.append({
                    "id": f"refined_finding_{i}",
                    "label": f"{friendly_label} (Refinado)",
                    "confidence": refined_mask["confidence"],
                    "segmentation": refined_mask["contour"].tolist()
                })
        
        # Adicionar detecções YOLO originais
        all_findings.extend(yolo_findings)
        
        mode = "Híbrido YOLO + U-Net" if unet_model else "Apenas YOLO"
        print(f"✅ Análise concluída [{mode}]. Enviando {len(all_findings)} achados.")
        return {"findings": all_findings}
        
    except Exception as e:
        print(f"❌ Erro crítico na análise: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro interno do servidor ao analisar imagem")

# -----------------------------------------------------------------------------
# 6. ENDPOINT CHAT (INALTERADO)
# -----------------------------------------------------------------------------
@app.post("/chat")
async def handle_chat(payload: ChatHistory):
    print("\n📡 Rota /chat acessada!")
    api_key = lifespan_storage.get('gemini_api_key')
    client = lifespan_storage.get('http_client')
    if not api_key:
        raise HTTPException(status_code=500, detail="A chave da API do Gemini não está configurada no servidor.")
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    request_body = {"contents": [item.dict() for item in payload.history]}
    try:
        response = await client.post(gemini_api_url, headers=headers, json=request_body, timeout=60.0)
        response.raise_for_status()
        print("✅ Resposta da API Gemini recebida com sucesso.")
        return response.json()
    except httpx.RequestError as e:
        print(f"❌ Erro ao chamar a API do Gemini: {e}")
        raise HTTPException(status_code=503, detail=f"Erro de comunicação com a API do Gemini: {e}")
    except Exception as e:
        print(f"❌ Erro Crítico no chat: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro interno do servidor no processamento do chat")

# -----------------------------------------------------------------------------
# 7. SERVINDO ARQUIVOS ESTÁTICOS (INALTERADO)
# -----------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse('static/index.html')

@app.get("/{path:path}")
async def serve_static_files(path: str):
    file_path = os.path.join('static', path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse('static/index.html')