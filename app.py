# app.py - VERSÃO COM SELEÇÃO DE MODELO (YOLO vs U-Net)
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
import torchvision
from ultralytics import YOLO
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# -----------------------------------------------------------------------------
# 1. ARQUITETURA U-NET CORRIGIDA PARA CORRESPONDER AO MODELO TREINADO
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
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
    model_type: str = "YOLO"  # Novo campo para identificar o modelo usado

class AnalysisResponse(BaseModel):
    findings: List[Finding]
    model_used: str
    status: str

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
        print("📥 Carregando modelo YOLO...")
        yolo_path = 'models/best.pt'
        if os.path.exists(yolo_path):
            lifespan_storage['yolo_model'] = YOLO(yolo_path)
            print("✅ Modelo YOLO carregado com sucesso.")
        else:
            print(f"⚠️ Modelo YOLO não encontrado em: {yolo_path}")
            lifespan_storage['yolo_model'] = None
        
        # Carregar modelo U-Net 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {device}")
        lifespan_storage['device'] = device
        
        print("📥 Tentando carregar modelo U-Net...")
        unet_path = 'models/radiologia_5classes_fold_1_best.pth'
        
        if os.path.exists(unet_path):
            print(f"📂 Arquivo U-Net encontrado: {unet_path}")
            try:
                # Tentar carregar com arquitetura padrão U-Net
                unet_model = UNet(n_channels=3, n_classes=5, bilinear=False)
                
                # Carregar estado do modelo
                checkpoint = torch.load(unet_path, map_location=device)
                
                # Se checkpoint é um dict com 'model_state_dict', usar isso
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Tentar carregar - se falhar, modelo não é compatível
                unet_model.load_state_dict(state_dict, strict=False)
                unet_model.to(device)
                unet_model.eval()
                
                lifespan_storage['unet_model'] = unet_model
                print("✅ Modelo U-Net carregado com sucesso.")
                
            except Exception as unet_error:
                print(f"⚠️ Erro ao carregar U-Net: {unet_error}")
                print("📝 Modelo U-Net incompatível com arquitetura atual")
                lifespan_storage['unet_model'] = None
        else:
            print(f"⚠️ Modelo U-Net não encontrado em: {unet_path}")
            lifespan_storage['unet_model'] = None
            
    except Exception as e:
        print(f"❌ Erro geral ao carregar modelos: {e}")
        print(f"📝 Detalhes do erro: {traceback.format_exc()}")
        lifespan_storage['yolo_model'] = None
        lifespan_storage['unet_model'] = None
    
    lifespan_storage['http_client'] = httpx.AsyncClient()
    print("✅ Cliente HTTP assíncrono criado.")
    print("🎉 Aplicação iniciada com sucesso!")
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

# Mapeamento para classes U-Net (assumindo 5 classes baseado no nome do arquivo)
UNET_CLASS_MAP = {
    0: "background",     # fundo
    1: "jaw",           # mandíbula  
    2: "maxila",        # maxila
    3: "dente",         # dente
    4: "canal",         # canal radicular
}

# -----------------------------------------------------------------------------
# 4. FUNÇÕES DE ANÁLISE COM U-NET
# -----------------------------------------------------------------------------
def preprocess_for_unet(image_pil, target_size=(512, 512)):
    """Preprocessa imagem para entrada no U-Net"""
    print(f"🔄 Preprocessando imagem para U-Net: {target_size}")
    
    # Redimensionar mantendo proporção
    image_resized = image_pil.resize(target_size, Image.LANCZOS)
    
    # Converter para tensor e normalizar
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
    return tensor, target_size

def analyze_with_unet(image_pil, unet_model, device):
    """
    Analisa imagem usando modelo U-Net
    """
    print("🔬 Iniciando análise com U-Net...")
    
    if unet_model is None:
        return []
    
    try:
        # Preprocessar imagem
        input_tensor, unet_size = preprocess_for_unet(image_pil)
        input_tensor = input_tensor.to(device)
        print(f"📊 Tensor de entrada: {input_tensor.shape}")
        
        # Predição U-Net
        with torch.no_grad():
            print("🧠 Executando predição U-Net...")
            unet_output = unet_model(input_tensor)
            print(f"📊 Saída U-Net: {unet_output.shape}")
            
            # Aplicar softmax e obter classes preditas
            unet_probs = torch.softmax(unet_output, dim=1)
            unet_masks = torch.argmax(unet_probs, dim=1).squeeze(0).cpu().numpy()
            probs_numpy = unet_probs.squeeze(0).cpu().numpy()
            
            print(f"📊 Máscaras U-Net: {unet_masks.shape}, classes únicas: {np.unique(unet_masks)}")
        
        findings = []
        original_size = image_pil.size
        print(f"🔄 Processando segmentações para tamanho original: {original_size}")
        
        # Processar cada classe (ignorar background = 0)
        for class_id in range(1, len(UNET_CLASS_MAP)):
            class_name = UNET_CLASS_MAP.get(class_id, f"class_{class_id}")
            class_mask = (unet_masks == class_id).astype(np.uint8)
            pixel_count = class_mask.sum()
            
            print(f"🔍 Classe {class_name} (ID {class_id}): {pixel_count} pixels")
            
            if pixel_count > 100:  # Filtro de área mínima
                # Redimensionar de volta para tamanho original
                class_mask_resized = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"🔍 {len(contours)} contornos encontrados para {class_name}")
                
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 200:  # Filtrar contornos pequenos
                        # Calcular confiança média para esta região
                        mask_region = np.zeros_like(class_mask_resized)
                        cv2.fillPoly(mask_region, [contour], 1)
                        mask_region_original = cv2.resize(mask_region, unet_size, interpolation=cv2.INTER_NEAREST)
                        confidence = float(probs_numpy[class_id][mask_region_original > 0].mean())
                        
                        # Converter contorno para formato esperado (normalizado 0-100)
                        contour_simplified = cv2.approxPolyDP(contour, 2, True)
                        contour_normalized = contour_simplified.reshape(-1, 2).astype(float)
                        contour_normalized[:, 0] = (contour_normalized[:, 0] / original_size[0]) * 100
                        contour_normalized[:, 1] = (contour_normalized[:, 1] / original_size[1]) * 100
                        
                        # Usar label amigável se disponível
                        friendly_label = LABEL_MAP.get(class_name, class_name.replace("_", " ").title())
                        
                        findings.append({
                            "id": f"unet_finding_{class_id}_{i}",
                            "label": friendly_label,
                            "confidence": max(0.5, min(0.99, confidence)),  # Clamp entre 0.5 e 0.99
                            "segmentation": contour_normalized.tolist(),
                            "model_type": "U-Net"
                        })
                        
                        print(f"✅ {friendly_label}: confiança={confidence:.3f}, área={area:.0f}")
        
        print(f"🎉 Análise U-Net concluída: {len(findings)} achados")
        return findings
        
    except Exception as e:
        print(f"❌ Erro na análise U-Net: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        return []

async def run_yolo_prediction(model, image):
    """Executa predição YOLO de forma assíncrona"""
    print("🔄 Executando predição YOLO...")
    loop = asyncio.get_event_loop()
    predict_with_args = functools.partial(model.predict, source=image, conf=0.5)
    results = await loop.run_in_executor(None, predict_with_args)
    return results

def analyze_with_yolo(image, yolo_model):
    """Analisa imagem usando modelo YOLO (versão síncrona para compatibilidade)"""
    print("🎯 Iniciando análise com YOLO...")
    
    if yolo_model is None:
        return []
    
    try:
        results = yolo_model.predict(source=image, conf=0.5)
        findings = []
        
        if not results:
            return findings

        prediction = results[0]
        class_names = prediction.names if hasattr(prediction, 'names') else {}
        
        print(f"📊 Classes YOLO disponíveis: {class_names}")

        if hasattr(prediction, 'masks') and prediction.masks is not None:
            print(f"🔍 Processando {len(prediction.masks)} máscaras YOLO...")
            
            for i, box in enumerate(prediction.boxes):
                try:
                    if i < len(prediction.masks.xyn):
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                        technical_label = class_names.get(cls_id, "Desconhecido")
                        friendly_label = LABEL_MAP.get(technical_label, technical_label.replace("_", " ").title())
                        
                        mask_points = prediction.masks.xyn[i]
                        confidence = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                        
                        findings.append({
                            "id": f"yolo_finding_{i}",
                            "label": friendly_label,
                            "confidence": confidence,
                            "segmentation": (mask_points * 100).tolist(),
                            "model_type": "YOLO"
                        })
                        
                        print(f"✅ YOLO - {friendly_label}: {confidence:.3f}")
                        
                except Exception as e:
                    print(f"⚠️ Erro processando detecção YOLO {i}: {e}")
                    continue
        else:
            print("⚠️ Nenhuma máscara encontrada no resultado YOLO")
        
        return findings
        
    except Exception as e:
        print(f"❌ Erro na análise YOLO: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        return []

# -----------------------------------------------------------------------------
# 5. ENDPOINT PRINCIPAL COM SELEÇÃO DE MODELO
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), model_type: str = Form(default="yolo")):
    print("\n" + "="*50)
    print(f"📡 Rota /analyze acessada com modelo: {model_type.upper()}")
    print("="*50)
    
    yolo_model = lifespan_storage.get('yolo_model')
    unet_model = lifespan_storage.get('unet_model')
    device = lifespan_storage.get('device', 'cpu')
    
    # Validar modelo solicitado
    model_type = model_type.lower()
    if model_type not in ["yolo", "unet"]:
        raise HTTPException(status_code=400, detail="Tipo de modelo deve ser 'yolo' ou 'unet'")
    
    if model_type == "yolo" and not yolo_model:
        raise HTTPException(status_code=500, detail="Modelo YOLO não está disponível")
    
    if model_type == "unet" and not unet_model:
        raise HTTPException(status_code=500, detail="Modelo U-Net não está disponível")
    
    try:
        print("📥 Lendo arquivo de imagem...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"📊 Imagem carregada: {image.size}")
        
        # Análise baseada no modelo escolhido
        if model_type == "yolo":
            print("🎯 Executando análise com YOLO")
            findings = analyze_with_yolo(image, yolo_model)
            model_used = "YOLO"
            
        elif model_type == "unet":
            print("🔬 Executando análise com U-Net")
            findings = analyze_with_unet(image, unet_model, device)
            model_used = "U-Net"
        
        print(f"\n🎉 Análise concluída com {model_used}")
        print(f"📊 Total de achados: {len(findings)}")
        print("="*50)
        
        return {
            "findings": findings,
            "model_used": model_used,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Erro crítico na análise: {e}")
        print(f"📝 Traceback completo:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")

# -----------------------------------------------------------------------------
# 6. ENDPOINT PARA VERIFICAR MODELOS DISPONÍVEIS
# -----------------------------------------------------------------------------
@app.get("/models/available")
async def get_available_models():
    """Retorna quais modelos estão disponíveis"""
    yolo_available = lifespan_storage.get('yolo_model') is not None
    unet_available = lifespan_storage.get('unet_model') is not None
    
    return {
        "yolo": yolo_available,
        "unet": unet_available,
        "default": "yolo" if yolo_available else ("unet" if unet_available else None)
    }

# -----------------------------------------------------------------------------
# 7. ENDPOINT CHAT (INALTERADO)
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
# 8. SERVINDO ARQUIVOS ESTÁTICOS
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