# app.py - VERSÃO OTIMIZADA COM LOGS DE TEMPO E DEBUGGING U-NET
import os
import io
import asyncio
import functools 
import traceback 
import time
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
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, disk

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
    model_type: str = "YOLO"
    area: Optional[float] = None

class AnalysisResponse(BaseModel):
    findings: List[Finding]
    model_used: str
    status: str
    timing_info: Optional[dict] = None

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
        # Detectar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {device}")
        lifespan_storage['device'] = device
        
        # Carregar modelo YOLO com otimizações
        print("📥 Carregando modelo YOLO...")
        yolo_path = 'models/best.pt'
        if os.path.exists(yolo_path):
            lifespan_storage['yolo_model'] = YOLO(yolo_path)
            # Mover para GPU se disponível
            if torch.cuda.is_available():
                lifespan_storage['yolo_model'].to(device)
            print("✅ Modelo YOLO carregado com sucesso.")
        else:
            print(f"⚠️ Modelo YOLO não encontrado em: {yolo_path}")
            lifespan_storage['yolo_model'] = None
        
        # Carregar modelo U-Net com debugging melhorado
        print("📥 Tentando carregar modelo U-Net...")
        unet_path = 'models/radiologia_5classes_fold_1_best.pth'
        
        if os.path.exists(unet_path):
            print(f"📂 Arquivo U-Net encontrado: {unet_path}")
            try:
                unet_model = UNet(n_channels=3, n_classes=5, bilinear=False)
                checkpoint = torch.load(unet_path, map_location=device)
                
                # Debug do checkpoint
                print(f"🔍 Tipo do checkpoint: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"🔍 Chaves do checkpoint: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print("📝 Usando 'model_state_dict'")
                    else:
                        state_dict = checkpoint
                        print("📝 Usando checkpoint diretamente")
                else:
                    state_dict = checkpoint
                    print("📝 Checkpoint é o state_dict diretamente")
                
                # Tentar carregar com strict=False para debug
                missing_keys, unexpected_keys = unet_model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"⚠️ Chaves faltando: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️ Chaves inesperadas: {unexpected_keys}")
                
                unet_model.to(device)
                unet_model.eval()
                
                lifespan_storage['unet_model'] = unet_model
                print("✅ Modelo U-Net carregado com sucesso.")
                
                # Teste rápido do U-Net
                print("🧪 Testando U-Net com tensor dummy...")
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 512, 512).to(device)
                    dummy_output = unet_model(dummy_input)
                    print(f"📊 Teste U-Net: entrada {dummy_input.shape} -> saída {dummy_output.shape}")
                    probs = torch.softmax(dummy_output, dim=1)
                    print(f"📊 Probabilidades por classe: {probs.mean(dim=(2,3)).squeeze()}")
                
            except Exception as unet_error:
                print(f"⚠️ Erro ao carregar U-Net: {unet_error}")
                print(f"📝 Traceback U-Net: {traceback.format_exc()}")
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
    "canal_mandibular": "Canal Mandibular",
    "impactado": "Impactado",
    "terceiro_ molar_inf": "Terceiro  Molar Inf",
    "terceiro_molar_sup": "Terceiro Molar Sup",
    "tto_endo": "Tto Endo",
    "decay": "Cárie",
    "filling": "Restauração",
    "periapical_lesion": "Lesão Periapical",
    "resto_residual": "Resto Residual"
}

# Mapeamento para classes U-Net otimizado
UNET_CLASS_MAP = {
    0: "background",
    1: "mandibula", 
    2: "maxila", 
    3: "dente",
    4: "canal"
}

# -----------------------------------------------------------------------------
# 4. FUNÇÕES DE PÓS-PROCESSAMENTO U-NET CORRIGIDAS
# -----------------------------------------------------------------------------
def merge_nearby_contours(contours, distance_threshold=50):
    """Merge contornos próximos para reduzir fragmentação"""
    if len(contours) <= 1:
        return contours
    
    merged = []
    used = set()
    
    for i, contour1 in enumerate(contours):
        if i in used:
            continue
            
        # Começar um novo grupo de contornos
        group = [contour1]
        used.add(i)
        
        # Encontrar contornos próximos
        for j, contour2 in enumerate(contours):
            if j in used or i == j:
                continue
                
            # Calcular distância entre centroides
            M1 = cv2.moments(contour1)
            M2 = cv2.moments(contour2)
            
            if M1["m00"] > 0 and M2["m00"] > 0:
                cx1, cy1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
                cx2, cy2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                if distance < distance_threshold:
                    group.append(contour2)
                    used.add(j)
        
        # Se há múltiplos contornos no grupo, merge them
        if len(group) > 1:
            # Combinar todos os contornos do grupo
            combined_contour = np.vstack(group)
            hull = cv2.convexHull(combined_contour)
            merged.append(hull)
        else:
            merged.append(group[0])
    
    return merged

def clean_segmentation_mask(mask, min_area=500, closing_size=3):
    """Limpa máscara de segmentação removendo ruído e pequenos objetos"""
    # Aplicar morfologia matemática para fechar buracos
    if closing_size > 0:
        kernel = disk(closing_size)
        mask = closing(mask, kernel)
    
    # Remover objetos pequenos
    mask_cleaned = remove_small_objects(mask.astype(bool), min_size=min_area)
    
    return mask_cleaned.astype(np.uint8)

def extract_major_regions(mask, max_regions=5, min_area_ratio=0.05):
    """Extrai apenas as regiões principais da máscara"""
    # Rotular regiões conectadas
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    if not regions:
        return mask
    
    # Calcular área total
    total_area = mask.sum()
    min_area = total_area * min_area_ratio
    
    # Filtrar regiões por tamanho
    major_regions = [r for r in regions if r.area >= min_area]
    
    # Ordenar por área (maiores primeiro)
    major_regions.sort(key=lambda x: x.area, reverse=True)
    
    # Manter apenas as maiores regiões
    major_regions = major_regions[:max_regions]
    
    # Criar nova máscara apenas com regiões principais
    new_mask = np.zeros_like(mask)
    for region in major_regions:
        coords = region.coords
        new_mask[coords[:, 0], coords[:, 1]] = 1
    
    return new_mask

def preprocess_for_unet(image_pil, target_size=(512, 512)):
    """Preprocessa imagem para entrada no U-Net"""
    print(f"🔄 Preprocessando imagem para U-Net: {target_size}")
    
    image_resized = image_pil.resize(target_size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image_resized).unsqueeze(0)
    return tensor, target_size

def analyze_with_unet(image_pil, unet_model, device):
    """Analisa imagem usando modelo U-Net com debugging melhorado"""
    start_time = time.time()
    print("🔬 Iniciando análise com U-Net otimizada...")
    
    if unet_model is None:
        return [], {"error": "U-Net model not available"}
    
    try:
        preprocess_start = time.time()
        input_tensor, unet_size = preprocess_for_unet(image_pil)
        input_tensor = input_tensor.to(device)
        preprocess_time = time.time() - preprocess_start
        print(f"📊 Tensor de entrada: {input_tensor.shape} ({preprocess_time:.2f}s)")
        
        with torch.no_grad():
            inference_start = time.time()
            print("🧠 Executando predição U-Net...")
            unet_output = unet_model(input_tensor)
            inference_time = time.time() - inference_start
            print(f"📊 Saída U-Net: {unet_output.shape} ({inference_time:.2f}s)")
            
            # Debug das probabilidades
            unet_probs = torch.softmax(unet_output, dim=1)
            mean_probs = unet_probs.mean(dim=(2,3)).squeeze()
            print(f"📊 Probabilidades médias por classe: {mean_probs}")
            max_probs = unet_probs.max(dim=(2,3))[0].squeeze()
            print(f"📊 Probabilidades máximas por classe: {max_probs}")
            
            unet_masks = torch.argmax(unet_probs, dim=1).squeeze(0).cpu().numpy()
            probs_numpy = unet_probs.squeeze(0).cpu().numpy()
            
            unique_classes, class_counts = np.unique(unet_masks, return_counts=True)
            print(f"📊 Máscaras U-Net: {unet_masks.shape}, classes únicas: {dict(zip(unique_classes, class_counts))}")
        
        postprocess_start = time.time()
        findings = []
        original_size = image_pil.size
        print(f"🔄 Processando segmentações otimizadas para tamanho original: {original_size}")
        
        # Parâmetros de filtros corrigidos
        class_params = {
            1: {"min_area": 1000, "max_regions": 2, "merge_distance": 100, "label": "Mandíbula", "min_confidence": 0.3},
            2: {"min_area": 1000, "max_regions": 2, "merge_distance": 80, "label": "Maxila", "min_confidence": 0.3},
            3: {"min_area": 200, "max_regions": 20, "merge_distance": 30, "label": "Dente", "min_confidence": 0.25},
            4: {"min_area": 100, "max_regions": 10, "merge_distance": 20, "label": "Canal", "min_confidence": 0.2}
        }
        
        for class_id in range(1, len(UNET_CLASS_MAP)):
            class_name = UNET_CLASS_MAP.get(class_id, f"class_{class_id}")
            
            if class_id not in unique_classes:
                print(f"⚪ Classe {class_name} (ID {class_id}): não detectada")
                continue
                
            params = class_params.get(class_id, {"min_area": 500, "max_regions": 5, "merge_distance": 50, "label": class_name.title(), "min_confidence": 0.25})
            
            # Criar máscara binária para a classe
            class_mask = (unet_masks == class_id).astype(np.uint8)
            pixel_count = class_mask.sum()
            print(f"🔍 Classe {class_name} (ID {class_id}): {pixel_count} pixels")
            
            # Verificar confiança média da classe
            class_confidence = float(probs_numpy[class_id][class_mask > 0].mean()) if pixel_count > 0 else 0.0
            print(f"🎯 Confiança média da classe {class_name}: {class_confidence:.3f}")
            
            if class_confidence < params["min_confidence"]:
                print(f"⚪ {class_name}: confiança muito baixa ({class_confidence:.3f} < {params['min_confidence']}), ignorando")
                continue
                
            if pixel_count < params["min_area"] // 4:
                print(f"⚪ {class_name}: muito poucos pixels, ignorando")
                continue
                
            # Limpeza da máscara
            class_mask_clean = clean_segmentation_mask(
                class_mask, 
                min_area=params["min_area"] // 10,
                closing_size=2
            )
            
            # Extrair apenas regiões principais
            class_mask_major = extract_major_regions(
                class_mask_clean,
                max_regions=params["max_regions"],
                min_area_ratio=0.02  # 2% da área total da classe
            )
            
            cleaned_pixel_count = class_mask_major.sum()
            print(f"🧹 {class_name}: {pixel_count} → {cleaned_pixel_count} pixels após limpeza")
            
            if cleaned_pixel_count < params["min_area"] // 4:
                print(f"⚪ {class_name}: área insuficiente após limpeza")
                continue
            
            # Redimensionar para tamanho original
            class_mask_resized = cv2.resize(
                class_mask_major, 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(
                class_mask_resized, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
                
            # Merge contornos próximos
            contours = merge_nearby_contours(contours, params["merge_distance"])
            print(f"🔗 {class_name}: {len(contours)} contornos após merge")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if area < params["min_area"]:
                    continue
                    
                # Calcular confiança baseada na probabilidade média da região
                mask_region = np.zeros_like(class_mask_resized)
                cv2.fillPoly(mask_region, [contour], 1)
                mask_region_unet = cv2.resize(mask_region, unet_size, interpolation=cv2.INTER_NEAREST)
                
                confidence = float(probs_numpy[class_id][mask_region_unet > 0].mean()) if mask_region_unet.sum() > 0 else class_confidence
                
                # Simplificar contorno
                epsilon = 0.005 * cv2.arcLength(contour, True)
                contour_simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(contour_simplified) < 3:
                    continue
                    
                # Normalizar coordenadas (0-100)
                contour_normalized = contour_simplified.reshape(-1, 2).astype(float)
                contour_normalized[:, 0] = (contour_normalized[:, 0] / original_size[0]) * 100
                contour_normalized[:, 1] = (contour_normalized[:, 1] / original_size[1]) * 100
                
                findings.append({
                    "id": f"unet_{class_name}_{i}",
                    "label": params["label"],
                    "confidence": max(0.5, min(0.95, confidence)),
                    "segmentation": contour_normalized.tolist(),
                    "model_type": "U-Net",
                    "area": area
                })
                
                print(f"✅ {params['label']}: confiança={confidence:.3f}, área={area:.0f}")
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - start_time
        
        timing_info = {
            "preprocess_time": preprocess_time,
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "total_time": total_time
        }
        
        print(f"🎉 Análise U-Net otimizada concluída: {len(findings)} achados ({total_time:.2f}s total)")
        return findings, timing_info
        
    except Exception as e:
        print(f"❌ Erro na análise U-Net: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        return [], {"error": str(e)}

def analyze_with_yolo(image, yolo_model, device):
    """Analisa imagem usando modelo YOLO com logs de tempo"""
    start_time = time.time()
    print(f"🎯 Iniciando análise com YOLO... Tamanho da imagem: {image.size}")
    
    if yolo_model is None:
        return [], {"error": "YOLO model not available"}
    
    try:
        # Configurações otimizadas para YOLO
        inference_start = time.time()
        results = yolo_model.predict(
            source=image, 
            conf=0.5,
            device=device if torch.cuda.is_available() else 'cpu',
            half=torch.cuda.is_available(),  # FP16 se GPU disponível
            verbose=False  # Reduzir logs verbosos
        )
        inference_time = time.time() - inference_start
        print(f"⏱️ Tempo de inferência YOLO: {inference_time:.2f}s")
        
        postprocess_start = time.time()
        findings = []
        
        if not results:
            return findings, {"inference_time": inference_time, "total_time": time.time() - start_time}

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
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - start_time
        
        timing_info = {
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "total_time": total_time
        }
        
        print(f"⏱️ Tempo total YOLO: {total_time:.2f}s (inferência: {inference_time:.2f}s, pós-processamento: {postprocess_time:.2f}s)")
        return findings, timing_info
        
    except Exception as e:
        print(f"❌ Erro na análise YOLO: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        return [], {"error": str(e)}

# -----------------------------------------------------------------------------
# 5. ENDPOINT PRINCIPAL COM SELEÇÃO DE MODELO E TIMING
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), model_type: str = Form(default="yolo")):
    print("\n" + "="*50)
    print(f"📡 Rota /analyze acessada com modelo: {model_type.upper()}")
    print("="*50)
    
    yolo_model = lifespan_storage.get('yolo_model')
    unet_model = lifespan_storage.get('unet_model')
    device = lifespan_storage.get('device', 'cpu')
    
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
        
        if model_type == "yolo":
            print("🎯 Executando análise com YOLO")
            findings, timing_info = analyze_with_yolo(image, yolo_model, device)
            model_used = "YOLO"
            
        elif model_type == "unet":
            print("🔬 Executando análise com U-Net")
            findings, timing_info = analyze_with_unet(image, unet_model, device)
            model_used = "U-Net"
        
        print(f"\n🎉 Análise concluída com {model_used}")
        print(f"📊 Total de achados: {len(findings)}")
        if "total_time" in timing_info:
            print(f"⏱️ Tempo total: {timing_info['total_time']:.2f}s")
        print("="*50)
        
        return {
            "findings": findings,
            "model_used": model_used,
            "status": "success",
            "timing_info": timing_info
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
        "default": "yolo" if yolo_available else ("unet" if unet_available else None),
        "device": str(lifespan_storage.get('device', 'cpu'))
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
