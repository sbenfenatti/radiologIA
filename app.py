# app.py - VERSÃO CORRIGIDA com ResNet-UNet
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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# -----------------------------------------------------------------------------
# 1. ARQUITETURA RESNET-UNET CORRETA
# -----------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=5):
        super(ResNetUNet, self).__init__()
        
        # ResNet34 como encoder
        resnet = torchvision.models.resnet34(pretrained=False)
        
        # Extrair camadas do ResNet
        self.encoder = nn.Module()
        self.encoder.conv1 = resnet.conv1
        self.encoder.bn1 = resnet.bn1
        self.encoder.relu = resnet.relu
        self.encoder.maxpool = resnet.maxpool
        self.encoder.layer1 = resnet.layer1
        self.encoder.layer2 = resnet.layer2
        self.encoder.layer3 = resnet.layer3
        self.encoder.layer4 = resnet.layer4
        
        # Decoder com 5 blocos
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 256),  # decoder.blocks.0
            DecoderBlock(256, 128),  # decoder.blocks.1
            DecoderBlock(128, 64),   # decoder.blocks.2
            DecoderBlock(64, 64),    # decoder.blocks.3
            DecoderBlock(64, 64),    # decoder.blocks.4
        ])
        
        # Cabeça de segmentação
        self.segmentation_head = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Camada de saída final
        self.outc = nn.Conv2d(n_classes, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        # Decoder
        for decoder_block in self.decoder:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = decoder_block(x)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        # Output layer
        x = self.outc(x)
        
        return x

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
        print("📥 Carregando modelo YOLO...")
        lifespan_storage['yolo_model'] = YOLO('models/best.pt')
        print("✅ Modelo YOLO carregado com sucesso.")
        
        # Carregar modelo U-Net para refinamento
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {device}")
        
        print("📥 Carregando modelo ResNet-UNet...")
        unet_model = ResNetUNet(n_classes=5)
        model_path = 'models/radiologia_5classes_fold_1_best.pth'
        
        if os.path.exists(model_path):
            print(f"📂 Arquivo encontrado: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            unet_model.load_state_dict(state_dict)
            unet_model.to(device)
            unet_model.eval()
            lifespan_storage['unet_model'] = unet_model
            lifespan_storage['device'] = device
            print("✅ Modelo ResNet-UNet carregado com sucesso.")
        else:
            print(f"⚠️ Modelo U-Net não encontrado em: {model_path}")
            lifespan_storage['unet_model'] = None
            
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
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

# -----------------------------------------------------------------------------
# 4. FUNÇÕES DE REFINAMENTO U-NET
# -----------------------------------------------------------------------------
def preprocess_for_unet(image_pil, target_size=(1052, 512)):
    """Preprocessa imagem para entrada no ResNet-UNet"""
    print(f"🔄 Preprocessando imagem para U-Net: {target_size}")
    
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
    Refina máscaras do YOLO usando ResNet-UNet
    Foca em mandíbula, maxila e dentes (ignora canal)
    """
    print("🔬 Iniciando refinamento com U-Net...")
    
    if unet_model is None:
        print("⚠️ Modelo U-Net não disponível, retornando máscaras YOLO originais")
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
            
            unet_probs = torch.softmax(unet_output, dim=1)
            unet_masks = torch.argmax(unet_probs, dim=1).squeeze(0).cpu().numpy()
            print(f"📊 Máscaras U-Net: {unet_masks.shape}, classes únicas: {np.unique(unet_masks)}")
        
        # Mapear classes U-Net (assumindo: 0=fundo, 1=mandíbula, 2=maxila, 3=dente, 4=canal)
        unet_class_map = {
            1: "jaw",      # mandíbula
            2: "maxila",   # maxila  
            3: "dente",    # dente
            # 4: canal ignorado por enquanto
        }
        
        refined_masks = []
        original_size = image_pil.size
        print(f"🔄 Processando refinamento para tamanho original: {original_size}")
        
        # Gerar máscaras refinadas das classes principais
        for class_id, class_name in unet_class_map.items():
            class_mask = (unet_masks == class_id).astype(np.uint8)
            pixel_count = class_mask.sum()
            
            print(f"🔍 Classe {class_name} (ID {class_id}): {pixel_count} pixels")
            
            if pixel_count > 100:  # Se há pixels suficientes desta classe
                # Redimensionar de volta para tamanho original
                class_mask_resized = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
                
                # Encontrar contornos
                contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"🔍 {len(contours)} contornos encontrados para {class_name}")
                
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 100:  # Filtrar contornos muito pequenos
                        # Converter contorno para formato esperado (normalizado)
                        contour_normalized = contour.reshape(-1, 2).astype(float)
                        contour_normalized[:, 0] /= original_size[0]  # normalizar x
                        contour_normalized[:, 1] /= original_size[1]  # normalizar y
                        
                        refined_masks.append({
                            "class_name": class_name,
                            "contour": contour_normalized * 100,  # formato esperado pelo frontend
                            "confidence": 0.85,  # confiança alta para refinamento
                            "area": area
                        })
                        print(f"✅ Contorno {i} de {class_name}: área={area:.0f}")
        
        print(f"🎉 Refinamento concluído: {len(refined_masks)} máscaras refinadas")
        return refined_masks
        
    except Exception as e:
        print(f"❌ Erro no refinamento U-Net: {e}")
        print(f"📝 Traceback: {traceback.format_exc()}")
        return []

async def run_model_prediction(model, image):
    print("🔄 Executando predição YOLO...")
    loop = asyncio.get_event_loop()
    predict_with_args = functools.partial(model.predict, source=image, conf=0.5)
    results = await loop.run_in_executor(None, predict_with_args)
    return results

# -----------------------------------------------------------------------------
# 5. ENDPOINT PRINCIPAL COM PIPELINE HÍBRIDO
# -----------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    print("\n" + "="*50)
    print("📡 Rota /analyze acessada!")
    print("="*50)
    
    yolo_model = lifespan_storage.get('yolo_model')
    unet_model = lifespan_storage.get('unet_model')
    device = lifespan_storage.get('device', 'cpu')
    
    if not yolo_model:
        raise HTTPException(status_code=500, detail="Modelo YOLO não está carregado.")
    
    try:
        print("📥 Lendo arquivo de imagem...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"📊 Imagem carregada: {image.size}")
        
        # ETAPA 1: Detecção inicial com YOLO
        print("\n🎯 ETAPA 1: Detecção YOLO")
        results_generator = await run_model_prediction(yolo_model, image)
        results = list(results_generator)
        
        all_findings = []
        
        if not results:
            print("⚠️ Nenhum resultado do YOLO.")
            return {"findings": []}

        prediction = results[0]
        class_names = prediction.names if hasattr(prediction, 'names') else {}
        
        print(f"📊 Classes disponíveis: {class_names}")

        # Processar detecções YOLO
        yolo_findings = []
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
                        
                        yolo_findings.append({
                            "id": f"yolo_finding_{i}",
                            "label": friendly_label,
                            "confidence": confidence,
                            "segmentation": (mask_points * 100).tolist()
                        })
                        print(f"✅ YOLO - {friendly_label}: {confidence:.3f}")
                        
                except Exception as e:
                    print(f"⚠️ Erro processando detecção YOLO {i}: {e}")
                    continue
        else:
            print("⚠️ Nenhuma máscara encontrada no resultado YOLO")

        # ETAPA 2: Refinamento com U-Net (se disponível)
        if unet_model:
            print(f"\n🔬 ETAPA 2: Refinamento U-Net")
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
                print(f"✅ U-Net - {friendly_label} (Refinado): área={refined_mask['area']:.0f}")
        else:
            print("⚠️ U-Net não disponível, usando apenas YOLO")
        
        # Adicionar detecções YOLO originais
        all_findings.extend(yolo_findings)
        
        mode = "Híbrido YOLO + U-Net" if unet_model else "Apenas YOLO"
        print(f"\n🎉 Análise concluída [{mode}]")
        print(f"📊 Total de achados: {len(all_findings)}")
        print("="*50)
        
        return {"findings": all_findings}
        
    except Exception as e:
        print(f"❌ Erro crítico na análise: {e}")
        print(f"📝 Traceback completo:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor ao analisar imagem: {str(e)}")

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
