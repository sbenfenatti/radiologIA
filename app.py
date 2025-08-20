# app.py - Versão com FastAPI (Suavização Removida)
import os
import io
import asyncio
import functools 
import traceback 
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from ultralytics import YOLO
# A dependência 'scipy' foi removida pois não é mais necessária
import httpx


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List


# -----------------------------------------------------------------------------
# 1. DEFINIÇÃO DOS MODELOS DE DADOS (PYDANTIC)
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


# Modelo para verificação de senha
class PasswordRequest(BaseModel):
    password: str


# -----------------------------------------------------------------------------
# 2. CONFIGURAÇÃO INICIAL E CICLO DE VIDA DA APLICAÇÃO
# -----------------------------------------------------------------------------
lifespan_storage = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando a aplicação...")
    lifespan_storage['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
    if not lifespan_storage['gemini_api_key']:
        print("❌ AVISO: A variável de ambiente 'GEMINI_API_KEY' não foi encontrada.")
        print("➡️ Adicione-a nos 'Secrets' das configurações do seu Space no Hugging Face.")

    # Configurar senha da aplicação
    lifespan_storage['app_password'] = os.getenv('APP_PASSWORD', 'radiologia2024')
    print(f"✅ Senha da aplicação configurada.")

    try:
        lifespan_storage['yolo_model'] = YOLO('models/best.pt')
        print("✅ Modelo de Segmentação YOLO (best.pt) carregado com sucesso.")
    except Exception as e:
        print(f"❌ Erro fatal ao carregar o modelo YOLO: {e}")
        lifespan_storage['yolo_model'] = None
    lifespan_storage['http_client'] = httpx.AsyncClient()
    print("✅ Cliente HTTP assíncrono criado.")
    yield
    print("👋 Encerrando a aplicação...")
    await lifespan_storage['http_client'].aclose()
    print("✅ Cliente HTTP assíncrono fechado.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DICIONÁRIO DE TRADUÇÕES EXPANDIDO ---
# Mapeia os nomes técnicos do modelo para nomes completos em português.
LABEL_MAP = {
    # Mapeamentos originais
    "lesao_periapical": "Lesão Periapical",
    "carie": "Cárie",
    "fratura_radicular": "Fratura Radicular",
    "calculo_dental": "Cálculo Dental",
    "restauracao": "Restauração",
    "implante": "Implante",
    "dente_incluso": "Dente Incluso",


    # Mapeamentos adicionados com base na sua imagem e possíveis variações
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
# 3. FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
# A função 'smooth_segmentation' foi removida.


async def run_model_prediction(model, image):
    loop = asyncio.get_event_loop()
    predict_with_args = functools.partial(model.predict, source=image, conf=0.5)
    results = await loop.run_in_executor(None, predict_with_args)
    return results


# -----------------------------------------------------------------------------
# 4. ENDPOINTS DA API
# -----------------------------------------------------------------------------

@app.post("/verify-password")
async def verify_password(payload: PasswordRequest):
    """Endpoint para verificar a senha de acesso ao protótipo"""
    print("\n🔐 Rota /verify-password acessada!")
    try:
        correct_password = lifespan_storage.get('app_password')
        user_password = payload.password.strip()

        print(f"🔍 Verificando senha... (tamanho: {len(user_password)} caracteres)")

        if user_password == correct_password:
            print("✅ Senha correta! Acesso liberado.")
            return {"success": True}
        else:
            print("❌ Senha incorreta.")
            return {"success": False}

    except Exception as e:
        print(f"❌ Erro na verificação de senha: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro interno do servidor na verificação de senha")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    print("\n📡 Rota /analyze acessada!")
    model = lifespan_storage.get('yolo_model')
    if not model:
        raise HTTPException(status_code=500, detail="Modelo YOLO não está carregado.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results_generator = await run_model_prediction(model, image)
        results = list(results_generator)

        all_findings = []
        if not results:
            print("⚠️ Nenhum resultado retornado pelo modelo.")
            return {"findings": []}


        prediction = results[0]
        class_names = prediction.names


        if prediction.masks:
            for i, box in enumerate(prediction.boxes):
                if i < len(prediction.masks.xyn):
                    technical_label = class_names.get(int(box.cls[0].item()), "Desconhecido")
                    friendly_label = LABEL_MAP.get(technical_label, technical_label.replace("_", " ").title())

                    # Usa diretamente os pontos originais da máscara fornecidos pelo modelo
                    mask_points = prediction.masks.xyn[i]

                    all_findings.append({
                        "id": f"finding_{i}",
                        "label": friendly_label,
                        "confidence": box.conf[0].item(),
                        # Envia os pontos originais, sem suavização
                        "segmentation": (mask_points * 100).tolist()
                    })
        print(f"✅ Análise de imagem concluída. Enviando {len(all_findings)} achados.")
        return {"findings": all_findings}
    except Exception as e:
        print(f"❌ Erro Crítico na análise: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Erro interno do servidor ao analisar imagem")


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
# 5. SERVINDO ARQUIVOS ESTÁTICOS (FRONT-END)
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