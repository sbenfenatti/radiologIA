import modal
import io
import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# 1. Definição da Imagem (Ambiente Linux + Dependências)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "Pillow",
        "shapely",
        "ultralytics",
        "google-generativeai" # SDK do Google AI
    )
    .pip_install("git+https://github.com/facebookresearch/detectron2.git")
)

volume = modal.Volume.from_name("radiologia-modelos")
app = modal.App("radiologia-backend-prod")

# Modelo de dados para o Chat
class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]
    context: str = "" 

@app.cls(
    image=image, 
    gpu="T4", 
    volumes={"/models": volume}, 
    scaledown_window=300,
    secrets=[modal.Secret.from_name("my-google-secret")] # Requer a chave GOOGLE_API_KEY
)
class RadiologiaBackend:
    
    @modal.enter()
    def load_models(self):
        import torch
        from ultralytics import YOLO
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
        import google.generativeai as genai

        print("🔄 Carregando modelos RadiologIA 2026...")
        
        # --- CONFIGURAÇÃO GEMINI 3 ---
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            # Usando a família 3.0 conforme documentação
            self.gemini_model = genai.GenerativeModel('gemini-3-flash-preview') 
            print("✅ Gemini 3 Flash ativado.")
        else:
            print("⚠️ AVISO: GOOGLE_API_KEY não encontrada.")
            self.gemini_model = None

        # --- CARREGAR YOLO ---
        try:
            self.yolo = YOLO("/models/best.pt")
            print("✅ YOLO carregado.")
        except Exception as e:
            print(f"❌ Erro YOLO: {e}")
            self.yolo = None

        # --- CARREGAR MASK R-CNN ---
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = "/models/model.pth"
            # Configurações do seu treino (5 classes)
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 3.0]]
            cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.predictor_mask = DefaultPredictor(cfg)
            print("✅ Mask R-CNN carregado.")
        except Exception as e:
            print(f"❌ Erro Mask R-CNN: {e}")
            self.predictor_mask = None

    def process_image(self, contents):
        nparr = np.frombuffer(contents, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @modal.asgi_app()
    def fastapi_app(self):
        web_app = FastAPI(title="RadiologIA Backend 2.0")
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Endpoint de Análise (Visão)
        @web_app.post("/analyze")
        async def analyze_endpoint(file: UploadFile = File(...), model_type: str = Form("yolo")):
            contents = await file.read()
            img = self.process_image(contents)
            if img is None: raise HTTPException(400, "Imagem inválida")
            
            if model_type == "yolo": return self.run_yolo(img)
            elif model_type == "mask_rcnn": return self.run_mask_rcnn(img)
            elif model_type == "full_integration": return self.run_integration(img)
            return {"error": "Modelo desconhecido"}

        # Endpoint de Chat (Gemini 3)
        @web_app.post("/chat")
        async def chat_endpoint(request: ChatRequest):
            if not self.gemini_model:
                raise HTTPException(503, "Gemini não configurado (Falta API Key).")
            
            try:
                chat_history = []
                
                # Prompt de Sistema Odontológico
                if request.context:
                    system_prompt = f"""
                    Você é a RadiologIA, uma IA assistente odontológica avançada (Versão 2026).
                    O usuário enviou uma radiografia com os seguintes achados detectados pelos modelos de visão:
                    {request.context}
                    
                    Responda com precisão técnica de um radiologista/dentista.
                    Seja direto, profissional e use termos técnicos corretos (mesial, distal, radiolúcido, radiopaco).
                    """
                    chat_history.append({"role": "user", "parts": [system_prompt]})
                    chat_history.append({"role": "model", "parts": ["Compreendido. Analisarei o caso com base nestes achados."]})

                for msg in request.history:
                    role = "user" if msg.role == "user" else "model"
                    chat_history.append({"role": role, "parts": [msg.text]})

                chat = self.gemini_model.start_chat(history=chat_history[:-1])
                response = chat.send_message(chat_history[-1]["parts"][0])
                
                return {"response": response.text}
            except Exception as e:
                print(f"Erro Gemini: {e}")
                raise HTTPException(500, "Erro ao processar resposta do Gemini.")
        
        return web_app

    def run_yolo(self, img):
        if not self.yolo: return {"error": "YOLO offline"}
        results = self.yolo(img)
        findings = []
        for result in results:
            for box in result.boxes:
                findings.append({
                    "model": "YOLO",
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
        return {"findings": findings}

    def run_mask_rcnn(self, img):
        if not self.predictor_mask: return {"error": "Mask R-CNN offline"}
        outputs = self.predictor_mask(img)
        instances = outputs["instances"].to("cpu")
        findings = []
        classes_map = {0: "dente", 1: "dentina", 2: "polpa", 3: "restauracao", 4: "carie"}
        for i, cls_id in enumerate(instances.pred_classes.numpy()):
            mask = instances.pred_masks.numpy()[i].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            seg = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist() if contours else []
            findings.append({
                "model": "Mask R-CNN", 
                "label": classes_map.get(cls_id, "desc"), 
                "segmentation": seg
            })
        return {"findings": findings}

    def run_integration(self, img):
        return {"message": "Em construção"}