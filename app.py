import os
import io
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import functools
import httpx
from ultralytics import YOLO
import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, closing, disk
import warnings
warnings.filterwarnings('ignore')

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Usando dispositivo: {DEVICE}")

# Storage global para modelos
lifespan_storage = {}

def resize_image_for_yolo(image, max_size=1024):
    """Redimensiona imagem mantendo aspect ratio para otimizar YOLO"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    
    # Calcula novo tamanho mantendo proporção
    if w > h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_w, new_h = int(w * max_size / h), max_size
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    print(f"📏 Imagem redimensionada de {image.size} para {resized.size}")
    return resized

def debug_unet_model(model, device):
    """Debug completo do modelo U-Net"""
    print("🧪 === DEBUG U-NET ===")
    
    try:
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        print(f"🔧 Testando com input: {dummy_input.shape}")
        
        with torch.no_grad():
            dummy_output = model(dummy_input)
        
        print(f"✅ Output shape: {dummy_output.shape}")
        print(f"📊 Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
        
        # Verificar probabilidades por classe
        probs = torch.softmax(dummy_output, dim=1)
        class_probs = []
        for i in range(probs.shape[1]):
            mean_prob = probs[0, i].mean().item()
            class_probs.append(mean_prob)
            print(f"📈 Classe {i}: probabilidade média = {mean_prob:.4f}")
        
        print(f"📊 Probabilidades por classe: {torch.tensor(class_probs)}")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
    
    print("🧪 === FIM DEBUG ===")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando a aplicação...")
    
    # Carregar YOLO
    print("📥 Carregando modelo YOLO...")
    yolo_path = "models/best.pt"
    if os.path.exists(yolo_path):
        lifespan_storage['yolo_model'] = YOLO(yolo_path)
        print("✅ Modelo YOLO carregado com sucesso.")
    else:
        print("❌ Modelo YOLO não encontrado!")
        lifespan_storage['yolo_model'] = None
    
    # Carregar U-Net
    print("📥 Tentando carregar modelo U-Net...")
    unet_path = "models/radiologia_5classes_fold_1_best.pth"
    
    if os.path.exists(unet_path):
        print(f"📂 Arquivo U-Net encontrado: {unet_path}")
        try:
            checkpoint = torch.load(unet_path, map_location=DEVICE)
            print(f"🔍 Tipo do checkpoint: {type(checkpoint)}")
            print(f"🔍 Chaves do checkpoint: {list(checkpoint.keys())[:20]}...")  # Primeiras 20 chaves
            
            # Tentar importar segmentation_models_pytorch
            try:
                import segmentation_models_pytorch as smp
                print("✅ Segmentation Models PyTorch importado com sucesso")
                
                # Usar ResNet34 + U-Net (como no treinamento)
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights=None,  # Sem pré-treino, usar pesos treinados
                    in_channels=3,
                    classes=5,
                    decoder_channels=[256, 128, 64, 32, 16]
                )
                print("✅ Modelo U-Net ResNet34 criado")
                
            except ImportError:
                print("⚠️ segmentation_models_pytorch não disponível, usando U-Net simples")
                # Fallback para U-Net simples se smp não estiver disponível
                from models.unet_model import UNet
                model = UNet(n_channels=3, n_classes=5)
            
            # Carregar pesos
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Estado do modelo carregado de 'model_state_dict'")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("📝 Usando checkpoint diretamente")
            
            model.to(DEVICE)
            model.eval()
            lifespan_storage['unet_model'] = model
            
            # Debug do modelo
            debug_unet_model(model, DEVICE)
            
            print("✅ Modelo U-Net carregado com sucesso.")
            
        except Exception as e:
            print(f"❌ Erro ao carregar U-Net: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            lifespan_storage['unet_model'] = None
    else:
        print("❌ Arquivo U-Net não encontrado!")
        lifespan_storage['unet_model'] = None
    
    # Cliente HTTP
    lifespan_storage['http_client'] = httpx.AsyncClient()
    print("✅ Cliente HTTP assíncrono criado.")
    
    print("🎉 Aplicação iniciada com sucesso!")
    yield
    
    # Cleanup
    await lifespan_storage['http_client'].aclose()
    print("🧹 Limpeza concluída.")

app = FastAPI(lifespan=lifespan)

def analyze_with_yolo(image, yolo_model):
    """Análise YOLO otimizada com timing detalhado"""
    total_start = time.time()
    print(f"🎯 Iniciando análise com YOLO... Tamanho da imagem: {image.size}")
    
    if yolo_model is None:
        return [], {"error": "Modelo YOLO não disponível"}
    
    try:
        # 1. CORREÇÃO: Redimensionar imagem SEMPRE que for grande
        preprocess_start = time.time()
        original_size = image.size
        image = resize_image_for_yolo(image, max_size=1024)  # ATIVADO!
        preprocess_time = time.time() - preprocess_start
        
        # 2. Predição otimizada
        inference_start = time.time()
        results = yolo_model.predict(
            source=image, 
            conf=0.5,
            device=DEVICE.type,
            half=False,
            verbose=False,
            imgsz=640,
            max_det=100
        )
        inference_time = time.time() - inference_start
        
        print(f"⏱️ Tempo de inferência YOLO: {inference_time:.2f}s")
        
        # 3. Pós-processamento
        postprocess_start = time.time()
        
        class_names = {
            0: 'Canal Mandibular', 1: 'Canino Inferior', 2: 'Canino Superior',
            3: 'Cárie', 4: 'Restauração', 5: 'Impactado',
            6: 'Incisivo Central Inferior', 7: 'Incisivo Central Superior',
            8: 'Incisivo Lateral Inferior', 9: 'Incisivo Lateral Superior',
            10: 'Mandíbula', 11: 'Maxila', 12: 'Molar Inferior',
            13: 'Molar Superior', 14: 'Lesão Periapical', 15: 'Pré-Molar Inferior',
            16: 'Pré-Molar Superior', 17: 'Resto Residual', 18: 'Terceiro  Molar Inf',
            19: 'Terceiro Molar Superior', 20: 'Tto Endo'
        }
        
        findings = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                print(f"🔍 Processando {len(masks)} máscaras YOLO...")
                
                for i, (mask, box, conf, cls) in enumerate(zip(masks, boxes, confidences, classes)):
                    class_name = class_names.get(cls, f'Classe {cls}')
                    
                    if conf >= 0.5:
                        findings.append({
                            "type": "YOLO",
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "area": int(np.sum(mask > 0))
                        })
                        print(f"✅ YOLO - {class_name}: {conf:.3f}")
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - total_start
        
        # Timing info
        timing = {
            "preprocess_ms": preprocess_time * 1000,
            "inference_ms": inference_time * 1000,
            "postprocess_ms": postprocess_time * 1000,
            "total_ms": total_time * 1000,
            "original_size": original_size,
            "processed_size": image.size
        }
        
        print(f"⏱️ Tempo total YOLO: {total_time:.2f}s (inferência: {inference_time:.2f}s, pós-processamento: {postprocess_time:.2f}s)")
        
        return findings, timing
        
    except Exception as e:
        print(f"❌ Erro na análise YOLO: {e}")
        return [], {"error": str(e)}

def analyze_with_unet(image, unet_model):
    """Análise U-Net com arquitetura corrigida"""
    total_start = time.time()
    print("🔬 Iniciando análise com U-Net otimizada...")
    
    if unet_model is None:
        return [], {"error": "Modelo U-Net não disponível"}
    
    try:
        # 1. Preprocessamento
        preprocess_start = time.time()
        original_size = image.size
        image_resized = image.resize((512, 512), Image.Resampling.LANCZOS)
        print(f"🔄 Preprocessando imagem para U-Net: {image_resized.size}")
        
        image_array = np.array(image_resized) / 255.0
        
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        
        tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        preprocess_time = time.time() - preprocess_start
        print(f"📊 Tensor de entrada: {tensor.shape} ({preprocess_time:.2f}s)")
        
        # 2. Inferência
        inference_start = time.time()
        with torch.no_grad():
            outputs = unet_model(tensor)
        inference_time = time.time() - inference_start
        print(f"📊 Saída U-Net: {outputs.shape} ({inference_time:.2f}s)")
        
        # 3. Pós-processamento
        postprocess_start = time.time()
        
        # Aplicar softmax para obter probabilidades
        probs = torch.softmax(outputs, dim=1)
        
        # Log das probabilidades médias por classe
        mean_probs = []
        for i in range(probs.shape[1]):
            mean_prob = probs[0, i].mean().item()
            mean_probs.append(mean_prob)
        print(f"📊 Probabilidades médias por classe: {torch.tensor(mean_probs)}")
        
        # CORREÇÃO DO BUG: Usar sintaxe correta do PyTorch
        try:
            # Calcular probabilidade máxima por classe
            max_probs = torch.max(torch.max(probs, dim=3)[0], dim=2)[0]  # CORRIGIDO!
            print(f"📊 Probabilidades máximas por classe: {max_probs}")
        except Exception as e:
            print(f"⚠️ Erro ao calcular max_probs, usando médias: {e}")
            max_probs = torch.tensor(mean_probs).to(DEVICE)
        
        # Obter predições (classe com maior probabilidade)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()[0]
        print(f"🎯 Classes únicas preditas: {np.unique(predictions)}")
        
        # Redimensionar para tamanho original
        predictions_resized = cv2.resize(
            predictions.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Nomes das classes U-Net
        class_names = {
            0: 'Background',
            1: 'Mandíbula', 
            2: 'Maxila',
            3: 'Dente',
            4: 'Canal'
        }
        
        findings = []
        
        # Processar cada classe (exceto background)
        for class_id in range(1, 5):
            mask = (predictions_resized == class_id)
            pixel_count = np.sum(mask)
            
            if pixel_count > 0:
                # Usar probabilidade média ou máxima
                if class_id < len(mean_probs):
                    class_confidence = mean_probs[class_id]
                else:
                    class_confidence = 0.1
                
                print(f"🔍 Classe {class_names[class_id]} (ID {class_id}): "
                      f"{pixel_count} pixels, confiança={class_confidence:.3f}")
                
                # Threshold mais alto para melhor qualidade
                if class_confidence > 0.15 and pixel_count > 2000:  # Aumentado para 0.15
                    findings.append({
                        "type": "U-Net",
                        "class": class_names[class_id],
                        "confidence": float(class_confidence),
                        "area": int(pixel_count),
                        "bbox": [0, 0, original_size[0], original_size[1]]
                    })
                    print(f"✅ {class_names[class_id]}: confiança={class_confidence:.3f}, "
                          f"área={pixel_count}")
                else:
                    print(f"⚪ {class_names[class_id]}: confiança muito baixa ou área pequena")
            else:
                print(f"⚪ Classe {class_names[class_id]} (ID {class_id}): não detectada")
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - total_start
        
        timing = {
            "preprocess_ms": preprocess_time * 1000,
            "inference_ms": inference_time * 1000,
            "postprocess_ms": postprocess_time * 1000,
            "total_ms": total_time * 1000,
            "original_size": original_size,
            "processed_size": (512, 512)
        }
        
        print(f"🎉 Análise U-Net otimizada concluída: {len(findings)} achados")
        
        return findings, timing
        
    except Exception as e:
        print(f"❌ Erro na análise U-Net: {e}")
        import traceback
        print(f"📝 Traceback: {traceback.format_exc()}")
        return [], {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Interface web simples"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RadiologIA - Análise Radiológica com IA</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .error { border-left-color: #dc3545; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦷 RadiologIA - Análise Radiológica Otimizada</h1>
            <p>Faça upload de uma radiografia panorâmica para análise com IA.</p>
            
            <form id="analysisForm" enctype="multipart/form-data">
                <div>
                    <label>Selecione a imagem:</label><br>
                    <input type="file" name="file" accept="image/*" required><br><br>
                </div>
                
                <div>
                    <label>Modelo de análise:</label><br>
                    <select name="model_type">
                        <option value="yolo">YOLO (Detecção de estruturas)</option>
                        <option value="unet">U-Net (Segmentação)</option>
                    </select><br><br>
                </div>
                
                <button type="submit">🔍 Analisar Imagem</button>
            </form>
            
            <div id="results"></div>
        </div>

        <script>
            document.getElementById('analysisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const resultsDiv = document.getElementById('results');
                
                resultsDiv.innerHTML = '<div class="result loading">🔄 Analisando imagem... Isso pode levar alguns segundos.</div>';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = `<div class="result">
                            <h3>✅ Análise concluída!</h3>
                            <p><strong>Modelo:</strong> ${data.model_used}</p>
                            <p><strong>Achados:</strong> ${data.findings.length}</p>
                        `;
                        
                        if (data.timing_info) {
                            html += `<p><strong>Tempo total:</strong> ${(data.timing_info.total_ms/1000).toFixed(2)}s</p>`;
                        }
                        
                        if (data.findings.length > 0) {
                            html += '<h4>Estruturas detectadas:</h4><ul>';
                            data.findings.forEach(finding => {
                                html += `<li><strong>${finding.class}</strong> - Confiança: ${(finding.confidence * 100).toFixed(1)}%</li>`;
                            });
                            html += '</ul>';
                        }
                        
                        html += '</div>';
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = `<div class="result error">❌ Erro: ${data.error}</div>`;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="result error">❌ Erro de comunicação: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/models/available")
async def get_available_models():
    """Retorna status dos modelos disponíveis"""
    return {
        "yolo": lifespan_storage.get('yolo_model') is not None,
        "unet": lifespan_storage.get('unet_model') is not None,
        "device": str(DEVICE)
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), model_type: str = Form("yolo")):
    """Endpoint principal de análise otimizado"""
    
    print("=" * 50)
    print(f"📡 Rota /analyze acessada com modelo: {model_type.upper()}")
    print("=" * 50)
    
    try:
        # Ler arquivo
        print("📥 Lendo arquivo de imagem...")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"📊 Imagem carregada: {image.size}")
        
        # Selecionar modelo e executar análise
        if model_type.lower() == "yolo":
            print("🎯 Executando análise com YOLO")
            yolo_model = lifespan_storage.get('yolo_model')
            findings, timing_info = analyze_with_yolo(image, yolo_model)
            model_used = "YOLO"
            
        elif model_type.lower() == "unet":
            print("🔬 Executando análise com U-Net")
            unet_model = lifespan_storage.get('unet_model')
            findings, timing_info = analyze_with_unet(image, unet_model)
            model_used = "U-Net"
            
        else:
            raise HTTPException(status_code=400, detail="Modelo não suportado")
        
        print(f"🎉 Análise concluída com {model_used}")
        print(f"📊 Total de achados: {len(findings)}")
        if 'total_ms' in timing_info:
            print(f"⏱️ Tempo total: {timing_info['total_ms']/1000:.2f}s")
        print("=" * 50)
        
        return JSONResponse({
            "success": True,
            "model_used": model_used,
            "findings": findings,
            "timing_info": timing_info,
            "total_findings": len(findings)
        })
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
