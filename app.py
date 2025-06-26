# app.py - Versão para Hugging Face Spaces
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import torch
from ultralytics import YOLO
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO INICIAL
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder='static')
CORS(app) # Configuração CORS simplificada para rodar no mesmo domínio
print("Servidor Flask inicializado.")

# -----------------------------------------------------------------------------
# 2. CARREGAMENTO DO MODELO
# -----------------------------------------------------------------------------
try:
    model = YOLO('models/best.pt')
    print("✅ Modelo de Segmentação YOLO (best.pt) carregado com sucesso.")
except Exception as e:
    print(f"❌ Erro fatal ao carregar o modelo YOLO: {e}")
    model = None

# -----------------------------------------------------------------------------
# 3. ROTA DA API DE ANÁLISE
# -----------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("\n📡 Rota /analyze acessada!")
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
        
        # Lógica de análise
        results = model.predict(source=image, conf=0.5)
        all_findings = []
        prediction = results[0]
        class_names = prediction.names

        if prediction.masks:
            for i, box in enumerate(prediction.boxes):
                if i < len(prediction.masks.xyn): # Garante que o índice existe
                    mask_points = prediction.masks.xyn[i]
                    all_findings.append({
                        "id": f"finding_{i}",
                        "label": class_names.get(int(box.cls[0].item()), "Desconhecido"),
                        "confidence": box.conf[0].item(),
                        "segmentation": (np.array(mask_points) * 100).tolist()
                    })
        print(f"✅ Análise concluída. Enviando {len(all_findings)} achados.")
        return jsonify({"findings": all_findings})

    except Exception as e:
        import traceback
        print(f"❌ Erro Crítico: {e}")
        traceback.print_exc()
        return jsonify({"error": "Erro interno do servidor"}), 500

# -----------------------------------------------------------------------------
# 4. ROTAS PARA SERVIR O SITE (FRONT-END)
# -----------------------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# O __main__ não é necessário para o Hugging Face, mas mantemos para teste local
if __name__ == '__main__':
    print("Iniciando servidor localmente na porta 5001...")
    app.run(debug=True, host='0.0.0.0', port=5001)

