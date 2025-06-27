# app.py - Versão corrigida e segura para Hugging Face Spaces
import os
import requests  # Importa a biblioteca para fazer chamadas HTTP
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
CORS(app)
print("Servidor Flask inicializado.")

# Carrega o segredo da API do Gemini a partir das variáveis de ambiente do HF Spaces
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ AVISO: A variável de ambiente 'GEMINI_API_KEY' não foi encontrada.")
    print("➡️ Adicione-a nos 'Secrets' das configurações do seu Space no Hugging Face.")

# URL da API do Gemini
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# -----------------------------------------------------------------------------
# 2. CARREGAMENTO DO MODELO YOLO
# -----------------------------------------------------------------------------
try:
    model = YOLO('models/best.pt')
    print("✅ Modelo de Segmentação YOLO (best.pt) carregado com sucesso.")
except Exception as e:
    print(f"❌ Erro fatal ao carregar o modelo YOLO: {e}")
    model = None

# -----------------------------------------------------------------------------
# 3. ROTA DA API DE ANÁLISE DE IMAGEM
# -----------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("\n📡 Rota /analyze acessada!")
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
        
        results = model.predict(source=image, conf=0.5)
        all_findings = []
        prediction = results[0]
        class_names = prediction.names

        if prediction.masks:
            for i, box in enumerate(prediction.boxes):
                if i < len(prediction.masks.xyn):
                    mask_points = prediction.masks.xyn[i]
                    all_findings.append({
                        "id": f"finding_{i}",
                        "label": class_names.get(int(box.cls[0].item()), "Desconhecido"),
                        "confidence": box.conf[0].item(),
                        "segmentation": (np.array(mask_points) * 100).tolist()
                    })
        print(f"✅ Análise de imagem concluída. Enviando {len(all_findings)} achados.")
        return jsonify({"findings": all_findings})

    except Exception as e:
        import traceback
        print(f"❌ Erro Crítico na análise: {e}")
        traceback.print_exc()
        return jsonify({"error": "Erro interno do servidor ao analisar imagem"}), 500

# -----------------------------------------------------------------------------
# 4. NOVA ROTA SEGURA PARA O CHAT COM GEMINI
# -----------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def handle_chat():
    print("\n📡 Rota /chat acessada!")
    if not GEMINI_API_KEY:
        return jsonify({"error": "A chave da API do Gemini não está configurada no servidor."}), 500

    try:
        data = request.json
        # O histórico de chat agora é gerenciado pelo frontend e enviado na requisição
        chat_history = data.get('history', []) 

        # A chamada para a API do Gemini agora acontece aqui, no backend
        headers = {'Content-Type': 'application/json'}
        payload = {'contents': chat_history}

        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Lança um erro para respostas HTTP 4xx/5xx

        print("✅ Resposta da API Gemini recebida com sucesso.")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Erro ao chamar a API do Gemini: {e}")
        return jsonify({"error": f"Erro de comunicação com a API do Gemini: {e}"}), 503
    except Exception as e:
        import traceback
        print(f"❌ Erro Crítico no chat: {e}")
        traceback.print_exc()
        return jsonify({"error": "Erro interno do servidor no processamento do chat"}), 500

# -----------------------------------------------------------------------------
# 5. ROTAS PARA SERVIR O SITE (FRONT-END)
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
    # Para teste local, você pode criar um arquivo .env com a chave ou defini-la manualmente
    # Ex: export GEMINI_API_KEY='sua_chave_aqui'
    app.run(debug=True, host='0.0.0.0', port=5001)