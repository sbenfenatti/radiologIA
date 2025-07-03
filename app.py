# app.py - Versão final com múltiplas senhas
import os
import requests
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

# --- CARREGAMENTO DOS SECRETS ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("❌ AVISO: Secret 'GEMINI_API_KEY' não encontrado.")

APP_PASSWORD = os.getenv('APP_PASSWORD')
if not APP_PASSWORD:
    print("❌ AVISO: Secret 'APP_PASSWORD' não encontrado.")

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
# 3. (MODIFICADO) ROTA DE VERIFICAÇÃO DE SENHA
# -----------------------------------------------------------------------------
@app.route('/verify-password', methods=['POST'])
def verify_password():
    print("\n📡 Rota /verify-password acessada!")
    try:
        data = request.get_json()
        submitted_password = data.get('password')

        if not submitted_password:
            return jsonify({"success": False, "error": "Senha não fornecida."}), 400
        
        # --- LÓGICA PARA MÚLTIPLAS SENHAS ---
        # 1. Pega a string de senhas do secret (ex: "rad2025,admin,guest")
        all_passwords_str = os.getenv('APP_PASSWORD', '')
        
        # 2. Transforma a string em uma lista de senhas válidas, removendo espaços extras.
        #    Ex: ["rad2025", "admin", "guest"]
        valid_passwords = [p.strip() for p in all_passwords_str.split(',') if p.strip()]

        # 3. Verifica se a senha enviada pelo usuário está DENTRO da lista de senhas válidas.
        if submitted_password in valid_passwords:
            print(f"✅ Senha válida ('{submitted_password}') recebida. Acesso permitido.")
            return jsonify({"success": True}), 200
        else:
            print(f"❌ Senha inválida ('{submitted_password}') recebida. Acesso negado.")
            return jsonify({"success": False, "error": "Senha incorreta."}), 401

    except Exception as e:
        print(f"❌ Erro na verificação de senha: {e}")
        return jsonify({"error": "Erro interno do servidor"}), 500

# -----------------------------------------------------------------------------
# 4. ROTA DA API DE ANÁLISE DE IMAGEM
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
# 5. ROTA SEGURA PARA O CHAT COM GEMINI
# -----------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def handle_chat():
    print("\n📡 Rota /chat acessada!")
    if not GEMINI_API_KEY:
        return jsonify({"error": "A chave da API do Gemini não está configurada no servidor."}), 500

    try:
        data = request.json
        chat_history = data.get('history', []) 

        headers = {'Content-Type': 'application/json'}
        payload = {'contents': chat_history}

        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

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
# 6. ROTAS PARA SERVIR O SITE (FRONT-END)
# -----------------------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print("Iniciando servidor localmente na porta 5001...")
    app.run(debug=True, host='0.0.0.0', port=5001)