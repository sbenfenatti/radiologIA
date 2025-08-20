# Define a imagem base do Python que vamos usar
FROM python:3.10

# Instala a dependência do sistema (libGL) para o OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Cria e define o diretório de trabalho
WORKDIR /app

# Atualiza o pip
RUN pip install --upgrade pip

# Copia o arquivo de requerimentos
COPY requirements.txt .

# Instala as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do projeto para o container
COPY . .

# Expõe a porta que a aplicação usará
EXPOSE 7860

# --- MUDANÇA PRINCIPAL ---
# Comando para iniciar o servidor Uvicorn com o FastAPI
# --host 0.0.0.0: permite conexões externas
# --port 7860: porta exposta
# app:app: refere-se ao objeto 'app' dentro do arquivo 'app.py'
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
