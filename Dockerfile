# Define a imagem base do Python que vamos usar
FROM python:3.10

# --- CORREÇÃO PRINCIPAL ---
# Instala a dependência do sistema (libgl1) para o OpenCV.
# O nome do pacote 'libgl1-mesa-glx' foi atualizado para 'libgl1' na versão base do SO.
RUN apt-get update && apt-get install -y libgl1

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

# Comando para iniciar o servidor Uvicorn com o FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
