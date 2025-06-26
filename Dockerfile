# Define a imagem base do Python que vamos usar
FROM python:3.10

# --- CORREÇÃO PRINCIPAL ---
# Instala a dependência do sistema (libGL) que estava faltando para o OpenCV funcionar.
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Cria e define o diretório de trabalho
WORKDIR /app

# Atualiza o pip para a versão mais recente
RUN pip install --upgrade pip

# Copia o arquivo de requerimentos para o container
COPY requirements.txt .

# Instala as bibliotecas Python listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do seu projeto (app.py, models/, static/) para o container
COPY . .

# Expõe a porta que o Hugging Face usará para se comunicar com a nossa aplicação
EXPOSE 7860

# O comando final que inicia o servidor Gunicorn quando o container rodar
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
