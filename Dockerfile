# Usa uma imagem Python mais completa para garantir compatibilidade
FROM python:3.10

# Cria e define o diretório de trabalho
WORKDIR /app

# Atualiza o pip para a versão mais recente
RUN pip install --upgrade pip

# Copia o arquivo de requerimentos ANTES de todo o resto
COPY requirements.txt .

# Instala as dependências de forma robusta
RUN pip install --no-cache-dir -r requirements.txt

# Agora, copia o resto do código da sua aplicação
COPY . .

# Expõe a porta que o Hugging Face usa
EXPOSE 7860

# Comando final para iniciar o servidor, usando o caminho completo para o python do ambiente
CMD ["/usr/local/bin/python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
