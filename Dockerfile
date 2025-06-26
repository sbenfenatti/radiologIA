# Define a imagem base do Python que vamos usar
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /code

# Copia o arquivo de requerimentos para o container
COPY ./requirements.txt /code/requirements.txt

# Instala as bibliotecas Python listadas no requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia todo o resto do seu projeto (app.py, models/, static/) para o container
COPY . /code/

# Expõe a porta que o Hugging Face usará para se comunicar com a nossa aplicação
EXPOSE 7860

# O comando final que inicia o servidor Gunicorn quando o container rodar
# CORREÇÃO: Usamos "python -m gunicorn" para garantir que o executável seja encontrado.
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
