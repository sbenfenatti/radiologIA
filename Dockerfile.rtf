{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww12720\viewh7800\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Define a imagem base do Python que vamos usar\
FROM python:3.10-slim\
\
# Define o diret\'f3rio de trabalho dentro do container\
WORKDIR /code\
\
# Copia o arquivo de requerimentos para o container\
COPY ./requirements.txt /code/requirements.txt\
\
# Instala as bibliotecas Python listadas no requirements.txt\
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt\
\
# Copia todo o resto do seu projeto (app.py, models/, static/) para o container\
COPY . /code/\
\
# Exp\'f5e a porta que o Hugging Face usar\'e1 para se comunicar com a nossa aplica\'e7\'e3o\
EXPOSE 7860\
\
# O comando final que inicia o servidor Gunicorn quando o container rodar\
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]\
}