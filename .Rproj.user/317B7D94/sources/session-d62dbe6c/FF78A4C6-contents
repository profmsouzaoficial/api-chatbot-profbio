# Usa uma versão leve e oficial do Python
FROM python:3.11-slim

# Define a pasta de trabalho no servidor
WORKDIR /app

# Instala dependências do sistema operativo que o PyMuPDF pode precisar
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copia e instala as bibliotecas do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia TUDO do seu projeto para o servidor (incluindo o banco_vetores já processado!)
COPY . .

# Expõe a porta 8000
EXPOSE 8000

# Comando para ligar a API (Repare que usa o nome do seu ficheiro: chatbot_api)
CMD ["uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]