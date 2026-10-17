# Usa uma versão super leve do Python
FROM python:3.11-slim

# Define a pasta de trabalho no servidor
WORKDIR /app

# Copia e instala as bibliotecas otimizadas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a API e o banco_vetores gerado no seu PC para o servidor
COPY . .

# Expõe a porta 8000
EXPOSE 8000

# Comando para ligar a API travada em 1 worker (segurança para os 2 núcleos do KVM2)
CMD ["uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]