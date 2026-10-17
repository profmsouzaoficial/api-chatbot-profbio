import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from llama_index.core import (
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Carrega variáveis de ambiente (.env)
load_dotenv()

# Força a rota da OpenAI para a DeepSeek
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
os.environ["OPENAI_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY")

# 1. Configurar o LLM (DeepSeek)
Settings.llm = OpenAILike(
    model="deepseek-chat", 
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com", 
    is_chat_model=True,
    context_window=64000, 
    max_tokens=2048
)

# 2. Configurar o mesmo Modelo Leve usado no gerar_banco.py
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


PASTA_BANCO = "banco_vetores"

# ==========================================
# MAPEAMENTO DE URLS PARA O BUBBLE
# ==========================================
try:
    with open("links_profbio.json", "r", encoding="utf-8") as f:
        urls_brutas = json.load(f)
        URLS_DOCUMENTOS = {chave.lower(): valor for chave, valor in urls_brutas.items()}
    print(f"🔗 Mapeamento blindado carregado: {len(URLS_DOCUMENTOS)} links prontos.")
except FileNotFoundError:
    URLS_DOCUMENTOS = {}
    print("⚠️ Ficheiro links_profbio.json não encontrado. Rode o scraper.py primeiro.")

URL_FALLBACK = "https://sistemaprofbio.icb.ufmg.br/pub/normasforms.php"

# Inicialização Global
motor_de_chat = None

def inicializar_banco():
    if not os.path.exists(PASTA_BANCO):
        raise FileNotFoundError(f"⚠️ A pasta '{PASTA_BANCO}' não existe. Rode o script 'gerar_banco.py' localmente primeiro!")
        
    print("⚡ Banco de dados vetorial carregado a partir do disco!")
    storage_context = StorageContext.from_defaults(persist_dir=PASTA_BANCO)
    return load_index_from_storage(storage_context)

# Carrega a IA para a memória RAM quando o servidor arranca
@asynccontextmanager
async def lifespan(app: FastAPI):
    global motor_de_chat
    index = inicializar_banco()
    
    prompt_sistema = (
        "Você é o assistente virtual acadêmico do PROFBIO. "
        "Sua missão é analisar os trechos de documentos fornecidos e responder às dúvidas com máxima precisão. "
        "REGRA 1: Baseie-se EXCLUSIVAMENTE nos documentos fornecidos. "
        "REGRA 2: Você DEVE citar explicitamente o nome do documento no meio do seu texto explicativo. "
        "REGRA 3: Se a resposta não estiver nos trechos fornecidos, diga claramente: 'Não encontrei essa informação nos regimentos atuais'."
    )
    
    motor_de_chat = index.as_chat_engine(
        chat_mode="context",
        system_prompt=prompt_sistema,
        similarity_top_k=30
    )
    yield

# ==========================================
# DEFINIÇÃO DA API (Instância do FastAPI)
# ==========================================
app = FastAPI(title="API Chatbot PROFBIO", lifespan=lifespan)

# Libera o acesso para o Bubble (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequisicaoChat(BaseModel):
    mensagem: str

# Rota principal - AGORA ASSÍNCRONA E ULTRA LEVE
@app.post("/chat")
async def rota_chat(req: RequisicaoChat):
    # 1. Consulta a IA de forma assíncrona (libera a CPU para outros alunos)
    resposta_ia = await motor_de_chat.achat(req.mensagem)
    
    # 2. Extrai e mapeia as referências
    fontes_encontradas = {}
    for node in resposta_ia.source_nodes:
        nome_original = node.metadata.get('file_name', 'Documento Desconhecido')
        nome_busca = nome_original.lower()
        url_doc = URLS_DOCUMENTOS.get(nome_busca, URL_FALLBACK)
        fontes_encontradas[nome_original] = url_doc
        
    lista_referencias = [{"titulo": k, "link": v} for k, v in fontes_encontradas.items()]
    
    return {
        "texto_resposta": resposta_ia.response,
        "referencias": lista_referencias
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
