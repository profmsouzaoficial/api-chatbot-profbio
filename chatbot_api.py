import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

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

# 2. Configurar o Motor Local e Fatiamento Cirúrgico
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.chunk_size = 512       
Settings.chunk_overlap = 50     

PASTA_DADOS = "base_profbio"
PASTA_BANCO = "banco_vetores"

# ==========================================
# MAPEAMENTO DE URLS PARA O BUBBLE
# ==========================================
try:
    with open("links_profbio.json", "r", encoding="utf-8") as f:
        urls_brutas = json.load(f)
        # Converte as chaves (nomes dos arquivos) para letras minúsculas
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
        print("📚 A criar banco fatiado com alta precisão (PyMuPDF)...")
        leitor_pdf = PyMuPDFReader()
        documentos = SimpleDirectoryReader(
            PASTA_DADOS, 
            file_extractor={".pdf": leitor_pdf}
        ).load_data()
        
        print(f"🚀 A vetorizar {len(documentos)} páginas. Isto pode demorar alguns minutos...")
        index = VectorStoreIndex.from_documents(documentos, show_progress=True)
        
        print("💾 A guardar o banco de dados no disco (PASTA_BANCO)...")
        index.storage_context.persist(persist_dir=PASTA_BANCO)
        return index
    else:
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

# Estrutura do payload recebido do Bubble
class RequisicaoChat(BaseModel):
    mensagem: str

# Rota principal (Endpoint /chat)
@app.post("/chat")
def rota_chat(req: RequisicaoChat):
    # 1. Consulta a IA
    resposta_ia = motor_de_chat.chat(req.mensagem)
    
    # 2. Extrai e mapeia as referências
    fontes_encontradas = {}
    for node in resposta_ia.source_nodes:
        # Extrai o nome do arquivo original
        nome_original = node.metadata.get('file_name', 'Documento Desconhecido')
        # Força para minúsculo para fazer a busca segura no dicionário
        nome_busca = nome_original.lower()
        
        # Faz a associação. Se falhar, usa a página genérica de normas
        url_doc = URLS_DOCUMENTOS.get(nome_busca, URL_FALLBACK)
        
        # Salva o nome original limpo e a URL correspondente
        fontes_encontradas[nome_original] = url_doc
        
    # 3. Formata as referências como uma lista de dicionários para o Bubble ler
    lista_referencias = [{"titulo": k, "link": v} for k, v in fontes_encontradas.items()]
    
    return {
        "texto_resposta": resposta_ia.response,
        "referencias": lista_referencias
    }

if __name__ == "__main__":
    import uvicorn
    # Arranque do servidor local (Pode ser utilizado para testes antes do deploy)
    uvicorn.run(app, host="0.0.0.0", port=8000)
