import os
from dotenv import load_dotenv
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

load_dotenv()

# Forçamos o sistema a usar a URL exata da API
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
os.environ["OPENAI_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY")

# 1. Configurar o LLM (DeepSeek)
Settings.llm = OpenAILike(
    model="deepseek-chat", 
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com", 
    is_chat_model=True,
    context_window=64000, # Aumentamos a janela de memória para o DeepSeek ler dezenas de páginas
    max_tokens=2048
)

# 2. Configurar o Motor Local
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

# 3. A MÁGICA DA PRECISÃO (Novas Configurações)
Settings.chunk_size = 512       # Fatias menores e precisas (tiro de sniper na busca)
Settings.chunk_overlap = 50     # Sobreposição para não cortar regras ao meio

PASTA_DADOS = "base_profbio"
PASTA_BANCO = "banco_vetores"

def inicializar_banco():
    if not os.path.exists(PASTA_BANCO):
        print("📚 A criar banco fatiado com alta precisão (PyMuPDF)...")
        
        # Forçamos o LlamaIndex a usar o motor PyMuPDF para extrair o texto humano limpo
        leitor_pdf = PyMuPDFReader()
        documentos = SimpleDirectoryReader(
            PASTA_DADOS, 
            file_extractor={".pdf": leitor_pdf}
        ).load_data()
        
        print(f"🚀 A vetorizar {len(documentos)} páginas usando o hardware local...")
        index = VectorStoreIndex.from_documents(documentos, show_progress=True)
        
        print("💾 A guardar o banco de dados no disco (PASTA_BANCO)...")
        index.storage_context.persist(persist_dir=PASTA_BANCO)
        return index
    else:
        print("⚡ Banco de dados carregado!")
        storage_context = StorageContext.from_defaults(persist_dir=PASTA_BANCO)
        index = load_index_from_storage(storage_context)
        return index
      
def iniciar_assistente():
    index = inicializar_banco()
    
    prompt_sistema = (
        "Você é o assistente virtual acadêmico do PROFBIO. "
        "Sua missão é analisar os trechos de documentos fornecidos e responder às dúvidas com máxima precisão. "
        "REGRA 1: Baseie-se EXCLUSIVAMENTE nos documentos fornecidos. "
        "REGRA 2: Você DEVE citar explicitamente o nome do documento (ex: Regimento Geral, Portaria 01, etc) no meio do seu texto. "
        "REGRA 3: Se a resposta realmente não estiver nos trechos fornecidos, diga claramente: 'Não encontrei essa informação nos regimentos atuais'."
    )
    
    print("✅ Base pronta! Digite 'sair' para terminar.\n")
    
    # 4. AUMENTAMOS O RADAR DE BUSCA
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=prompt_sistema,
        similarity_top_k=50
    )

    while True:
        pergunta = input("👤 Você: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        
        resposta = chat_engine.chat(pergunta)
        print(f"\n🤖 Assistente PROFBIO: {resposta}\n")
        
        print("📑 REFERÊNCIAS CONSULTADAS:")
        fontes_citadas = set() 
        for node in resposta.source_nodes:
            arquivo = node.metadata.get('file_name', 'Documento desconhecido')
            fontes_citadas.add(f"- {arquivo}")
            
        for fonte in fontes_citadas:
            print(fonte)
            
        print("\n" + "-" * 50)

if __name__ == "__main__":
    iniciar_assistente()
