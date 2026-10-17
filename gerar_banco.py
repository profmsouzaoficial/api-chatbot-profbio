import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

# 1. Configurar o Modelo Leve para CPU
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
Settings.chunk_size = 512       
Settings.chunk_overlap = 50     

PASTA_DADOS = "base_profbio"
PASTA_BANCO = "banco_vetores"

def construir_banco():
    print("📚 A iniciar leitura dos PDFs (Pode usar bastante CPU local)...")
    leitor_pdf = PyMuPDFReader()
    documentos = SimpleDirectoryReader(
        PASTA_DADOS, 
        file_extractor={".pdf": leitor_pdf}
    ).load_data()
    
    print(f"🚀 A vetorizar {len(documentos)} páginas com o modelo leve. Aguarde...")
    index = VectorStoreIndex.from_documents(documentos, show_progress=True)
    
    print(f"💾 A guardar o banco de vetores na pasta '{PASTA_BANCO}'...")
    index.storage_context.persist(persist_dir=PASTA_BANCO)
    print("✅ Concluído! A pasta banco_vetores foi atualizada e está pronta para o servidor.")

if __name__ == "__main__":
    construir_banco()
