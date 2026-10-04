import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PAGINAS_ALVO = [
    "https://sistemaprofbio.icb.ufmg.br/pub/reg.php",
    "https://sistemaprofbio.icb.ufmg.br/pub/portarias.php",
    "https://sistemaprofbio.icb.ufmg.br/pub/normasforms.php",
    "https://sistemaprofbio.icb.ufmg.br/pub/tutoriais.php"
]

dicionario_links = {}

print("🔍 A iniciar varredura agressiva no site do PROFBIO...")

for pagina in PAGINAS_ALVO:
    print(f"Acessando: {pagina}")
    try:
        resposta = requests.get(pagina, verify=False)
        resposta.raise_for_status()
        sopa = BeautifulSoup(resposta.text, 'html.parser')
        
        # Procura em todos os links da página
        for link in sopa.find_all('a'):
            href = link.get('href')
            if not href:
                continue
                
            href_lower = href.lower()
            # Se a palavra pdf, doc ou docx estiver EM QUALQUER PARTE do link
            if '.pdf' in href_lower or '.doc' in href_lower or '.docx' in href_lower:
                url_completa = urljoin(pagina, href)
                
                # O urlparse limpa o "lixo" do final (ex: tira o ?v=123)
                caminho_limpo = urlparse(url_completa).path
                nome_arquivo = caminho_limpo.split('/')[-1]
                
                # Salva no dicionário apenas se houver um nome válido
                if nome_arquivo:
                    dicionario_links[nome_arquivo] = url_completa
                    
    except Exception as e:
        print(f"⚠ Erro ao processar {pagina}: {e}")

with open("links_profbio.json", "w", encoding="utf-8") as f:
    json.dump(dicionario_links, f, indent=4, ensure_ascii=False)

print(f"\n✅ Scraping concluído! {len(dicionario_links)} ficheiros encontrados.")
