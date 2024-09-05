import requests
from langchain_community.document_loaders import SeleniumURLLoader
from typing import Optional

class JinaAI:
    def __init__(self):
        self.api_key = 'jina_71f4a7a68d654f91a57de239d447d003abAJOru2J46Wc4wrVGOign-9BSbP'
        self.reader_base_url = 'https://r.jina.ai/'
        self.search_base_url = 'https://s.jina.ai/'

    def _make_request(self, url: str, headers: dict) -> Optional[str]:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None

    def reader(self, url: str) -> Optional[str]:
        reader_url = self.reader_base_url + url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Return-Format": "text"
        }
        return self._make_request(reader_url, headers)

    def search(self, url: str) -> Optional[str]:
        search_url = self.search_base_url + url
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return self._make_request(search_url, headers)
    
    def url_summary(self, url: str) -> Optional[str]:
        loader = SeleniumURLLoader(urls=[url])
        try:
            data = loader.load()
            return data[0] if data else None
        except Exception:
            return None

if __name__ == "__main__":
    from config import MILVUSLOADPTAH
    from milvus_load_retrieval import MilvusLoadRetrieval

    jina_ai = JinaAI()
    sources = ['https://zh.wikipedia.org/zh-cn/%E6%9E%97%E5%88%99%E5%BE%90']

    for source in sources:
        content = jina_ai.reader(source)
        if content:
            with open(MILVUSLOADPTAH, 'w', encoding='utf-8') as file:
                file.write(content)
            
            m = MilvusLoadRetrieval()
            m.load()
            retrieve_content = m.retrieval('早年生活', 4)
            print('--')
            print("retrieve_content:", retrieve_content)
        else:
            print(f"无法读取源：{source}")