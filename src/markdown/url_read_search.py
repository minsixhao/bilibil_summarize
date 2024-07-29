import requests
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class JinaAI():
    def __init__(self):
        self.apiKey = 'jina_71f4a7a68d654f91a57de239d447d003abAJOru2J46Wc4wrVGOign-9BSbP'
        self.readerBaseUrl = 'https://r.jina.ai/'
        self.searchBaseUrl = 'https://s.jina.ai/'

    def reader(self, url: str):
        readerUrl = self.readerBaseUrl + url
        try:
            response = requests.get(readerUrl, headers={
                "Authorization": f"Bearer {self.apiKey}",
                "X-Return-Format": "text"
            })
            response.raise_for_status()  # Ensure we raise an error for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            return None

    def search(self, url: str):
        searchUrl = self.searchBaseUrl + url
        try:
            response = requests.get(searchUrl, headers={"Authorization": f"Bearer {self.apiKey}"})
            response.raise_for_status()  # Ensure we raise an error for bad status codes
            return response.text
        except requests.exceptions.RequestException as e:
            return None
    
    def url_summary(self, url: str):
        loader = SeleniumURLLoader(urls=[url])
        try:
            data = loader.load()
            return data[0]
        except Exception as e:
            return None
    

if __name__ == "__main__":

    # reader_result = JinaAI().reader('https://m.gushiwen.cn/gushiwen_0fd00ff1aa.aspx')
    # print(reader_result)
    #
    # search_result = JinaAI().search('https://www.sohu.com/a/656087478_121123846')
    # print(search_result)

    # search_result = JinaAI().url_summary('https://www.sohu.com/a/656087478_121123846')
    # print(search_result)

    from config import MILVUSLOADPTAH
    from url_read_search import JinaAI
    from milvus_load_retrieval import MilvusLoadRetrieval

    sourceReader = JinaAI()

    sources = ['https://zh.wikipedia.org/zh-cn/%E6%9E%97%E5%88%99%E5%BE%90']
    for source in sources:
        content = sourceReader.reader(source)
        with open(MILVUSLOADPTAH, 'w', encoding='utf-8') as file:
            file.write(content)
        m = MilvusLoadRetrieval()
        m.load()
        retrieve_content = m.retrieval('早年生活', 4)
        print('--')
        print("retrieve_content:", retrieve_content)