import requests
import json
import os

class DouyinBaikeAPI:
    def __init__(self):
        self.base_url = 'https://www.baike.com/api/v2/search/getDocData'
        self.headers = {
            "Accept": "application/json,/;q=0.8",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "Cookie": "refresh=true; ttwid=1|-trngjRzjW2k6vi06aa8EFwe5gXCRl7nIjt6MwZ0TNM|1713013578|f0f92007333d9841bf7e75aae9e86ec04829ae82de54d35e95b5746b9e02874e; ttwid_ss=1|-trngjRzjW2k6vi06aa8EFwe5gXCRl7nIjt6MwZ0TNM|1713013578|f0f92007333d9841bf7e75aae9e86ec04829ae82de54d35e95b5746b9e02874e; COOKIE_IS_LOGIN_FLAG=0; COOKIE_IS_LOGIN_FLAG=0; first_screen_node_count=5; view_id=202407241514285FA48E17AED666980568; timestamp=1721805268824",
            "Origin": "https://www.baike.com",
            "Referer": "https://www.baike.com/search?keyword=%E9%98%BF%E6%96%AF%E9%A1%BF%E5%8F%91%E9%80%81%E5%88%B0%E5%8F%91&activeTab=DOC_TAB",
            "Sec-Ch-Ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"macOS\"",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "X-Build-Version": "1.0.1.8724",
        }

    def get_document_data(self, query, start=0, count=2):
        payload = {
            "args": [query, start, count]
        }
        response = requests.post(self.base_url, data=json.dumps(payload), headers=self.headers)
        print(response.text)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 0 and data['message'] == 'success':
                doc_list = data['data']['DocList']
                if len(doc_list) > 0:
                    return doc_list[0]['ID']
                else:
                    return None
            else:
                raise Exception(f"Request failed with code: {data['code']}, message: {data['message']}")
        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

    def download_html(self, file_id: str):
        try:
            directory = '/Users/mins/Desktop/github/bilibili_summarize/static/html'
            url = f"https://www.baike.com/wikiid/{file_id}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                file_path = os.path.join(directory, f"{file_id}.html")

                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)

                return response.text
            else:
                raise Exception(f"请求失败，状态码: {response.status_code}")
        except Exception as e:
            raise e

from langchain_community.document_loaders import BSHTMLLoader
class LoadHtmlFile():
    def __init__(self):
        self.directory = '/Users/mins/Desktop/github/bilibili_summarize/static/html'

    def load_html(self, file_id: str):
        file_path = os.path.join(self.directory, f"{file_id}.html")
        loader = BSHTMLLoader(file_path)
        data = loader.load()
        return data

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
class Html2Text:
    def __init__(self):
        self.urls = ["https://www.baike.com/wikiid/7176566420128727096", 
                     "https://www.baike.com/wikiid/7153244797457072159"]

    def docs_transformed(self):
        loader = AsyncHtmlLoader(self.urls)
        docs = loader.load()
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed

if __name__ == "__main__":
    api = DouyinBaikeAPI()
    load = LoadHtmlFile()
    try:
        id = '7198178132255785017'
        api.download_html(id)
        data = load.load_html(id)
        print(data)

    except Exception as e:
        print(e)

    # API = Html2Text()
    # res = API.docs_transformed()
    # print(res)