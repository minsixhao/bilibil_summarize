

from douyin_baike import DouyinBaikeAPI, LoadHtmlFile
from config import TOPICS_KEYWORDS

class InitializeResearch:
    def __init__(self, keywords):
        self.keywords = keywords
        self.baike = DouyinBaikeAPI()
        self.load = LoadHtmlFile()

    def get_research_ids(self):
        research_ids = []
        for keyword in self.keywords:
            baike_data_id = self.baike.get_document_data(keyword)
            if baike_data_id is None:
                print(f"{keyword} not found in baike")
                continue
            research_ids.append(baike_data_id)
        return research_ids


    def get_research_data(self, research_ids):
        research_data = []
        for research_id in research_ids:
            html = self.baike.download_html(research_id)
            if html is None:
                print(f"{research_id} not found in html")
                continue
            research_data.append(html)
        return research_data


    def load_html(self, file_ids):
        html_data = []
        for file_id in file_ids:
            html = self.load.load_html(file_id)
            if html is None:
                print(f"{file_id} not found in html")
                continue
            html_data.append(html)
        return html_data


    def split_and_vectorize(self, html_data: str):
        """ 向量化，用于后面检索 """


import json
def save_to_json_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)




initialize_research = InitializeResearch(TOPICS_KEYWORDS)

ids = initialize_research.get_research_ids()
htmls = initialize_research.get_research_data(ids)
datas = initialize_research.load_html(ids)
print(datas)
save_to_json_file(datas, 'data.json')  # 保存到 data.json 文件中