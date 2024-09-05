import logging
from typing import List, Optional
import json
from douyin_baike import DouyinBaikeAPI, LoadHtmlFile
from config import TOPICS_KEYWORDS, DATABASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchException(Exception):
    pass

class InitializeResearch:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        self.baike = DouyinBaikeAPI()
        self.load = LoadHtmlFile()

    def get_research_ids(self) -> List[str]:
        research_ids = []
        for keyword in self.keywords:
            logger.info(f"Processing keyword: {keyword}")
            baike_data_id = self.baike.get_document_data(keyword)
            if baike_data_id is None:
                logger.warning(f"{keyword} not found in baike")
                continue
            research_ids.append(baike_data_id)
        return research_ids

    def get_research_data(self, research_ids: List[str]) -> List[Optional[str]]:
        return [self.baike.download_html(research_id) for research_id in research_ids if self.baike.download_html(research_id) is not None]

    def load_html(self, file_ids: List[str]) -> List[Optional[str]]:
        return [self.load.load_html(file_id) for file_id in file_ids if self.load.load_html(file_id) is not None]

    def split_and_vectorize(self, html_data: str):
        """ 向量化，用于后面检索 """
        # 实现向量化逻辑

def save_to_json_file(data, filename: str):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to {filename}")
    except IOError as e:
        logger.error(f"Error saving data to {filename}: {e}")
        raise ResearchException(f"Failed to save data to {filename}")

def main():
    initialize_research = InitializeResearch(TOPICS_KEYWORDS)
    try:
        ids = initialize_research.get_research_ids()
        logger.info(f"Retrieved research IDs: {ids}")

        htmls = initialize_research.get_research_data(ids)
        datas = initialize_research.load_html(ids)
        
        save_to_json_file(datas, 'data.json')
    except ResearchException as e:
        logger.error(f"Research initialization failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()