import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import logging

# 使用环境变量
MILVUS_PATH = os.getenv('MILVUS_PATH')
MILVUS_LOAD_PATH = os.getenv('MILVUS_LOAD_PATH')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusLoadRetrieval:
    def __init__(self):
        self.milvus_path = MILVUS_PATH
        self.milvus_load_path = MILVUS_LOAD_PATH
        self.vector_db = None

    def load(self) -> None:
        try:
            loader = TextLoader(self.milvus_load_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            self.vector_db = Milvus.from_documents(
                docs,
                embeddings,
                connection_args={"uri": self.milvus_path},
            )
            logger.info("Documents loaded successfully into Milvus.")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def retrieval(self, query: str, k: int) -> str:
        if not self.vector_db:
            raise ValueError("Vector database not initialized. Call load() first.")
        try:
            docs = self.vector_db.similarity_search(query, k)
            return '\n\n'.join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

def main():
    try:
        m = MilvusLoadRetrieval()
        m.load()
        res = m.retrieval("早年生活", 4)
        print(res)
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()