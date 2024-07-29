from langchain_community.document_loaders import TextLoader
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from config import MILVUSPATH, MILVUSLOADPTAH

from url_read_search import JinaAI

class MilvusLoadRetrieval:
    
    def __init__(self):
        self.milvus_path = MILVUSPATH
        self.milvus_load_path = MILVUSLOADPTAH

    def load(self):
        # 加载向量库
        # milvus = Milvus.load(self.milvus_path)
        # 加载文档
        loader = TextLoader(self.milvus_load_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        self.vector_db = Milvus.from_documents(
            docs,
            embeddings,
            connection_args={"uri": self.milvus_path},
        )

    def retrieval(self, query):
        docs = self.vector_db.similarity_search(query, 2)
        content = docs[0].page_content + '\n\n' + docs[1].page_content
        return content
# sourceReader = JinaAI()
# content = sourceReader.url_summary('https://www.sohu.com/a/656087478_121123846')
# print(content)
# # 将 content 写入文件
# with open(MILVUSPATH, 'w', encoding='utf-8') as file:
#     file.write(content)



# loader = TextLoader(self.milvus_load_path)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
# docs = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()

# URI = "/Users/mins/Desktop/github/bilibili_summarize/db/milvus/milvus.db"

# vector_db = Milvus.from_documents(
#     docs,
#     embeddings,
#     connection_args={"uri": URI},
# )

# query = "早年生活"
# docs = vector_db.similarity_search(query)
# print(docs)
# docs[0].page_content

# m =  MilvusLoadRetrieval()
# m.load()
# res = m.retrieval("早年生活")
# print(res)