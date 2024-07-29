from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from generate_refine_outline import Subsection
from milvus_load_retrieval import MilvusLoadRetrieval
from config import REFINED_OUTILINE, TOPIC, REFERENCES, MILVUSLOADPTAH
from database import Database
DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'

fast_llm = ChatOpenAI(model="gpt-4o")
long_context_llm = ChatOpenAI(model="gpt-4o")

from url_read_search import JinaAI

class Retriever():

    def __init__(self, references):
        self.references = references

    def create(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        reference_docs = [
            Document(page_content=str(v), metadata={"source": k})
            for k, v in self.references.items()
        ]
        print(len(reference_docs))
        vectorstore = SKLearnVectorStore.from_documents(
            reference_docs,
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever(k=10)
        return retriever



class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class WikiSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
                f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
                + f"\n\n{citations}".strip()
        )


from langchain_core.output_parsers import StrOutputParser
class GenerateSections():
    def __init__(self, refine_outline, section_title, references, topic):
        self.refine_outline = refine_outline
        self.section_title = section_title
        self.references = references['references']
        self.topic = topic
        self.article = str
        self.section = str

    async def generate_section(self):
        section_writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位专业的维基百科作者。请根据以下维基百科大纲重写指定的章节。请保留大纲中其他章节的现有内容不变。\n\n"
                    "维基百科大纲：\n\n{outline}\n\n对于您需要重写的章节，请确保使用以下参考资料来丰富内容：\n\n{docs}\n",
                ),
                ("user", "请为{section}章节编写详细，丰富，完整的维基百科章节内容。"),
            ]
        )


        async def retrieve(inputs: dict):
            print("reference:",self.references)
            retriever = Retriever(self.references).create()
            docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])

            formatted = "\n".join(
                [
                    f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                    for doc in docs
                ]
            )

            sources = [doc.metadata['source'] for doc in docs if 'source' in doc.metadata]
            sourceReader = JinaAI()
            formatted = ""
            for source in sources:
                content = sourceReader.reader(source)
                with open(MILVUSLOADPTAH, 'w', encoding='utf-8') as file:
                    file.write(content)
                m = MilvusLoadRetrieval()
                m.load()
                retrieve_content = m.retrieval(inputs["section"])
                formatted += retrieve_content


            return {"docs": formatted, **inputs}


        section_writer = (
                retrieve
                | section_writer_prompt
                | long_context_llm.with_structured_output(WikiSection)
        )

        # 应该是循环遍历
        section = await section_writer.ainvoke(
            {
                "outline": self.refine_outline.as_str,
                "section": self.section_title,
                "topic": self.topic,
                # "outline": self.refine_outline,
                # "section": "鲁迅作品的社会影响",
                # "topic": "鲁迅的生平与背景",
            }
        )
        self.section = section.as_str
        return section.as_str

class GenerateArticle():
    def __init__(self, topic: str, draft: str):
        self.db = Database(DATABASE_PATH)
        self.id = id
        self.topic = topic
        self.draft = draft

    async def generate_article(self):

        writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    你是一位专业的维基百科作者。使用以下部分草稿，编写关于{topic}的完整维基百科文章：
                    
                    
                    草稿:{draft}\n\n
                    
                    要求：严格遵守维基百科格式指南，根据草稿内容，调整它生成完整的维基百科文章，只能增加描述丰富内容,不可以漏掉草稿的内容
                    """
                ),
                (
                    "user",
                    "使用markdown格式编写完整的维基百科文章。使用脚注格式组织引用，例如'[1]'，"
                    "避免在页脚中重复。在页脚中包含URL。",
                ),
            ]
        )
        print("---- 草稿：", self.draft)
        writer = writer_prompt | long_context_llm | StrOutputParser()
        article = await writer.ainvoke({"topic": self.topic, "draft": self.draft})
        print("--- 生成的文章：", article)
        self.db.update('dynamic', 'refine_content_md = ?', 'id = ?', (article, self.id))
        return article


if __name__ == "__main__":
    refine_outline = REFINED_OUTILINE
    references = REFERENCES
    topic = TOPIC

    # 定义异步函数
    async def main():
        gen_article = GenerateArticle(refine_outline, references, topic)
        await gen_article.generate_section()
        article = await gen_article.generate_article()
        # 这里可以对生成的article进行进一步处理

    # 运行异步函数
    import asyncio
    asyncio.run(main())
