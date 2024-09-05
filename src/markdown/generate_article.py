from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from milvus_load_retrieval import MilvusLoadRetrieval
from config import REFINED_OUTILINE, TOPIC, REFERENCES, MILVUSLOADPTAH
from database import Database

# 常量定义
DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
FAST_LLM = ChatOpenAI(model="gpt-4o")
LONG_CONTEXT_LLM = ChatOpenAI(model="gpt-4o")

class Retriever:
    def __init__(self, references):
        self.references = references

    def create(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        reference_docs = [Document(page_content=str(v), metadata={"source": k}) for k, v in self.references.items()]
        vectorstore = SKLearnVectorStore.from_documents(reference_docs, embedding=embeddings)
        return vectorstore.as_retriever(k=10)

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

class GenerateSections:
    def __init__(self, refine_outline, section_title, references, topic):
        self.refine_outline = refine_outline
        self.section_title = section_title
        self.references = references['references']
        self.topic = topic

    async def generate_section(self):
        section_writer_prompt = ChatPromptTemplate.from_messages([
            ("system", "您是一位专业的维基百科编辑。请根据如下维基百科大纲重写指定的章节。在修改过程中，请务必遵守大纲的结构和内容，不删除大纲中的其他章节，同时允许您对现有内容进行适当的调整和补充，以提高其准确性和丰富度。\n\n"
                       "维基百科大纲：\n\n{outline}\n\n为了使您撰写的章节更加详细和权威，请使用以下参考资料丰富内容：\n\n{docs}\n"),
            ("user", "请为章节 {section} 编写一个详细且完备的维基百科内容。"),
        ])

        async def retrieve(inputs: dict):
            retriever = Retriever(self.references).create()
            docs = await retriever.ainvoke(inputs["section"])
            formatted = "\n".join([f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>' for doc in docs])

            m = MilvusLoadRetrieval()
            m.load()
            for doc in docs:
                retrieve_content = m.retrieval(inputs["section"] + doc.page_content, 3)
                formatted += retrieve_content

            return {"docs": formatted, **inputs}

        section_writer = retrieve | section_writer_prompt | LONG_CONTEXT_LLM.with_structured_output(WikiSection)
        section = await section_writer.ainvoke({
            "outline": self.refine_outline.as_str,
            "section": self.section_title,
            "topic": self.topic,
        })
        return section.as_str

class GenerateArticle:
    def __init__(self, id: str, topic: str, draft: str, references):
        self.db = Database(DATABASE_PATH)
        self.id = id
        self.topic = topic
        self.draft = draft
        self.references = references

    async def generate_article(self):
        writer_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位专业的维基百科作者。使用以下部分草稿，编写关于{topic}的完整维基百科文章：\n\n"
                       "草稿:{draft}\n\n"
                       "维基百科的引用链接:{references}\n\n"
                       "要求：严格遵守维基百科格式指南，根据草稿内容，调整它生成完整的维基百科文章，只能增加描述丰富内容,不可以漏掉草稿的内容"),
            ("user", "使用markdown格式编写完整的维基百科文章。使用脚注格式组织引用，例如'[1]'，"
                     "避免在页脚中重复。在页脚中包含URL。"),
        ])

        writer = writer_prompt | LONG_CONTEXT_LLM | StrOutputParser()
        article = await writer.ainvoke({"topic": self.topic, "draft": self.draft, "references": self.references})
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
