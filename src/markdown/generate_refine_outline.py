import os
from uuid import uuid4
unique_id = uuid4().hex[0:8]

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from config import MESSAGE, OLD_OUTLINE
from database import Database
DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'

long_context_llm = ChatOpenAI(model="gpt-4o")

class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()



class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


class RefineOutline():
    def __init__(self, conversations, id: str, topic: str):
        self.db = Database(DATABASE_PATH)
        self.id = id
        self.topic = topic
        self.conversations = conversations
        self.old_outline = self.db.query("dynamic", "summary_md", "id = ?", (id,))

    def generate_refine_outline(self):
        refine_outline_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """作为维基百科的编辑，你负责撰写和完善特定主题的页面。你已经从领域专家和搜索引擎中搜集了相关资料。现在，你需要根据这些资料和专家的见解，进一步细化和完善维基百科页面的大纲。 \
                    主题： {topic} 

                    请遵循以下步骤来优化大纲：
                    1. 确保大纲覆盖了主题的所有关键方面。
                    2. 使用清晰的标题和子标题来组织内容。
                    3. 包含必要的引用和参考资料链接。
                    4. 保持语言的客观性和准确性。
                    5. 内容详尽无遗，深入挖掘主题每个层面，确保每一部分经过精心构思和充分扩展。
                            
                    旧的大纲:
        
                    {old_outline}
                    """,
                ),
                (
                    "user",
                    """
                    根据与领域专家的深入对话和进一步的研究，以下是对维基百科页面大纲的优化建议:


                    对话摘要:
                    
                    {conversations}
                    
                    
                    你优化后的维基百科页面大纲：
                    """,
                ),
            ]
        )

        refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
            Outline
        )

        refined_outline = refine_outline_chain.invoke(
            {
                "topic": self.topic,
                "old_outline": self.old_outline,
                "conversations": self.conversations,
            }
        )

        self.db.update('dynamic', 'refine_outline_md = ?', 'id = ?', (refined_outline, self.id))

        return refined_outline

if __name__ == "__main__":
    refine_outline = RefineOutline(MESSAGE)
    res = refine_outline.generate_refine_outline()
    print("==:",res)