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

long_context_llm = ChatOpenAI(model="gpt-4-turbo-preview")

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

    def generate_refine_outline(self):
        refine_outline_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page. \
        You need to make sure that the outline is comprehensive and specific. \
        Topic you are writing about: {topic} 
        
        Old outline:
        
        {old_outline}""",
                ),
                (
                    "user",
                    "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
                ),
            ]
        )

        refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
            Outline
        )

        refined_outline = refine_outline_chain.invoke(
            {
                "topic": self.topic,
                "old_outline": OLD_OUTLINE,
                "conversations": self.conversations,
                # "conversations": "\n\n".join(
                #     f"### {m.name}\n\n{m.content}" for m in self.conversations
                # ),
            }
        )

        self.db.update('dynamic', 'refine_outline_md = ?', 'id = ?', (refined_outline, self.id))

        return refined_outline

if __name__ == "__main__":
    refine_outline = RefineOutline(MESSAGE)
    res = refine_outline.generate_refine_outline()
    print("==:",res)