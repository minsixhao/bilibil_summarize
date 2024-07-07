
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OLD_OUTLINE, TOPIC, CONVERSATION
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

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

class WikipediaOutlineRefiner:
    def __init__(self, llm):
        self.llm = llm
        self.refine_outline_prompt = ChatPromptTemplate.from_messages(
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

        self.refine_outline_chain = self.refine_outline_prompt | self.llm.with_structured_output(
           Outline
        )
    
    def refine_outline(self, topic, old_outline, conversations):
        # formatted_conversations = "\n\n".join(
        #     f"### {name}\n\n{content}" for name, content in conversations.items()
        # )
        formatted_conversations = conversations
        print(formatted_conversations)
        result = self.refine_outline_chain.invoke(
            {
                "topic": topic,
                "old_outline": old_outline,
                "conversations": formatted_conversations,
            }
        )

        # for token in result:
        #     print("---")
        #     print(token)
        
        return result


# Example usage
llm = ChatOpenAI(model="gpt-4o")
refiner = WikipediaOutlineRefiner(llm)

print("===")
refined_outline = refiner.refine_outline(TOPIC, OLD_OUTLINE, CONVERSATION)
print("refined_outline:", refined_outline)