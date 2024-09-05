from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing import List

from database import Database
from config import *

DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'

class BaseMarkdownProcessor:
    def __init__(self):
        self.db = Database(DATABASE_PATH)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    def _invoke_chain(self, prompt_template, input_data, output_model):
        chain = prompt_template | self.llm.with_structured_output(output_model)
        return chain.invoke(input_data)

class GenerateMarkdown(BaseMarkdownProcessor):
    def content_2_markdown(self, content: str, id: str):
        """
        content 文本转换成 markdown 文档
        """

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                    你是一位经验丰富的维基百科编辑专家，熟知如何组织和呈现信息，使其既清晰又吸引人。你的任务是接收一篇长文章，并将其转换成Markdown格式，以便于阅读和编辑。在这一过程中，你不能对文章的内容进行任何增删或修改，只能通过调整格式来提升文章的可读性和结构化程度。
                    
                    步骤：
                    
                    保留所有原始信息，确保没有任何内容被添加、删除或改变。
                    使用Markdown语法创建标题层级，以反映文章的主题结构。
                    将重要段落或关键点转换为列表或引用，以增强视觉效果和重点突出。
                    为每个段落分配适当的Markdown样式，如加粗、斜体等，用于强调关键词或短语。
                    保持原文的连贯性，确保转换后的Markdown文档逻辑清晰，易于理解。
                    文章内容：
                    {content}
                    
                    转换要求：
                    
                    请使用Markdown语法中的#、##、###等符号创建标题层级。
                    利用* 、1. 等列表格式整理信息。
                    对于重要的短语或词语，使用**或*进行加粗或斜体标记。
                    如果文章中有代码或特殊格式，使用代码块（```）包裹起来。
                """
            ),
            (
                "user",
                """
                {content}
                """
            )
        ])

        output_parser = StrOutputParser()
        chain = prompt_template | self.llm | output_parser
        rlt = chain.invoke({"content": content})
        self.db.update('dynamic', 'content_md = ?', 'id = ?', (rlt, id))
        return rlt

class SummaryMarkdown(BaseMarkdownProcessor):
    def summary_markdown(self, markdown: str, id: str):
        """
        摘要精简 markdown 文档
        """

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
        
                你是一位精通文本摘要的专家，专长在于从大量信息中提炼关键内容，同时维持文档的结构性和逻辑性。你被赋予了一篇完整的Markdown格式文档，任务目标是在不改变其原有的标题层级结构（包含一级、二级以及三级标题）的基础上，对各个章节的内容进行精准的摘要处理。摘要需要高度概括各段落的核心要点，剔除不必要的冗余信息，确保读者能够快速获取文章精华，同时对整体内容有清晰的理解。
                
                Markdown 内容：
                {markdown}
        
                在处理摘要时，请遵循以下指导原则：
                
                1. 紧扣主题，只保留与标题直接相关的重点信息。
                2. 使用简洁明了的语言，避免复杂句式和重复表述。
                3. 维持原文档的逻辑流程，确保摘要内容连贯且易于理解。
                4. 尽可能保留引用、数据和事实，因为它们往往是支撑论点的关键。
                5. 对于较长的段落，提取首句和尾句作为摘要的起点和终点，中间部分则选取最具代表性的句子进行概括。
                请依据上述提示词和原则，开始你的摘要工作。
                """
            ),
            (
                "user",
                """
                {markdown}
                """
            )
        ])

        class SummaryMarkdown(BaseModel):
            """摘要markdown文档"""
            summary_markdown: str = Field(
                description="摘要后的 markdown 文档。",
            )

        rlt = self._invoke_chain(prompt_template, {"markdown": markdown}, SummaryMarkdown)
        self.db.update('dynamic', 'summary_md = ?', 'id = ?', (rlt.summary_markdown, id))
        return rlt.summary_markdown

class TopicsMarkdown(BaseMarkdownProcessor):
    def generate_topics(self, markdown: str, id: str):
        """
        对的精简 markdown 文档进行主题提炼
        """

        prompt_template = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
                    你是一位精通Markdown文档分析与提炼的专家，擅长从复杂的文档中识别和提取关键主题。你被赋予了一份Markdown格式的文档，任务是深入阅读并理解文档内容，然后从中提炼出若干个关键主题。这些主题应当全面覆盖文档的主要观点和信息，并且可以直接用于维基百科搜索，以便于你针对每个主题进行深入研究和探索。
                    在提炼主题时，请遵循以下指导原则：
                    识别核心观点：深入理解文档的主旨，识别出文档试图传达的核心观点和信息。
                    保持主题的独立性：确保每个提炼出的主题都是独立且完整的，避免主题之间的重叠。
                    注意主题的深度和广度：提炼的主题不仅要涵盖文档的主要信息，还要有足够的深度引导深入研究，同时具有广度以覆盖不同方面的信息。
                    使用清晰的语言：在提炼主题时，使用清晰、准确、易于理解的语言，避免使用模糊或技术性过强的术语。
                    考虑主题的逻辑关系：在提炼主题时，考虑它们之间的逻辑关系和层次结构，确保主题的组织有助于读者对文档内容的整体理解。
                    Markdown 内容：
                    {markdown}
                    根据上述原则，开始你的分析和提炼工作，确保提炼出的主题能够帮助你或他人对文档的主要内容有一个清晰、全面的认识，并且可以直接用于维基百科搜索。
                    """
                ),
                (
                    "user",
                    """
                    {markdown}
                    """
                )
            ])

        class Topics(BaseModel):
            """
            生成主题
            """
            topics: List[str] = Field(
                description="一个字符串列表，包含提炼出来的主题",
                default_factory=list
            )

        rlt = self._invoke_chain(prompt_template, {"markdown": markdown}, Topics)
        return rlt.topics

class KeyWordMarkdown(BaseMarkdownProcessor):
    def generate_keywords(self, markdown: str, id: str):
        """
        对摘要进行提炼主题
        """

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                你是一位精通Markdown文档分析与提炼的专家，擅长从复杂的文档中识别和提取关键字。你被赋予了一份Markdown格式的文档，任务是深入阅读并理解文档内容，然后从中提炼出若干个关键字。这些关键字应当全面覆盖文档的主要观点和信息，并且可以直接用于维基百科搜索，以便于你针对每个关键字进行深入研究和探索。
                
                在提炼关键字时，请遵循以下指导原则：
                
                1. **识别核心观点**：深入理解文档的主旨，识别出文档试图传达的核心观点和信息。
                2. **保持关键字的独立性**：确保每个提炼出的关键字都是独立且完整的，避免关键字之间的重叠。
                3. **注意关键字的深度和广度**：提炼的关键字不仅要涵盖文档的主要信息，还要有足够的深度引导深入研究，同时具有广度以覆盖不同方面的信息。
                4. **使用清晰的语言**：在提炼关键字时，使用清晰、准确、易于理解的语言，避免使用模糊或技术性过强的术语。
                5. **考虑关键字的逻辑关系**：在提炼关键字时，考虑它们之间的逻辑关系和层次结构，确保关键字的组织有助于读者对文档内容的整体理解。
                6. **适用于维基百科搜索**：确保提炼出的关键字可以直接用于维基百科搜索，涵盖单个词语、短语、人名、地名、事件、日期、作品名称、组织和公司名称、科学术语和技术术语等。
                
                Markdown 内容：
                {markdown}
                
                根据上述原则，开始你的分析和提炼工作，确保提炼出的关键字能够帮助你或他人对文档的主要内容有一个清晰、全面的认识，并且可以直接用于维基百科搜索。
                """
            ),
            (
                "user",
                """
                {markdown}
                """
            )
        ])

        class Keywords(BaseModel):
            """
            生成主题
            """
            keywords: List[str] = Field(
                description="一个字符串列表，包含提炼出来的关键字",
                default_factory=list
            )

        rlt = self._invoke_chain(prompt_template, {"markdown": markdown}, Keywords)
        return rlt.keywords

class TopicMarkdown(BaseMarkdownProcessor):
    def generate_topic(self, topics_keywords: str, id: str):
        """Generate a topic from the topic_keywords."""
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """
        
                您是一位专业的信息整合专家，擅长将复杂的信息和关键词转化为详细且吸引人的主题描述。您将接收一个包含多个主题和关键词的字符串 topics_keywords。您的任务是深入分析这些信息，并提炼出一个清晰、准确、全面且具有吸引力的主题描述。

                输入：
                topics_keywords 字符串： {topics_keywords}

                任务要求：

                深入分析：对输入字符串中的每个主题和关键词进行深入分析，确保主题描述能够全面反映这些要点。
                详细性：提供足够的细节和背景信息，使主题描述丰富而具体，但同时保持清晰和条理。
                相关性：确保主题描述与输入的关键词紧密相关，能够准确传达信息内容和上下文。
                创造性：在保持相关性的同时，尝试创造性地组合关键词和信息，以产生新颖且有吸引力的主题描述。
                通用性：考虑主题描述在不同上下文中的适用性，确保它在相关领域内具有广泛的认知度。
                易于理解：确保主题描述易于理解，避免使用过于专业或晦涩的术语，除非它们对于描述是必要的。
                记忆点：在主题描述中嵌入易于记忆的元素，如引人入胜的开头、有力的结论或引人注目的统计数据。
                输出：
                提供一个不少于 50 字符合上述要求的主题描述，它应该能够为读者提供对主题的全面理解，并激发他们的兴趣。
                """
            ),
            (
                "user",
                """
                {topics_keywords}
                """
            )
        ])

        class Topic(BaseModel):
            """ 生成主题 """
            topic: str = Field(
                description="整合输入字符串中的主题和关键词，生成一个清晰、准确、全面且具有吸引力的主题描述。",
            )

        rlt = self._invoke_chain(prompt_template, {"topics_keywords": topics_keywords}, Topic)
        self.db.update('dynamic', 'topic = ?', 'id = ?', (rlt.topic, id))
        return rlt.topic

from langchain.globals import set_verbose