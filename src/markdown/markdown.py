from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain.schema import Document

from fake_useragent import UserAgent
from openai import OpenAI
from config import DATABASE_PATH


import os
import sys

# sys.path.append('/Users/mins/Desktop/github/bilibili_summarize')

from tools import GenerateMarkdown, SummaryMarkdown, TopicsMarkdown, KeyWordMarkdown, TopicMarkdown
from database import Database

class Sentence(TypedDict):
    sentence: str
    refs: List[str]

class ContentSection(TypedDict):
    section_title: str
    section_content: List[Sentence]
    subsections: List['ContentSection']

class References(TypedDict):
    key: str

class SpicyEntities(TypedDict):
    entity: str

class State(TypedDict):
    id: str
    title: str
    url: str
    summary: str
    content_text: str
    markdown: str
    summary_markdown: str
    topic: str
    perspectives: Annotated[str, lambda x, y: f"{x}+{y}"]
    topics: Annotated[str, lambda x, y: f"{x}+{y}"]
    keywords: Annotated[str, lambda x, y: f"{x}+{y}"]
    topics_keywords: str


class LoaderKnowledge:
    def __init__(self):
        self.db = Database(DATABASE_PATH)




class BilibiliSummarizer:
    def __init__(self):
        self.db = Database(DATABASE_PATH)

    def get_content_text(self, state: State):
        content_text = self.db.query('dynamic', 'content', 'id = ?', (state['id'],))[0][0]
        return {"content_text": content_text}

    def generate_markdown(self, state: State):
        id = state['id']
        content_text = state['content_text']
        markdown = GenerateMarkdown().content_2_markdown(content_text, id)
        return {"markdown": markdown}

    def summary_markdown(self, state: State):
        id = state['id']
        markdown = state['markdown']
        summary_md = SummaryMarkdown().summary_markdown(markdown, id)
        return {"summary_markdown": summary_md}

    def get_topics(self, state: State):
        id = state['id']
        summary_md = state['summary_markdown']
        topics = TopicsMarkdown().generate_topics(summary_md, id)
        return {"topics": topics}

    def get_keywords(self, state: State):
        id = state['id']
        summary_md = state['summary_markdown']
        keywords = KeyWordMarkdown().generate_keywords(summary_md, id)
        return {"keywords": keywords}

    def merge_topics_keywords(self, state: State):
        topic_list = state['topics']
        keywords_list = state['keywords']
        meger_list = topic_list + keywords_list
        return {"topics_keywords": meger_list}
    
    def generate_topic(self, state: State):
        id = state['id']
        topics_keywords = state['topics_keywords']
        topic = TopicMarkdown().generate_topic(topics_keywords, id)
        return {"topic": topic}

    def create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("ContentGetter", self.get_content_text)
        workflow.add_node("MarkdownCreator", self.generate_markdown)
        workflow.add_node("ContentSummarizer", self.summary_markdown)
        workflow.add_node("TopicsExtractor", self.get_topics)
        workflow.add_node("KeywordExtractor", self.get_keywords)
        # workflow.add_node("BaikeResearcher", self.baike_search)
        workflow.add_node("Merger", self.merge_topics_keywords)
        workflow.add_node("TopicGenerator", self.generate_topic)
        workflow.set_entry_point("ContentGetter")

        workflow.add_edge("ContentGetter", "MarkdownCreator")
        workflow.add_edge("MarkdownCreator", "ContentSummarizer")
        workflow.add_edge("ContentSummarizer", "TopicsExtractor")
        workflow.add_edge("ContentSummarizer", "KeywordExtractor")
        workflow.add_edge(["TopicsExtractor", "KeywordExtractor"], "Merger")
        workflow.add_edge("Merger", "TopicGenerator")
        workflow.add_edge("TopicGenerator", END)
        # workflow.add_edge("BaikeResearcher", END)

        return workflow


from generate_perspectives_conversation import PerspectiveGenerator, Editor, InterviewSystem
from generate_refine_outline import RefineOutline
from generate_article import GenerateArticle, GenerateSections
from collections import defaultdict
import asyncio

class MarkdownState(TypedDict):
    id: str
    keywords: str
    topic: str
    perspectives: List[Editor]
    conversations: List[Any]
    references: List[Any]
    refine_outline: Any
    article: Any

class MarkdownGenerator:
    def __init__(self):
        self.db = Database(DATABASE_PATH)

    def get_topic(self, state: State):
        topic = self.db.query('dynamic', 'topic', 'id = ?', (state['id'],))[0][0]
        return {"topic": topic}

    # def search_keywords(self, state: MarkdownState):
    #     keywords = state['keywords']

    async def generate_perspectives(self, state: MarkdownState):
        id = state['id']
        topic = state['topic']
        perspectives = await PerspectiveGenerator().generate_perspectives(topic, id)
        return {"perspectives": perspectives}

    async def generate_conversation(self, state: MarkdownState):
        perspectives = state['perspectives']
        topic = state['topic']
        interview_system = InterviewSystem()
        interview_graph = interview_system.build_graph()
        conversations = []
        merged_references = defaultdict(str)
        for editor in perspectives.editors:
            initial_state = {
                "editor": editor,
                "messages": [
                    AIMessage(
                        content=f"听说你在写一篇关于 {topic} 的文章？",
                        name="Subject_Matter_Expert",
                    )
                ],
            }
            ask_answer = await interview_graph.ainvoke(initial_state)

            # 聚合对话消息
            # conversations 向量检索
            conversations += ask_answer["messages"]
            for key, value in ask_answer.get("references", {}).items():
                merged_references[key] = value
        references = {"references": dict(merged_references)}
        return {"conversations": conversations, "references": references}

    def generate_refine_outline(self, state: MarkdownState):
        id = state['id']
        conversations = state['conversations']
        topic = state['topic']
        refine_outline = RefineOutline(conversations, id, topic).generate_refine_outline()
        return {"refine_outline": refine_outline}

    async def generate_article(self, state: MarkdownState):
        refine_outline = state['refine_outline']
        references = state['references']
        topic = state['topic']
        id = state['id']

        print("refine_outline:", refine_outline)
        print("refine_outline.sections:", refine_outline.sections)
        sections = ""
        for section in refine_outline.sections:
            section_title = section.section_title
            print('-- generate_article:', section_title, references, topic)
            gen_section = GenerateSections(refine_outline, section_title, references, topic)
            section = await gen_section.generate_section()
            sections += '\n\n' + section
            print('-- generate_article sections:', sections)

        gen_article = GenerateArticle(id, topic, sections, references)
        article = await gen_article.generate_article()
        return {"article": article}

    def generate_markdown(self):
        workflow = StateGraph(MarkdownState)
        workflow.add_node('TopicGetter', self.get_topic)
        workflow.add_node('PerspectivesGenerator', self.generate_perspectives)
        workflow.add_node('ConversationGenerator', self.generate_conversation)
        workflow.add_node('RefineOutlineGenerator', self.generate_refine_outline)
        workflow.add_node('ArticleGenerator', self.generate_article)
        workflow.set_entry_point("TopicGetter")

        workflow.add_edge("TopicGetter", "PerspectivesGenerator")
        workflow.add_edge("PerspectivesGenerator", "ConversationGenerator")
        workflow.add_edge("ConversationGenerator", "RefineOutlineGenerator")
        workflow.add_edge("RefineOutlineGenerator", "ArticleGenerator")
        workflow.add_edge("ArticleGenerator", END)
        return workflow



# 运行下面代码，对 id 对应的视频进行总结，并生成 markdown 文件是，摘要 md 大纲， 提炼主题和关键字
# if __name__ == "__main__":
#     summarizer = BilibiliSummarizer()
#     workflow = summarizer.create_workflow()

#     id = "BV1tz421i7PT"
#     input = {
#         "id": id,
#     }

#     graph = workflow.compile()
#     events = graph.stream(input)

#     from IPython.display import Image, display
#     display(Image(graph.get_graph().draw_mermaid_png()))

#     for s in events:
#         print(s)
#         print("----")


# 运行下面代码，对 id 对应的视频进行深入研究，生成一篇 markdown 文档
from config import TOPIC
if __name__ == "__main__":
    markdown = MarkdownGenerator()
    workflow = markdown.generate_markdown()

    id = "BV1tz421i7PT"
    input = {
        "id": id,
    }

    graph = workflow.compile()

    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))

    async def process_events(events):
        async for s in events:
            print(s)
            print("----")

    # 新建事件循环并运行 process_events 函数
    import asyncio
    asyncio.run(process_events(graph.astream(input)))