from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

from fake_useragent import UserAgent
from openai import OpenAI
from config import DATABASE_PATH


import os
import sys

# sys.path.append('/Users/mins/Desktop/github/bilibili_summarize')

from tools import GenerateMarkdown, SummaryMarkdown, TopicMarkdown, KeyWordMarkdown
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
    perspectives: Annotated[str, lambda x, y: f"{x}+{y}"]
    topics: Annotated[str, lambda x, y: f"{x}+{y}"]
    keywords: Annotated[str, lambda x, y: f"{x}+{y}"]
    topics_keywords: List[str]


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
        topics = TopicMarkdown().generate_topics(summary_md, id)
        return {"topics": topics}

    def get_keywords(self, state: State):
        id = state['id']
        summary_md = state['summary_markdown']
        keywords = KeyWordMarkdown().generate_keywords(summary_md, id)
        return {"keywords": keywords}

    def merge_topics_keywords(self, state: State):
        topic_list = state['topics']
        keywords_list = state['keywords']
        return {"topics_keywords": topic_list + keywords_list }

    def create_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("ContentGetter", self.get_content_text)
        workflow.add_node("MarkdownCreator", self.generate_markdown)
        workflow.add_node("ContentSummarizer", self.summary_markdown)
        workflow.add_node("TopicExtractor", self.get_topics)
        workflow.add_node("KeywordExtractor", self.get_keywords)
        # workflow.add_node("BaikeResearcher", self.baike_search)
        workflow.add_node("Merger", self.merge_topics_keywords)
        workflow.set_entry_point("ContentGetter")

        workflow.add_edge("ContentGetter", "MarkdownCreator")
        workflow.add_edge("MarkdownCreator", "ContentSummarizer")
        workflow.add_edge("ContentSummarizer", "TopicExtractor")
        workflow.add_edge("ContentSummarizer", "KeywordExtractor")
        workflow.add_edge(["TopicExtractor", "KeywordExtractor"], "Merger")
        workflow.add_edge("Merger", END)
        # workflow.add_edge("BaikeResearcher", END)

        return workflow

if __name__ == "__main__":
    summarizer = BilibiliSummarizer()
    workflow = summarizer.create_workflow()

    id = "BV1jK421b7Z2"
    # id = "BV1AS421N7Rc"
    input = {
        "id": id,
    }

    graph = workflow.compile()
    events = graph.stream(input)

    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))

    for s in events:
        print(s)
        print("----")