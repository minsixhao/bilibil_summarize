
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

from fake_useragent import UserAgent
from openai import OpenAI

from tools import DyanmicTools

import moviepy.editor as mp
import os
import json
import requests
import sqlite3
import tempfile

DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
BASE_URL = '/Users/mins/Desktop/github/bilibili_summarize/static'
COOKIE_PATH = '/bilibili_summarize/cookie/cookie.json'

class MyState(TypedDict):
    id: str
    video: str
    audio: str
    content: str
    summary: str
    dynamic: bool

dyanmicTools = DyanmicTools()
def download_video_node(state: MyState):
    # 下载视频
    return {"video": "已完成"}

def video_2_audio_node(state: MyState):
    # 视频转音频
    id = state["id"]
    try:
        dyanmicTools.video_2_audio(id)
        return {"audio": "已完成"}
    except Exception as e:
        return {"audio": e}

def audio_2_content_node(state: dict):
    # 音频转换为文本
    id = state["id"]
    try:
        dyanmicTools.audio_2_content(id)
        return {"content": "已完成"}
    except Exception as e:
        return {"content": e}

def content_2_summary_node(state: MyState):
    # 文本摘要
    id = state["id"]
    try:
        dyanmicTools.content_2_summary(id)
        return {"summary": "已完成"}
    except Exception as e:
        return {"summary": e}

def sent_bilibili_dynamic_node(state: MyState):
    # 发布 bilibili 动态
    id = state["id"]
    try:
        dyanmicTools.sent_bilibili_dynamic(id)
        return {"dynamic": "已完成"}
    except Exception as e:
        return {"dynamic": e}


workflow = StateGraph(MyState)
workflow.add_node("VideoDownloader", download_video_node)
workflow.add_node("AudioExtractor", video_2_audio_node)
workflow.add_node("Transcriber", audio_2_content_node)
workflow.add_node("Summarizer", content_2_summary_node)
workflow.add_node("BilibiliDynamicPoster", sent_bilibili_dynamic_node)

workflow.set_entry_point("VideoDownloader")
workflow.add_edge("VideoDownloader", "AudioExtractor")
workflow.add_edge("AudioExtractor", "Transcriber")
workflow.add_edge("Transcriber", "Summarizer")
workflow.add_edge("Summarizer", "BilibiliDynamicPoster")
workflow.add_edge("BilibiliDynamicPoster", END)

# 编译工作流图
graph = workflow.compile()

from IPython.display import Image, display
# 画出工作流图
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

input = {
    "id":"BV1AS421N7Rc",
    "video": "未完成",
    "audio": "未完成",
    "content": "未完成",
    "summary": "未完成",
    "dynamic": False,
}
# 执行工作流图，流式输出
events = graph.stream(input)
for s in events:
    print(s)
    print("----")