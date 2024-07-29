import asyncio
import os
import json
import httpx
from uuid import uuid4
from typing import TypedDict
from langgraph.graph import END, StateGraph
from tools import DyanmicTools, BilibiliDownloader
unique_id = uuid4().hex[0:8]

DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
BASE_URL = '/Users/mins/Desktop/github/bilibili_summarize/static'
COOKIE_PATH = '/bilibili_summarize/cookie/cookie.json'
os.environ["OPENAI_API_KEY"] = "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"


class MyState(TypedDict):
    id: str
    video: str
    audio: str
    content: str
    summary: str
    dynamic: bool


class DynamicToolsWorkflow:
    def __init__(self):
        self.workflow = StateGraph(MyState)
        self.setup_workflow()
        self.dyanmicTools = DyanmicTools()

    def setup_workflow(self):

        def video_2_audio_node(state: MyState):
            id = state["id"]
            try:
                self.dyanmicTools.video_2_audio(id)
                return {"audio": "已完成"}
            except Exception as e:
                return {"audio": str(e)}

        def audio_2_content_node(state: MyState):
            id = state["id"]
            try:
                self.dyanmicTools.audio_2_content(id)
                return {"content": "已完成"}
            except Exception as e:
                return {"content": str(e)}

        def content_2_summary_node(state: MyState):
            id = state["id"]
            try:
                self.dyanmicTools.content_2_summary(id)
                return {"summary": "已完成"}
            except Exception as e:
                return {"summary": str(e)}

        def sent_bilibili_dynamic_node(state: MyState):
            id = state["id"]
            try:
                self.dyanmicTools.sent_bilibili_dynamic(id)
                return {"dynamic": "已完成"}
            except Exception as e:
                return {"dynamic": str(e)}

        # self.workflow.add_node("VideoDownloader", download_video_node)
        self.workflow.add_node("AudioExtractor", video_2_audio_node)
        self.workflow.add_node("Transcriber", audio_2_content_node)
        self.workflow.add_node("Summarizer", content_2_summary_node)
        self.workflow.add_node("BilibiliDynamicPoster", sent_bilibili_dynamic_node)
        self.workflow.set_entry_point("AudioExtractor")

        # self.workflow.add_edge("VideoDownloader", "AudioExtractor")
        self.workflow.add_edge("AudioExtractor", "Transcriber")
        self.workflow.add_edge("Transcriber", "Summarizer")
        self.workflow.add_edge("Summarizer", "BilibiliDynamicPoster")
        self.workflow.add_edge("BilibiliDynamicPoster", END)

    def compile_workflow(self):
        self.graph = self.workflow.compile()
        from IPython.display import Image, display
        display(Image(self.graph.get_graph(xray=1).draw_mermaid_png()))

    def run_workflow(self, input_data):
        events = self.graph.stream(input_data)
        for s in events:
            print(s)
            print("----")

# 使用示例
if __name__ == "__main__":
    id = "BV1tz421i7PT"
    bilibiliDownloader = BilibiliDownloader()
    asyncio.get_event_loop().run_until_complete(bilibiliDownloader.download_video(id))

    # loop = asyncio.get_event_loop()
    # task = loop.create_task(bilibiliDownloader.download_video(id))
    # loop.run_until_complete(task)

    input_data = {
        "id": id, # 梁启超
        "video": "未完成",
        "audio": "未完成",
        "content": "未完成",
        "summary": "未完成",
        "dynamic": False,
    }

    workflow_manager = DynamicToolsWorkflow()
    workflow_manager.compile_workflow()
    workflow_manager.run_workflow(input_data)
