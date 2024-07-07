
from langgraph.prebuilt import create_react_agent

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


from tools import DyanmicTools
processor = DyanmicTools()
@tool
def video_2_audio_tool(id: str):
    """
    Convert a video to its corresponding audio by its bvid.

    Args:
        id (str): The ID of the video to convert to audio.

    Returns:
        str: The ID of the video that was successfully converted to audio.
    """
    processor.video_2_audio(id)

@tool
def audio_2_content_tool(id: str):
    """
    Convert a audio to its corresponding content by its bvid.

    Args:
        id (str): The ID of the audio to convert to content.

    Returns:
        str: The ID of the audio that was successfully converted to content.
    """
    processor.audio_2_content(id)

@tool
def content_2_summary_tool(id: str):
    """
    Summarize content by its ID.

    Args:
        id (str): The ID of the content to summarize.

    Returns:
        str: The summary of the content that was successfully generated.
    """
    processor.content_2_summary(id)

@tool
def sent_bilibili_dynamic_tool(id: str):
    """
    Summarize content by its ID and post the summary to Bilibili's dynamic.

    Args:
        id (str): The ID of the content to summarize.

    Returns:
        str: A message indicating whether the summary was successfully posted to Bilibili's dynamic.
    """
    processor.send_bilibili_dynamic(id)

tools = [video_2_audio_tool, audio_2_content_tool, content_2_summary_tool, sent_bilibili_dynamic_tool]
model = ChatOpenAI(model="gpt-4o")
graph = create_react_agent(model, tools=tools)

from langchain_core.messages import HumanMessage
input = {"messages": [HumanMessage(content="请将视频（id：BV1AS421N7Rc）下载为音频，音频转为文本，再对文本进行摘要，并发送到bilibili动态")]}


from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))


for s in graph.stream(input, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()