import os


# from langchain.globals import set_debug
# set_debug(True)

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from utils import get_record_manager, clear_vectorstore, bilibli_index
from splitter import Splitter
from config import config
from load import load_markdown
from pprint import pprint

import get_bilibili_vid
import download_video_tool
import asyncio

from langchain.agents import AgentExecutor, XMLAgent, tool, Tool, initialize_agent, AgentType
import moviepy.editor as mp
from openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain_core.runnables import chain
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain.prompts import PromptTemplate

import textwrap
import time
import json
import requests


def init_vectorstore() -> None:
    # text_splitter = Splitter.from_tiktoken_encoder(
    #     chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    # )
    # loader = UnstructuredMarkdownLoader('data/aQ.md')
    # docs = loader.load()
    # splits = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory='bilibili')

    markdown_path = "data/aQ.md"

    loader = UnstructuredMarkdownLoader(markdown_path)
    docs = loader.load()
    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())


    print(docs)
    print(splits)
    print(vectorstore)

    info = bilibli_index(docs)
    print(info)
    pprint(info)


@tool
def mp4_2_mp3(bvids):
    """将MP4文件转换为MP3文件。"""
    videos_file_path = os.path.join(r'C:\Users\Administrator\Desktop\Bilibili2\static\video', f"{bvids}.mp4")
    my_clip = mp.VideoFileClip(videos_file_path)
    my_clip.audio.write_audiofile(os.path.join(r'C:\Users\Administrator\Desktop\Bilibili2\static\speech', f'{os.path.splitext(bvids)[0]}.mp3'))
    return bvids

@tool
def mp3_2_text(bvids):
    """将MP3文件转换为txt文本。"""
    client = OpenAI(
        api_key="sk-8DfsD4B1Kfs096y01066435cE6B0419cB5965cB8679c2d59",
        base_url="https://oneapi.xty.app/v1"
    )

    path = r'C:\Users\Administrator\Desktop\Bilibili2\static\speech'
    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\text'
    speech_path = os.path.join(path, f"{bvids}.mp3")
    print(speech_path)
    retry_limit = 3
    retry_count = 0

    while retry_count < retry_limit:
        try:
            audio_file = open(speech_path, "rb")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json"
            )
            print(transcript.text)

            output_file_path = os.path.join(output_path, f"{bvids}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(transcript.text)

            print(f'Transcript saved to: {output_file_path}')
            return bvids
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to transcribe {bvids} after {retry_limit} attempts')
    return bvids

@tool
def summarize_text(bvids):
    """将文本发送给模型进行总结。"""
    llm = ChatOpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    base_dir = r'C:\Users\Administrator\Desktop\Bilibili2\static\text'
    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\summarize'
    file_name = f"{bvids}.txt"
    file_path = os.path.join(base_dir, file_name)
    documents = []

    if os.path.exists(file_path) and file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    else:
        print(f"File {file_name} not found or is not a .txt file.")

    retry_limit = 3
    retry_count = 0

    while retry_count < retry_limit:
        try:
            output_summary = chain.run(documents)
            output_file_path = os.path.join(output_path, f"{bvids}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(output_summary)

            wrapped_text = textwrap.fill(output_summary, width=100)
            return bvids
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to summarize {bvids} after {retry_limit} attempts')
    return None

@tool
def create_bilibili_dynamic(bvids):
    """创建Bilibili动态"""
    try:
        file = open(f'cookie/cookie.json', 'r')
        cookie = dict(json.load(file))
    except FileNotFoundError:
        msg = '未查询到用户文件，请确认资源完整'
        cookie = 'null'
        print(msg)

    url = "https://api.vc.bilibili.com/dynamic_svr/v1/dynamic_svr/create"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": "; ".join([f"{key}={value}" for key, value in cookie.items()])
    }

    path = r'C:\Users\Administrator\Desktop\Bilibili2\static\summarize'
    text_path = os.path.join(path, f"{bvids}.txt")
    with open(text_path, 'r', encoding='utf-8') as file:
        prompt_content = file.read()

    template_prompt = f"https://www.bilibili.com/video/{bvids}" + "\n" + prompt_content
    data = {
        "type": 4,
        "rid": 0,
        "content": template_prompt
    }
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        print("动态发布成功！")
    else:
        print(f"动态发布失败，状态码: {response.status_code}, 响应内容: {response.text}")

    return bvids


@tool
def get_cid_by_bvid(bvid):
    """根据BV号获取CID"""
    url = f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}&jsonp=jsonp"
    response = requests.get(url)
    data = json.loads(response.text)
    if data['code'] != 0:
        raise Exception(f"Bilibili API returned error: {data}")
    cid = data['data'][0]['cid']
    return cid

def agent_generate_bvid_resources(bvid):
    tool_list = [mp4_2_mp3, mp3_2_text, summarize_text, create_bilibili_dynamic]

    model = ChatOpenAI()
    prompt_template = PromptTemplate(
        template="将视频文件{input},转换为音频，音频再转为文字,对文字进行总结摘要,将摘要发到b站。输出格式化为:{input}",
        input_variables=["input"],
    )

    agent = initialize_agent(tool_list, model, verbose=True, handle_parsing_errors=True,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    agent.invoke(prompt_template.format(input=bvid)).get("output")

@tool
def get_sources_text(bvid):
    """获取本地资源文件中的bvid对应的text内容"""
    base_dir = r'C:\Users\Administrator\Desktop\Bilibili2\static\text'
    file_name = f"{bvid}.txt"
    file_path = os.path.join(base_dir, file_name)

    if os.path.exists(file_path) and file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    else:
        print(f"File {file_name} not found or is not a .txt file.")

@chain
def get_sources_summarize(bvid):
    # """获取本地资源文件中的bvid对应txt文档的内容"""
    base_dir = r'C:\Users\Administrator\Desktop\Bilibili2\static\summarize'
    bvid = bvid['bvid']
    print(bvid)

    file_name = f"{bvid}.txt"
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path) and file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    else:
        print(f"File {file_name} not found or is not a .txt file.")


@chain
def agent_get_iframe_by_bvid(bvid):
    # 定义工具列表
    tool_list = [get_cid_by_bvid]
    model = ChatOpenAI()
    prompt = XMLAgent.get_default_prompt()

    def convert_intermediate_steps(intermediate_steps):
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        return log

    def convert_tools(tools):
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    agent = (
            {
                "question": lambda x: x["question"],
                "intermediate_steps": lambda x: convert_intermediate_steps(
                    x["intermediate_steps"]
                ),
            }
            | prompt.partial(tools=convert_tools(tool_list))
            | model.bind(stop=["</tool_input>", "</final_answer>"])
            | XMLAgent.get_default_output_parser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
    question_template = PromptTemplate(
        template="""
            获取视频{bvid}对应的cid,然后将获取到的cid和bvid替换到下面的iframe标签中,输出对应的iframe标签
            输出：<iframe src="//player.bilibili.com/player.html?aid=623717179&bvid={bvid}&cid=cid&page=1&high_quality=1"  width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
        """,
        input_variables=["bvid"],
    )

    iframe = agent_executor.invoke({"question":question_template.format(bvid=bvid)}).get("output")
    return iframe
@chain
def agent_get_sources_summarize(bvid):
    tool_list = [get_sources_summarize]
    model = ChatOpenAI()
    prompt = XMLAgent.get_default_prompt()

    def convert_intermediate_steps(intermediate_steps):
        log = ""
        for action, observation in intermediate_steps:
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{observation}</observation>"
            )
        return log
    def convert_tools(tools):
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    agent = (
            {
                "question": lambda x: x["question"],
                "intermediate_steps": lambda x: convert_intermediate_steps(
                    x["intermediate_steps"]
                ),
            }
            | prompt.partial(tools=convert_tools(tool_list))
            | model.bind(stop=["</tool_input>", "</final_answer>"])
            | XMLAgent.get_default_output_parser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
    question_template = PromptTemplate(
        template="""
            获取{bvid}文件的内容，按获取到的内容原样输出，不要做任何修改
        """,
        input_variables=["bvid"],
    )

    summarize = agent_executor.invoke({"question":question_template.format(bvid=bvid)}).get("output")
    print(summarize)
    return summarize

@chain
def agent_get_sources_text(bvid):
    tool_list = [get_sources_text]
    model = ChatOpenAI()
    prompt_template = PromptTemplate(
        template="""
            通过{bvid}获取对应的text内容信息。
            如果不存在，则返回空字符串。
        """,
        input_variables=["bvid"],
    )

    agent = initialize_agent(tool_list, model, verbose=True, handle_parsing_errors=True,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    text = agent.invoke(prompt_template.format(bvid=bvid)).get("output")
    return text


@chain
def get_analyze(input):
    bvid = input['bvid']
    text = get_sources_text(bvid)
    model = ChatOpenAI()

    prompt_analyze = PromptTemplate(
        template="""
        根据提供的文章，我需要进行内容归纳，这些文章可能涉及各种主题，包括历史、文学、科技和旅行等。我会将文章分阶段归纳，确保归纳信息准确无误且不丢失重要细节。归纳内容将完全基于原文，关键词将被加粗，这些关键词主要指代名词，如人名、地名、物品或事件等。
        文章内容如下：{text}

        输出为markdown格式：
        
            ### 标题
            - 
            
            ### 标题
            -
        """,
        input_variables=["text"]
    )

    chain_analyze_tetx_to_markdown = {"text": RunnablePassthrough()} | prompt_analyze | model | StrOutputParser()

    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\analyze'

    retry_limit = 3
    retry_count = 0

    while retry_count < retry_limit:
        try:
            output_analyze = chain_analyze_tetx_to_markdown.invoke(text)
            output_file_path = os.path.join(output_path, f"{bvid}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(output_analyze)

            wrapped_text = textwrap.fill(output_analyze, width=100)
            return {"bvid":bvid,"text":output_analyze}
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to analyze {bvid} after {retry_limit} attempts')
    return None


## 从分析文本中提取关键字并加粗
@chain
def extract_analyze_and_bold_keywords(input):
    text =  input['text']
    bvid = input['bvid']
    model = ChatOpenAI()

    prompt_extract= PromptTemplate(
        template="""
        根据提供的文章，你根据文章内容提炼出关键词。
        文章内容如下：{text}

        输出为原来的 markdown 格式，其中关键词用在原位置加粗的方式标记出来。
        ```
        {text}
        ```
                """,
        input_variables=["text"]
    )

    chain_extract = {"text": RunnablePassthrough()} | prompt_extract | model | StrOutputParser()


    base_dir = r'C:\Users\Administrator\Desktop\Bilibili2\static\analyze'
    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\analyze'
    file_name = f"{bvid}.txt"
    file_path = os.path.join(base_dir, file_name)
    documents = []

    if os.path.exists(file_path) and file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    else:
        print(f"File {file_name} not found or is not a .txt file.")

    retry_limit = 3
    retry_count = 0

    while retry_count < retry_limit:
        try:
            output_analyze = chain_extract.invoke(text)
            output_file_path = os.path.join(output_path, f"{bvid}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(output_analyze)

            wrapped_text = textwrap.fill(output_analyze, width=100)
            return {"bvid":bvid,"text":output_analyze}
            # return output_analyze
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to summarize {bvid} after {retry_limit} attempts')
    return None



@chain
def prompt_keywords_to_link(input):
    text =  input['text']
    bvid = input['bvid']
    model = ChatOpenAI()
    prompt_keywords_to_link= PromptTemplate(
        template=f"""
        根据提供的markdown文档：{text}，给里面加粗的关键字进行url编码，再加上维基百科的超链接。
        例如：**[潮汕](https://zh.wikipedia.org/wiki/%E6%BD%AE%E6%B1%95)** 并非一个城市
                """,
        input_variables=["text"]
    )
    chain_prompt_keywords_to_link ={"text": RunnablePassthrough() } | prompt_keywords_to_link | model | StrOutputParser()
    base_dir = r'C:\Users\Administrator\Desktop\Bilibili2\static\analyze'
    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\analyze'
    file_name = f"{bvid}.txt"
    file_path = os.path.join(base_dir, file_name)
    documents = []

    if os.path.exists(file_path) and file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    else:
        print(f"File {file_name} not found or is not a .txt file.")

    retry_limit = 3
    retry_count = 0

    while retry_count < retry_limit:
        try:
            output_analyze = chain_prompt_keywords_to_link.invoke(text)
            output_file_path = os.path.join(output_path, f"{bvid}.txt")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(output_analyze)

            wrapped_text = textwrap.fill(output_analyze, width=100)
            # return {"bvid":bvid,"text":output_analyze}
            return output_analyze
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to summarize {bvid} after {retry_limit} attempts')
    return None

def generate_markdown(bvid):

    model = ChatOpenAI()

    prompt_generate_markdown = PromptTemplate(
        template="""
            将你收到的三部分markdown内容做整合输出完整的markdown格式。
            输入分别对应视频:{iframe}，摘要：{summarize}，详细分析：{analyze}
            
            下面是输出的markdown格式:
            # 标题
            
            ## 视频：
            
            {iframe}
            
            ---
            
            ## 摘要：
            
            {summarize}
            
            ---
            
            ## 详细分析:
            
            {analyze}
            
            ---
        """,
        input_variables=["iframe", "summarize", "analyze"]
    )

    # 怎么将传入的字典类型{}中某一个键具体的值传递给{a:"",b:""}其中一个
    chain_generate_markdown = (
        {
            "iframe": agent_get_iframe_by_bvid,
            "summarize": get_sources_summarize,
            "analyze": get_analyze | extract_analyze_and_bold_keywords | prompt_keywords_to_link
        }
        | prompt_generate_markdown
        | model
        | StrOutputParser()
    )

    output_path = r'C:\Users\Administrator\Desktop\Bilibili2\static\post'
    retry_limit = 3
    retry_count = 0
    while retry_count < retry_limit:
        try:
            markdown = chain_generate_markdown.invoke({"bvid": bvid})
            output_file_path = os.path.join(output_path, f"{bvid}.md")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(markdown)

            wrapped_text = textwrap.fill(markdown, width=100)
            return markdown
        except Exception as e:
            print(f'Error: {e}')
            retry_count += 1
            print(f'Retry attempt {retry_count}/{retry_limit}')
            time.sleep(2)

    print(f'Failed to summarize {bvid} after {retry_limit} attempts')
    return None


if __name__ == '__main__':
    # init_vectorstore()

    # 打开文件并逐行读取内容
    with open('cache/cache_bvids.txt', 'r') as file:
        # 读取文件内容，将每行作为列表的一个元素
        cache_bvids = [line.strip() for line in file]

    # 新更视频
    new_bvids = get_bilibili_vid.get_bvids()
    increment_bvids = list(set(new_bvids) - set(cache_bvids))

    increment_bvids = ['BV1AS421N7Rc']
    for bvid in increment_bvids:
        asyncio.get_event_loop().run_until_complete(download_video_tool.download_video(bvid))

    for bvid in increment_bvids:
        # 3. 将下载的视频转换为mp3格式
        # 4. 将mp3传给大模型转文字
        # 5. 大模型摘要文本
        # 6. 将摘要内容发在B站
        agent_generate_bvid_resources(bvid)
        generate_markdown(bvid)
