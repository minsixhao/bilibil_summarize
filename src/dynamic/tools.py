import os
import sqlite3
import tempfile
import json
import requests
from fake_useragent import UserAgent
from openai import OpenAI
import moviepy.editor as mp
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from database import Database
from bilibili_api import video, Credential, HEADERS
import httpx
import os
import json
import time
from pydub import AudioSegment

DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
BASE_URL = '/Users/mins/Desktop/github/bilibili_summarize/static'
COOKIE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/cookie/cookie.json'

# class DyanmicTools:
#     def video_2_audio(self, id: str):
#         try:
#             videos_path = os.path.join(BASE_URL, 'video', f"{id}.mp4")
#             audio_path = os.path.join(BASE_URL, 'audio', f"{id}.mp3")
#
#             conn = sqlite3.connect(DATABASE_PATH)
#             cursor = conn.cursor()
#
#             cursor.execute("SELECT COUNT(*) FROM dynamic WHERE id = ?", (id,))
#             exists = cursor.fetchone()[0]
#
#             my_clip = mp.VideoFileClip(videos_path)
#             my_clip.audio.write_audiofile(audio_path)
#
#             with open(audio_path, 'rb') as audio_file:
#                 audio_bytes = audio_file.read()
#
#             if exists:
#                 cursor.execute("UPDATE dynamic SET audio = ? WHERE id = ?", (audio_bytes, id))
#             else:
#                 cursor.execute("INSERT INTO dynamic (id, audio) VALUES (?, ?)", (id, audio_bytes))
#
#             conn.commit()
#             conn.close()
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"文件未找到: {e}")
#         except Exception as e:
#             raise Exception(f"发生错误: {e}")
#
#     def audio_2_content(self, id: str):
#         if not id:
#             raise ValueError("id 未提供")
#
#         client = OpenAI()
#
#         conn = sqlite3.connect(DATABASE_PATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT audio FROM dynamic WHERE id = ?", (id,))
#         audio_data = cursor.fetchone()
#
#         if audio_data is None:
#             raise ValueError("未找到音频数据")
#
#         temp_audio_file_path = None
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
#                 temp_audio_file.write(audio_data[0])
#                 temp_audio_file_path = temp_audio_file.name
#
#             with open(temp_audio_file_path, "rb") as audio_file:
#                 transcript = client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file,
#                     response_format="json"
#                 )
#             print(transcript)
#             content = transcript.text
#
#             cursor.execute("UPDATE dynamic SET content = ? WHERE id = ?", (content, id))
#             conn.commit()
#         except Exception as e:
#             raise Exception(f"OpenAI API 错误: {e}")
#         finally:
#             if temp_audio_file_path:
#                 os.remove(temp_audio_file_path)
#             conn.close()
#
#     def content_2_summary(self, id: str):
#         if not id:
#             raise ValueError("id 未提供")
#
#         try:
#             prompt_template = """
#             Write a concise summary of the following:
#             "{text}"
#             CONCISE SUMMARY，请用简体中文回答:
#             """
#             prompt = PromptTemplate.from_template(prompt_template)
#
#             llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
#             llm_chain = LLMChain(llm=llm, prompt=prompt)
#             stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
#
#             conn = sqlite3.connect(DATABASE_PATH)
#             cursor = conn.cursor()
#             cursor.execute("SELECT content FROM dynamic WHERE id = ?", (id,))
#             text = cursor.fetchone()
#
#             if text is None:
#                 raise ValueError("未找到文本数据")
#
#             documents = [Document(page_content=text[0])]
#             summary = stuff_chain.invoke(documents)["output_text"]
#             print(summary)
#             cursor.execute("UPDATE dynamic SET summary = ? WHERE id = ?", (summary, id))
#             conn.commit()
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"File not found error: {e}")
#         except IOError as e:
#             raise IOError(f"IO error: {e}")
#         except Exception as e:
#             raise Exception(f"发生错误: {e}")
#         finally:
#             conn.close()
#
#     def sent_bilibili_dynamic(self, id: str):
#         if not id:
#             raise ValueError("id 未提供")
#
#         try:
#             with open(COOKIE_PATH, 'r') as file:
#                 cookie = dict(json.load(file))
#         except FileNotFoundError:
#             raise FileNotFoundError("未查询到用户文件，请确认资源完整")
#         except json.JSONDecodeError:
#             raise ValueError("用户文件格式错误")
#
#         ua = UserAgent()
#         url = "https://api.vc.bilibili.com/dynamic_svr/v1/dynamic_svr/create"
#         headers = {
#             "Content-Type": "application/x-www-form-urlencoded",
#             "User-Agent": ua.random,
#             "Cookie": "; ".join([f"{key}={value}" for key, value in cookie.items()])
#         }
#
#         conn = sqlite3.connect(DATABASE_PATH)
#         cursor = conn.cursor()
#         cursor.execute("SELECT summary FROM dynamic WHERE id = ?", (id,))
#         summary = cursor.fetchone()[0]
#         print(summary)
#         content = f"https://www.bilibili.com/video/{id}\n{summary}"
#         data = {
#             "type": 4,
#             "rid": 0,
#             "content": content
#         }
#
#         try:
#             response = requests.post(url, headers=headers, data=data)
#             response.raise_for_status()
#             cursor.execute("UPDATE dynamic SET is_sent = ? WHERE id = ?", (1, id))
#             conn.commit()
#         except requests.RequestException as e:
#             raise Exception(f"动态发布失败，错误信息: {str(e)}")
#         except Exception as e:
#             raise Exception(f"发生错误: {e}")
#         finally:
#             conn.close()

class DyanmicTools:
    def __init__(self):
        self.db = Database(DATABASE_PATH)

    def video_2_audio(self, id: str):
        try:
            videos_path = os.path.join(BASE_URL, 'video', f"{id}.mp4")
            audio_path = os.path.join(BASE_URL, 'audio', f"{id}.mp3")

            exists = self.db.query("dynamic", "COUNT(*)", "id = ?", (id,))[0][0]

            my_clip = mp.VideoFileClip(videos_path)
            my_clip.audio.write_audiofile(audio_path)

            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            if exists:
                self.db.update("dynamic", "audio = ?", "id = ?", (audio_bytes, id))
            else:
                self.db.insert("dynamic", "id, audio", (id, audio_bytes))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件未找到: {e}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")

    def audio_2_content(self, id: str):
        if not id:
            raise ValueError("id 未提供")

        client = OpenAI()
        audio_data = self.db.query("dynamic", "audio", "id = ?", (id,))

        if not audio_data:
            raise ValueError("未找到音频数据")

        temp_audio_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(audio_data[0][0])
                temp_audio_file_path = temp_audio_file.name

            audio_file = AudioSegment.from_file(temp_audio_file_path)
            chunk_size = 100 * 1000  # 100秒
            chunks = [audio_file[i:i+chunk_size] for i in range(0, len(audio_file), chunk_size)]

            transcript = ""
            for chunk in chunks:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_chunk_file:
                    chunk.export(temp_chunk_file.name, format="wav")
                    with open(temp_chunk_file.name, "rb") as f:
                        prompt='以下是普通话的句子'
                        result = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            response_format="json",
                            language="zh",
                            prompt=prompt
                        )
                        print(result)
                    transcript += result.text

            print(transcript)

            self.db.update("dynamic", "content = ?", "id = ?", (transcript, id))

        except Exception as e:
            raise Exception(f"OpenAI API 错误: {e}")
        finally:
            if temp_audio_file_path:
                os.remove(temp_audio_file_path)

    def content_2_summary(self, id: str):
        if not id:
            raise ValueError("id 未提供")

        try:
            prompt_template = """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY，请用简体中文回答:
            """
            prompt = PromptTemplate.from_template(prompt_template)

            llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

            text = self.db.query("dynamic", "content", "id = ?", (id,))

            if not text:
                raise ValueError("未找到文本数据")

            documents = [Document(page_content=text[0][0])]
            summary = stuff_chain.invoke(documents)["output_text"]
            print(summary)
            self.db.update("dynamic", "summary = ?", "id = ?", (summary, id))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found error: {e}")
        except IOError as e:
            raise IOError(f"IO error: {e}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")

    def sent_bilibili_dynamic(self, id: str):
        if not id:
            raise ValueError("id 未提供")

        try:
            with open(COOKIE_PATH, 'r') as file:
                cookie = dict(json.load(file))
        except FileNotFoundError:
            raise FileNotFoundError("未查询到用户文件，请确认资源完整")
        except json.JSONDecodeError:
            raise ValueError("用户文件格式错误")

        ua = UserAgent()
        url = "https://api.vc.bilibili.com/dynamic_svr/v1/dynamic_svr/create"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": ua.random,
            "Cookie": "; ".join([f"{key}={value}" for key, value in cookie.items()])
        }

        summary = self.db.query("dynamic", "summary", "id = ?", (id,))[0][0]
        print("summary:",)
        content = f"https://www.bilibili.com/video/{id}\n{summary}"
        data = {
            "type": 4,
            "rid": 0,
            "content": content
        }

        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            self.db.update("dynamic", "is_sent = ?", "id = ?", (1, id))
        except requests.RequestException as e:
            raise Exception(f"动态发布失败，错误信息: {str(e)}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")


class BilibiliDownloader:
    def __init__(self):
        with open(COOKIE_PATH, 'r') as file:
            cookies = json.load(file)
        self.SESSDATA = cookies["SESSDATA"]
        self.BILI_JCT = cookies["bili_jct"]
        self.BUVID3 = cookies["sid"]
        self.FFMPEG_PATH = "ffmpeg"
        self.PATH = "/Users/mins/Desktop/github/bilibili_summarize/static/video"

    async def download_url(self, url: str, out: str, info: str):
        async with httpx.AsyncClient(headers=HEADERS) as sess:
            resp = await sess.get(url)
            length = resp.headers.get('content-length')
            with open(out, 'wb') as f:
                process = 0
                for chunk in resp.iter_bytes(1024):
                    if not chunk:
                        break
                    process += len(chunk)
                    print(f'下载 {info} {process} / {length}')
                    f.write(chunk)

    async def download_video(self, id: str):
        credential = Credential(sessdata=self.SESSDATA, bili_jct=self.BILI_JCT, buvid3=self.BUVID3)
        v = video.Video(bvid=id, credential=credential)
        info = await v.get_info()
        download_url_data = await v.get_download_url(0)
        detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
        streams = detecter.detect_best_streams()
        await self.download_url(streams[0].url, os.path.join(self.PATH, "video_temp.m4s"), "视频流")
        await self.download_url(streams[1].url, os.path.join(self.PATH, "audio_temp.m4s"), "音频流")
        output_path = os.path.join(self.PATH, f"{id}.mp4")
        os.system(f'{self.FFMPEG_PATH} -i {os.path.join(self.PATH, "video_temp.m4s")} -i {os.path.join(self.PATH, "audio_temp.m4s")} -vcodec copy -acodec copy {output_path}')
        os.remove(os.path.join(self.PATH, "video_temp.m4s"))
        os.remove(os.path.join(self.PATH, "audio_temp.m4s"))
        print(f'已下载为：{output_path}')
        return output_path

