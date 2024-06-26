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

DATABASE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/db/sqlite/bilibili.db'
BASE_URL = '/Users/mins/Desktop/github/bilibili_summarize/static'
COOKIE_PATH = '/Users/mins/Desktop/github/bilibili_summarize/cookie/cookie.json'

class DyanmicTools:
    def video_2_audio(self, id: str):
        try:
            videos_path = os.path.join(BASE_URL, 'video', f"{id}.mp4")
            audio_path = os.path.join(BASE_URL, 'audio', f"{id}.mp3")

            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM dynamic WHERE id = ?", (id,))
            exists = cursor.fetchone()[0]

            my_clip = mp.VideoFileClip(videos_path)
            my_clip.audio.write_audiofile(audio_path)

            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            if exists:
                cursor.execute("UPDATE dynamic SET audio = ? WHERE id = ?", (audio_bytes, id))
            else:
                cursor.execute("INSERT INTO dynamic (id, audio) VALUES (?, ?)", (id, audio_bytes))

            conn.commit()
            conn.close()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件未找到: {e}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")

    def audio_2_content(self, id: str):
        if not id:
            raise ValueError("id 未提供")

        client = OpenAI()

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT audio FROM dynamic WHERE id = ?", (id,))
        audio_data = cursor.fetchone()

        if audio_data is None:
            raise ValueError("未找到音频数据")

        temp_audio_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(audio_data[0])
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="json"
                )
            print(transcript)
            content = transcript.text

            cursor.execute("UPDATE dynamic SET content = ? WHERE id = ?", (content, id))
            conn.commit()
        except Exception as e:
            raise Exception(f"OpenAI API 错误: {e}")
        finally:
            if temp_audio_file_path:
                os.remove(temp_audio_file_path)
            conn.close()

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

            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM dynamic WHERE id = ?", (id,))
            text = cursor.fetchone()

            if text is None:
                raise ValueError("未找到文本数据")

            documents = [Document(page_content=text[0])]
            summary = stuff_chain.invoke(documents)["output_text"]
            print(summary)
            cursor.execute("UPDATE dynamic SET summary = ? WHERE id = ?", (summary, id))
            conn.commit()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found error: {e}")
        except IOError as e:
            raise IOError(f"IO error: {e}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")
        finally:
            conn.close()

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

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT summary FROM dynamic WHERE id = ?", (id,))
        summary = cursor.fetchone()[0]
        print(summary)
        content = f"https://www.bilibili.com/video/{id}\n{summary}"
        data = {
            "type": 4,
            "rid": 0,
            "content": content
        }

        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            cursor.execute("UPDATE dynamic SET is_sent = ? WHERE id = ?", (1, id))
            conn.commit()
        except requests.RequestException as e:
            raise Exception(f"动态发布失败，错误信息: {str(e)}")
        except Exception as e:
            raise Exception(f"发生错误: {e}")
        finally:
            conn.close()