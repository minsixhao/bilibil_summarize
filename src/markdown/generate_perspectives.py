from typing import Dict, Any, List, Optional
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompts import MessagesPlaceholder

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import Annotated, TypedDict
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchResults
from config import OLD_OUTLINE
import re

class Editor(BaseModel):
    affiliation: str = Field(
        description="编辑的主要隶属机构。",
    )
    name: str = Field(
        description="编辑的名字。英文命名", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )
    role: str = Field(
        description="编辑在该主题中的角色。",
    )
    description: str = Field(
        description="编辑的关注点、关切和动机的描述。",
    )

    @property
    def persona(self) -> str:
        return f"名字: {self.name}\n角色: {self.role}\n隶属机构: {self.affiliation}\n描述: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="包含编辑及其角色和隶属机构的综合列表。",
    )


class PerspectiveGenerator:
    def __init__(self):
        self.gen_perspectives_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你需要选择一组多样化（且不同）的维基百科编辑，他们将共同努力创建一个关于该主题的全面文章。每个人代表与该主题相关的不同观点、角色或隶属机构。\
                你可以使用其他相关主题的维基百科页面作为灵感。对于每个编辑，添加他们将关注的内容的描述。
                相关主题的维基页面大纲供参考：
                {examples}""",
                ),
                ("user", "感兴趣的主题: {topic}"),
            ]
        )
        self.gen_perspectives_chain = (self.gen_perspectives_prompt | ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(Perspectives))
    async def generate_perspectives(self, topic: str):
        formatted = OLD_OUTLINE
        return await self.gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})



def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right

def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references

def update_editor(editor, new_editor):
    if not editor:
        return new_editor
    return editor

class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], 'update_references']
    editor: Annotated[Optional['Editor'], 'update_editor']

class Editor:
    def __init__(self, name: str, persona: str):
        self.name = name
        self.persona = persona

class Queries(BaseModel):
    queries: List[str] = Field(default=[], description="为用户问题提供详细的搜索引擎查询列表。")


class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="全面回答用户的问题与引用。",
    )
    cited_urls: List[str] = Field(default=[], description="答案中引用的url列表。")

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )


class InterviewSystem:
    def __init__(self):
        self.fast_llm = ChatOpenAI(model="gpt-4o")
        self.max_num_turns = 5
        self.wrapper = DuckDuckGoSearchAPIWrapper(region="cn-zh", max_results=10)
        self.search = DuckDuckGoSearchResults(api_wrapper=self.wrapper, source="news")

        self.gen_qn_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位经验丰富的维基百科作者，想要编辑一个特定的页面。除了你作为维基百科作者的身份外，你在研究这个主题时还有一个特定的关注点。现在，你正在与一位专家交谈以获取信息。请提出好的问题来获取更多有用的信息。
            当你没有更多问题要问时，说"Thank you so much for your help!"来结束对话。请每次只问一个问题，不要问你之前已经问过的问题。你的问题应该与你想要撰写的主题相关。
            要全面且充满好奇心，尽可能从专家那里获得独特的见解。
            请始终保持你特定的视角：
            {persona}"""),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ])

        self.gen_queries_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位专业的研究助理，请利用搜索引擎回答用户的提问。"),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ])

        self.gen_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位能够有效利用信息的专家。你正在与一位想要撰写你所了解主题的维基百科页面的作者交谈。你已经收集了相关信息，现在将使用这些信息来形成回答。
                请尽可能使你的回答具有信息量，并确保每一句话都有收集到的信息作为支持。
                每个回答都必须有来自可靠来源的引用作为支持，以脚注的形式呈现，并在你的回答后重现URL链接。
            """),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ])

        self.gen_queries_chain = self.gen_queries_prompt | self.fast_llm.with_structured_output(Queries, include_raw=True)
        self.gen_answer_chain = self.gen_answer_prompt | self.fast_llm.with_structured_output(
            AnswerWithCitations, include_raw=True
        ).with_config(run_name="GenerateAnswer")

    @staticmethod
    def tag_with_name(ai_message: AIMessage, name: str):
        ai_message.name = name
        return ai_message

    @staticmethod
    def swap_roles(state: InterviewState, name: str):
        converted = []
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.name != name:
                message = HumanMessage(**message.dict(exclude={"type"}))
            converted.append(message)
        return {"messages": converted}

    async def generate_question(self, state: InterviewState):
        print("=====进入提问generate：state:", state)
        editor = state["editor"]
        gn_chain = (
                RunnableLambda(self.swap_roles).bind(name=editor.name)
                | self.gen_qn_prompt.partial(persona=editor.persona)
                | self.fast_llm
                | RunnableLambda(self.tag_with_name).bind(name=editor.name)
        )
        result = await gn_chain.ainvoke(state)
        return {"messages": [result]}

    @tool
    async def search_engine(self, query: str):
        """在互联网搜索引擎搜索"""
        results = self.search_engine._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in results]



    async def gen_answer(self, state: InterviewState, config: Optional[RunnableConfig] = None, name: str = "Subject_Matter_Expert", max_str_len: int = 15000):
        print("=====进入answer：state:", state)

        swapped_state = self.swap_roles(state, name)
        # print("====格式化message：swapped_state:", swapped_state)
        queries = await self.gen_queries_chain.ainvoke(swapped_state)
        # print("AI怎么去分步骤搜索queries:", queries)
        # print(queries["parsed"].queries)
        query_results = await self.search.abatch(queries["parsed"].queries, config, return_exceptions=True)
        # print("批处理搜索的结果query_results:", query_results)
        successful_results = [res for res in query_results if not isinstance(res, Exception)]
        # print("successful_results:", successful_results)
        # all_query_results = {res['link']: res['snippet'] for results in successful_results for res in results}
        # print("all_query_results:", all_query_results)
        # all_query_results = []
        # for item in successful_results:
        #     # 提取link和snippet部分
        #     print(item)
        #     print(type(item))
        #     print(item.length)
        #     print(item.link == '')
        #     # if item.link is None:
        #     if "link:" in item:
        #         parts = item.split(", link: ")
        #         snippet_part = parts[0].replace('[snippet: ', '')
        #         link_part = parts[1].replace(']', '')
        #
        #         # 创建字典条目
        #         all_query_results[link_part] = snippet_part
        # 使用正则表达式提取 snippet 和 link
        pattern = re.compile(r'\[snippet: (.*?), link: (.*?)\]')

        # 提取并转换为字典
        all_query_results = {match.group(2): match.group(1) for item in successful_results if (match := pattern.search(item))}
        # print("所有请求的返回all_query_results:",all_query_results)
        dumped = json.dumps(all_query_results, ensure_ascii=True)
        # print("格式化JSON dumped:", dumped)
        ai_message: AIMessage = queries["raw"]
        tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
        tool_id = tool_call["id"]
        tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
        # print("工具 Message tool_message:", tool_message)
        swapped_state["messages"].extend([ai_message, tool_message])
        # print("状态 swapped_state:", swapped_state)
        generated = await self.gen_answer_chain.ainvoke(swapped_state)
        # print("生成的回答 generated:", generated)
        cited_urls = set(generated["parsed"].cited_urls)
        cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
        formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
        return {"messages": [formatted_message], "references": cited_references, "mess": swapped_state["messages"]}

    def route_messages(self, state: InterviewState, name: str = "Subject_Matter_Expert"):
        messages = state["messages"]
        print("=====进入route_messages：state:", state)

        # print("路由消息 messages:", messages)
        num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
        # print("路由消息 num_responses:", num_responses)
        if num_responses >= self.max_num_turns:
            return END

        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"

    def build_graph(self):
        builder = StateGraph(InterviewState)
        builder.add_node("ask_question", self.generate_question)
        builder.add_node("answer_question", self.gen_answer)
        builder.add_conditional_edges("answer_question", self.route_messages)
        builder.add_edge("ask_question", "answer_question")
        builder.set_entry_point("ask_question")
        return builder.compile().with_config(run_name="Conduct Interviews")


async def main():
    interview_system = InterviewSystem()
    interview_graph = interview_system.build_graph()
    perspective_generator = PerspectiveGenerator()

    # Step 1: 生成专家
    topic = "鲁迅的作品及思想"
    perspectives = await perspective_generator.generate_perspectives(topic)
    # print("Generated Perspectives:", perspectives.editors[0])

    # Step 2: Initialize the interview state with one of the generated editors
    # initial_state = InterviewState(
    #     messages=[],
    #     references={},
    #     editor=perspectives.editors[0]
    # )

    # Step 3: Generate the first question
    # question_result = await interview_system.generate_question(initial_state)
    # print("First Question:", question_result)
    #
    # # Simulate an answer
    # initial_state["messages"].append(HumanMessage(content="专家您好，我正在撰写关于鲁迅的维基百科页面，特别关注他的文学作品和思想。首先，我想请教一下，您认为鲁迅的哪一部作品最能代表他的文学思想？为什么？"))
    # answer_result = await interview_system.gen_answer(initial_state)
    # print("First Answer:", answer_result)

    # Continue the interview process by updating the state iteratively
    # current_state = initial_state
    # for _ in range(interview_system.max_num_turns):
    #     # Generate next question
    #     question_result = await interview_system.generate_question(current_state)
    #     current_state["messages"] += question_result["messages"]
    #
    #     # Generate answer
    #     answer_result = await interview_system.gen_answer(current_state)
    #     current_state["messages"] += answer_result["messages"]
    #
    #     # Print each question/answer pair
    #     print("Question:", question_result["messages"][-1].content)
    #     print("============================")
    #     print("Answer:", answer_result["messages"][-1].content)

    final_step = None

    initial_state = {
        "editor": perspectives.editors[0],
        "messages": [
            AIMessage(
                content=f"听说你在写一篇关于 {topic} 的文章？",
                name="Subject_Matter_Expert",
            )
        ],
    }

    # async for step in interview_graph.astream(initial_state):
    #     # print("Step:", step)
    #     name = next(iter(step))
    #     print(name)
    #     # ask_question 就是历史记录+提问
    #     # answer_question 就是回答的问题 AIMessage
    #     print("-- ", str(step[name]["messages"])[:300])
    #     if END in step:
    #         print("limian Final MESSAGE:", initial_state)
    #         final_step = step
    #
    # print("Final State:", final_step)
    # print("Final MESSAGE:", initial_state)
    # final_state = next(iter(final_step.values()))
    # print("==final_state:", final_state)

    res = await interview_graph.ainvoke(initial_state)
    print("结束：", res)


# Run the main function
import asyncio
asyncio.run(main())