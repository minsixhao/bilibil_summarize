outline = """

# Impact of million-plus token context window language models on RAG

## Introduction

Overview of million-plus token context window language models and RAG (Retrieval-Augmented Generation).

## Million-Plus Token Context Window Language Models

Explanation of million-plus token context window language models, their architecture, training data, and applications.

## RAG (Retrieval-Augmented Generation)

Overview of RAG, its architecture, how it combines retrieval and generation models, and its use in natural language processing tasks.

## Impact on RAG

Discuss the impact of million-plus token context window language models on RAG, including improvements in performance, efficiency, and challenges faced.

"""


editors = """
{'editors': [{'affiliation': 'Academic Research',
   'name': 'Dr. Linguist',
   'role': 'Language Model Expert',
   'description': 'Dr. Linguist will focus on explaining the technical aspects of million-plus token context window language models and their impact on RAG (Retrieval-Augmented Generation) systems.'},
  {'affiliation': 'Industry',
   'name': 'TechTrendz',
   'role': 'AI Solutions Architect',
   'description': 'TechTrendz will provide insights on the practical applications of million-plus token context window language models in RAG systems and discuss their benefits and challenges in real-world scenarios.'},
  {'affiliation': 'Open Source Community',
   'name': 'CodeGenius',
   'role': 'Machine Learning Enthusiast',
   'description': 'CodeGenius will explore the open-source tools and frameworks available for implementing million-plus token context window language models in RAG systems and share their experiences with the community.'},
  {'affiliation': 'Tech Journalism',
   'name': 'DataDive',
   'role': 'AI Technology Journalist',
   'description': 'DataDive will cover the latest developments and advancements in million-plus token context window language models and their implications for RAG systems, focusing on industry trends and use cases.'}]}
"""



from typing import List
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import AIMessage
from generate_perspectives_conversation import Editor, InterviewState, InterviewSystem
from generate_refine_outline import Outline
from generate_article import WikiSection
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START

class ResearchState(BaseModel):
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[InterviewState]
    sections: List[WikiSection] = []
    article: str = ""

import asyncio

interview_system = InterviewSystem()
interview_graph = interview_system.build_graph()

async def initialize_research(state: ResearchState):
    topic = state["topic"]
    # coros = (
    #     chushihuadagang,
    #     survey_subjects.ainvoke(topic),
    # )
    # results = await asyncio.gather(*coros)
    # return {
    #     **state,
    #     "outline": results[0],
    #     "editors": results[1].editors,
    # }
    return {
        **state,
        "outline": outline,
        "editors": editors,
    }


async def conduct_interviews(state: ResearchState):
    topic = state["topic"]
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="Subject_Matter_Expert",
                )
            ],
        }
        for editor in state["editors"]
    ]
    # We call in to the sub-graph here to parallelize the interviews
    interview_results = await interview_graph.abatch(initial_states)

    return {
        **state,
        "interview_results": interview_results,
    }


def format_conversation(interview_state):
    messages = interview_state["messages"]
    convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
    return f'Conversation with {interview_state["editor"].name}\n\n' + convo


async def refine_outline(state: ResearchState):
    convos = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ]
    )

    updated_outline = await refine_outline_chain.ainvoke(
        {
            "topic": state["topic"],
            "old_outline": state["outline"].as_str,
            "conversations": convos,
        }
    )
    return {**state, "outline": updated_outline}


async def index_references(state: ResearchState):
    all_docs = []
    for interview_state in state["interview_results"]:
        reference_docs = [
            Document(page_content=v, metadata={"source": k})
            for k, v in interview_state["references"].items()
        ]
        all_docs.extend(reference_docs)
    await vectorstore.aadd_documents(all_docs)
    return state


async def write_sections(state: ResearchState):
    outline = state.outline
    sections = await section_writer.abatch(
        [
            {
                "outline": outline.as_str,
                "section": section.section_title,
                "topic": state.topic,
            }
            for section in outline.sections
        ]
    )
    state.sections = sections
    return state

async def write_article(state: ResearchState):
    draft = "\n\n".join([section.as_str for section in state.sections])
    state.article = await writer.ainvoke({"topic": state.topic, "draft": draft})
    return state



builder_of_storm = StateGraph(ResearchState)

nodes = [
    ("init_research", initialize_research),
    ("conduct_interviews", conduct_interviews),
    ("refine_outline", refine_outline),
    ("index_references", index_references),
    ("write_sections", write_sections),
    ("write_article", write_article),
]
for i, (name, node) in enumerate(nodes):
    builder_of_storm.add_node(name, node)
    if i > 0:
        builder_of_storm.add_edge(nodes[i - 1][0], name)

builder_of_storm.add_edge(START, nodes[0][0])
builder_of_storm.add_edge(nodes[-1][0], END)
storm = builder_of_storm.compile(checkpointer=MemorySaver())



config = {"configurable": {"thread_id": "my-thread"}}
async for step in storm.astream(
        {
            "topic": "Groq, NVIDIA, Llamma.cpp and the future of LLM Inference",
        },
        config,
):
    name = next(iter(step))
    print(name)
    print("-- ", str(step[name])[:300])


