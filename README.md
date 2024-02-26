# 目标
已关注的up主更新了视频之后，对视频进行摘要总结。一方面将摘要的内容在 B 站转发，另一方面对视频内容进行更详细的摘要分析，发表到 blog：https://minsixhao.github.io/docs/bilibili

# 实现方案
![alt text](结构图.png)
借助了 Langchain 框架，侧重使用了 chain 模块和 agent 模块，和最新的 LECL 声明式编程。

Agent 部分代码：
```
def agent_generate_bvid_resources(bvid):
    # 工具集
    tool_list = [mp4_2_mp3, mp3_2_text, summarize_text, create_bilibili_dynamic]

    model = ChatOpenAI()
    prompt_template = PromptTemplate(
        template="将视频文件{input},转换为音频，音频再转为文字,对文字进行总结摘要,将摘要发到b站。输出格式化为：{input}",
        input_variables=["input"],
    )

    agent = initialize_agent(tool_list, model, verbose=True, handle_parsing_errors=True,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    agent.invoke(prompt_template.format(input=bvid)).get("output")
```

LCEL 部分代码：
```
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
    

```


# 效果
## Blog 展示效果：
![alt text](YZy1Ox52bl.jpg)


## B 站动态效果:
![alt text](image.png)
