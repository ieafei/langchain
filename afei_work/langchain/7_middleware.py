from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

agent = create_agent(
    model=model,
    tools=[...],
    middleware=[SummarizationMiddleware(), HumanInTheLoopMiddleware()],
)