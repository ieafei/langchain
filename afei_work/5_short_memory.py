from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]
from langchain_openai import ChatOpenAI



model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

agent = create_agent(model, checkpointer=InMemorySaver())

response = agent.invoke({"messages": [{"role": "user", "content": "我叫Bob"}]},
             {"configurable": {"thread_id": "1"}})
print(response["messages"][-1].content)

response = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]},
             {"configurable": {"thread_id": "1"}})
print(response["messages"][-1].content)