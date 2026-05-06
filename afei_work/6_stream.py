from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer 

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

# def get_weather(city: str) -> str:
#     """获取给定城市的天气。"""

#     return f"It's always sunny in {city}!"

# agent = create_agent(
#     model=model,
#     tools=[get_weather],
# )
# # for chunk in agent.stream(  # [!code highlight]
# #     {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
# #     stream_mode="updates",
# # ):
# #     for step, data in chunk.items():
# #         print(f"step: {step}")
# #         print(f"content: {data['messages'][-1].content_blocks}")

# for token, metadata in agent.stream(  # [!code highlight]
#     {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
#     stream_mode="messages",
# ):
#     print(f"node: {metadata['langgraph_node']}")
#     print(f"content: {token.content_blocks}")
#     print("\n")

def get_weather(city: str) -> str:
    """获取给定城市的天气。"""
    writer = get_stream_writer()  # [!code highlight]
    # 流式传输任何任意数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"  # [!code highlight]
):
    print(chunk)