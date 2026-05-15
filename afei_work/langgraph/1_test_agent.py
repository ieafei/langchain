from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

for chunk in client.runs.stream(
    None,  # 无线程运行
    "agent", # 助手名称。在langgraph.json中定义。
    input={
        "messages": [{
            "role": "human",
            "content": "什么是LangGraph？",
        }],
    },
    stream_mode="messages-tuple",
):
    print(f"接收类型为: {chunk.event} 的新事件...")
    print(chunk.data)
    print("\n\n")