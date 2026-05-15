# 访问多个 MCP 服务器
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",  # 本地子进程通信
                "command": "python",
                # 您的 math_server.py 文件的绝对路径
                "args": ["./12_mcp_server.py"],
            }
        }
    )

    model = ChatOpenAI(
        model="qwen3.6-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
        api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
    )

    tools = await client.get_tools()
    print(tools)
    agent = create_agent(
        model=model,
        tools=tools
    )
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    print(math_response.messages[-1].content)
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    print(weather_response.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())