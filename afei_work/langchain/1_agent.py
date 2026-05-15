from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
import dotenv

dotenv.load_dotenv()


SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""


@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

# 这里使用 dataclass，但也支持 Pydantic 模型。
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None

@tool
def get_weather_for_location(city: str, runtime: ToolRuntime[Context]) -> str:
    """获取指定城市的天气。"""
    print(f"agent state: {runtime.state}")
    return f"{city}总是阳光明媚！"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        result = handler(request)
        return result
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


llm = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
    extra_body={"enable_thinking": False},  # 关闭思考模式，避免与 tool_choice 冲突
)

agent = create_agent(
    model=llm,
    tools=[get_user_location, get_weather_for_location],
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseFormat, 
    context_schema=Context,
    middleware=[handle_tool_errors]
)

# `thread_id` 是给定对话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])

