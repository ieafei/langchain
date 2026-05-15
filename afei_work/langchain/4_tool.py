from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"

@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

print(search.name)

from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")


from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, AgentMiddleware
from langchain.agents.middleware.types import AgentState
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage, AIMessage
from typing import NotRequired, Any
from langgraph.runtime import Runtime
import dotenv

dotenv.load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    punny_response: str
    weather_conditions: str | None = None


# ① 定义扩展的 State，继承 AgentState 并添加自定义字段
class WeatherAgentState(AgentState[ResponseFormat]):
    query_count: NotRequired[int]          # 本次对话查询了几次天气
    last_queried_city: NotRequired[str]    # 最后查询的城市


# ② 定义 Middleware，绑定自定义 state_schema
class WeatherTrackingMiddleware(AgentMiddleware[WeatherAgentState, Context, ResponseFormat]):
    
    state_schema = WeatherAgentState   # ← 关键：绑定自定义 state

    def before_model(
        self, state: WeatherAgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """每次调用模型前，打印当前查询统计。"""
        count = state.get("query_count", 0)
        city = state.get("last_queried_city", "无")
        print(f"[Middleware] 当前查询次数: {count}, 最后查询城市: {city}")
        return None   # 不修改 state，返回 None

    def after_model(
        self, state: WeatherAgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """模型调用后，检查是否发起了天气查询，更新计数。"""
        messages = state.get("messages", [])
        if not messages:
            return None

        # 找最后一条 AIMessage
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if not last_ai or not last_ai.tool_calls:
            return None

        # 检查是否调用了 get_weather_for_location
        weather_calls = [
            tc for tc in last_ai.tool_calls
            if tc["name"] == "get_weather_for_location"
        ]
        if not weather_calls:
            return None

        # ③ 返回 state 更新（框架自动合并进 state）
        current_count = state.get("query_count", 0)
        last_city = weather_calls[-1]["args"].get("city", "未知")
        return {
            "query_count": current_count + len(weather_calls),
            "last_queried_city": last_city,
        }


# ④ 工具中访问自定义 state 字段
@tool
def get_weather_for_location(city: str, runtime: ToolRuntime[Context]) -> str:
    """获取指定城市的天气。"""
    count = runtime.state.get("query_count", 0)
    print(f"[Tool] 这是第 {count + 1} 次天气查询，城市: {city}")
    return f"{city}总是阳光明媚！"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


llm = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41",
    extra_body={"enable_thinking": False},
)

agent = create_agent(
    model=llm,
    tools=[get_user_location, get_weather_for_location],
    system_prompt="你是一位擅长用双关语表达的专家天气预报员。",
    response_format=ResponseFormat,
    context_schema=Context,
    middleware=[WeatherTrackingMiddleware()],   # ← 传入中间件实例
)

config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]},
    config=config,
    context=Context(user_id="1")
)
print(response["structured_response"])