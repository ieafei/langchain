from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore


# InMemoryStore 将数据保存到内存字典中。在生产环境中使用基于数据库的存储。
store = InMemoryStore() # [!code highlight]

@dataclass
class Context:
    user_id: str

# TypedDict 定义了供 LLM 使用的用户信息结构
class UserInfo(TypedDict):
    name: str

# 允许智能体更新用户信息的工具（适用于聊天应用）
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    # 访问 store - 与提供给 `create_agent` 的 store 相同
    store = runtime.store # [!code highlight]
    user_id = runtime.context.user_id # [!code highlight]
    # 在 store 中存储数据 (namespace, key, data)
    store.put(("users",), user_id, user_info) # [!code highlight]
    return "Successfully saved user info."

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[save_user_info],
    store=store, # [!code highlight]
    context_schema=Context
)

# 运行智能体
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # 在 context 中传入 user_id 以识别正在更新谁的信息
    context=Context(user_id="user_123") # [!code highlight]
)

# 您可以直接访问 store 来获取该值
store.get(("users",), "user_123").value