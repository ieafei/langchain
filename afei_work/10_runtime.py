from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

# 工具：读取用户偏好
@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id
    preferences = "The user prefers you to write a brief and polite email."
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]
    return preferences

# 工具：保存用户偏好
@tool
def save_user_preference(preference: str, runtime: ToolRuntime[Context]) -> str:
    """Save user's email preference to the store."""
    user_id = runtime.context.user_id
    if runtime.store:
        runtime.store.put(("users",), user_id, {"preferences": preference})
        return f"Saved preference for {user_id}"
    return "No store available"

# 创建 store 并预填数据
store = InMemoryStore()
store.put(("users",), "alice", {"preferences": "Alice likes concise, bullet-point emails."})

# 创建 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[fetch_user_email_preferences, save_user_preference],
    store=store,
    context=Context(user_id="alice"),
)

result = agent.invoke({"messages": [{"role": "user", "content": "Write me an email"}]})