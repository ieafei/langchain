from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # [!code highlight]
from langchain_openai import ChatOpenAI



# model = ChatOpenAI(
#     model="qwen3.6-flash",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
#     api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
# )

# # agent = create_agent(model, checkpointer=InMemorySaver())

# # response = agent.invoke({"messages": [{"role": "user", "content": "我叫Bob"}]},
# #              {"configurable": {"thread_id": "1"}})
# # print(response["messages"][-1].content)

# # response = agent.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]},
# #              {"configurable": {"thread_id": "1"}})
# # print(response["messages"][-1].content)


# from langchain.messages import RemoveMessage
# from langgraph.graph.message import REMOVE_ALL_MESSAGES
# from langgraph.checkpoint.memory import InMemorySaver
# from langchain.agents import create_agent, AgentState
# from langchain.agents.middleware import before_model
# from langgraph.runtime import Runtime
# from langchain_core.runnables import RunnableConfig
# from typing import Any


# @before_model
# def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#     """Keep only the last few messages to fit context window."""
#     messages = state["messages"]

#     if len(messages) <= 3:
#         return None  # No changes needed

#     first_msg = messages[0]
#     recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
#     new_messages = [first_msg] + recent_messages

#     return {
#         "messages": [
#             RemoveMessage(id=REMOVE_ALL_MESSAGES),
#             *new_messages
#         ]
#     }

# agent = create_agent(
#     model,
#     middleware=[trim_messages],
#     checkpointer=InMemorySaver(),
# )

# config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# agent.invoke({"messages": "hi, my name is bob"}, config)
# agent.invoke({"messages": "write a short poem about cats"}, config)
# agent.invoke({"messages": "now do the same but for dogs"}, config)
# final_response = agent.invoke({"messages": "what's my name?"}, config)

# final_response["messages"][-1].pretty_print()
# """
# ================================== Ai Message ==================================

# Your name is Bob. You told me that earlier.
# If you'd like me to call you a nickname or use a different name, just say the word.
# """


from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id  # [!code highlight]
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state["user_name"]
    return f"Hello {user_name}!"


agent = create_agent(
    model=model,
    tools=[get_user_info, update_user_info, greet],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.