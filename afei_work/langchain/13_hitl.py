from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import HumanInTheLoopMiddleware # [!code highlight]
from langgraph.checkpoint.memory import InMemorySaver # [!code highlight]

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

agent = create_agent(
    model=model,
    middleware=[HumanInTheLoopMiddleware({"search": True})],
)

from langgraph.types import Command

# 人在回路利用 LangGraph 的持久化层。
# 您必须提供一个线程 ID (thread ID) 以将执行与对话线程关联起来，
# 从而使对话能够暂停和恢复（这对于人工审查是必需的）。
config = {"configurable": {"thread_id": "some_id"}} # [!code highlight]
# 运行图直到遇到中断。
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config # [!code highlight]
)

# 中断包含完整的 HITL 请求，带有 action_requests 和 review_configs
print(result['__interrupt__'])
# > [
# >     Interrupt(
# >         value={
# >           'action_requests': [
# >               {
# >                   'name': 'execute_sql',
# >                   'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
# >                   'description': 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
# >               }
# >            ],
# >            'review_configs': [
# >               {
# >                    'action_name': 'execute_sql',
# >                    'allowed_decisions': ['approve', 'reject']
# >               }
# >            ]
# >         }
# >     )
# > ]


# 以批准决策恢复
agent.invoke(
    Command( # [!code highlight]
        resume={"decisions": [{"type": "approve"}]}  # 或 "edit", "reject" [!code highlight]
    ), # [!code highlight]
    config=config # 相同的线程 ID 以恢复暂停的对话
)

{
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "tool_name",
                "args": {"param": "new_value"}
            }
        },
        {
            "type": "reject",
            "message": "This action is not allowed"
        }
    ]
}