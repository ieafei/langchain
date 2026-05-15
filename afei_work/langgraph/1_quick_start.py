from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """将`a`和`b`相乘。

    参数：
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """将`a`和`b`相加。

    参数：
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """将`a`除以`b`。

    参数：
        a: 第一个整数
        b: 第二个整数
    """
    return a / b


# 增强LLM的工具能力
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: dict):
    """LLM决定是否调用工具"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="你是一个有用的助手，负责对一组输入执行算术运算。"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: dict):
    """执行工具调用"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """决定是否继续循环或停止，基于LLM是否进行了工具调用"""

    messages = state["messages"]
    last_message = messages[-1]

    # 如果LLM进行了工具调用，则执行操作
    if last_message.tool_calls:
        return "tool_node"

    # 否则，我们停止（回复用户）
    return END

# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 添加边连接节点
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# 编译代理
agent = agent_builder.compile()

# 显示代理
# 替换 display 那两行为：
with open("./1_agent_graph.png", "wb") as f:
    f.write(agent.get_graph(xray=True).draw_mermaid_png())

# 调用
from langchain.messages import HumanMessage
messages = [HumanMessage(content="3加4等于多少。")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()