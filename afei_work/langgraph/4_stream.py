from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# stream()方法返回一个迭代器，生成流式输出
for chunk in graph.stream(
    {"topic": "ice cream"},
    # 设置stream_mode="updates"以仅流式传输每个节点后图状态的更新
    # 也可以使用其他流式传输模式。详见支持的流式传输模式
    stream_mode="updates",
):
    print(chunk)