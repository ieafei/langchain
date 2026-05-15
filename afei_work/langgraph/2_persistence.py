from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str
    bar: list[str]

def node_a(state: State) -> State:
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State) -> State:
    return {"foo": "b", "bar": ["b"]}

# 创建图
graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

# 设置边
graph.add_edge("__start__", "node_a")
graph.add_edge("node_a", "node_b")

# 创建检查点
saver = InMemorySaver()

# 编译图时启用检查点
graph = graph.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "1"}}

# 先执行图，生成状态历史
result = graph.invoke({"foo": "", "bar": []}, config=config)
print("执行结果:", result)

# 然后获取状态历史
history = list(graph.get_state_history(config))

# 格式化状态历史以便更好阅读
def format_state_history(history):
    """格式化状态历史为易读的字符串"""
    output = []
    for i, snapshot in enumerate(history):
        output.append(f"步骤: {snapshot.metadata.get('step', 'N/A')}")
        output.append(f"值: {snapshot.values}")
        output.append(f"下一个节点: {snapshot.next}")
        output.append(f"创建时间: {snapshot.created_at}")
        output.append(f"来源: {snapshot.metadata.get('source', 'N/A')}")
        output.append("---")
    return "\n".join(output)

print("\n格式化后的状态历史:")
print(format_state_history(history))