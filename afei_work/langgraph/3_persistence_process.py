#!/usr/bin/env python3
"""
LangGraph interrupt + 持久执行示例（StateGraph 图方式）

核心演示：
  1. random_score 节点生成随机分数（每次调用结果不同）
  2. random_recommend 节点生成随机推荐（每次调用结果不同）
  3. interrupt 暂停，等待人工审批
  4. Command(resume=...) 恢复后，random_score 和 random_recommend 不会重新执行
     → 随机值保持不变，证明 checkpointer 持久化了已完成节点的结果
"""

import uuid
import random
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ========== 1. Define State ==========
class State(TypedDict):
    user_id: str
    random_score: Optional[int]             # random value from step 1
    random_recommendations: Optional[List[str]]  # random value from step 2
    human_approval: Optional[str]           # human input from interrupt
    final_result: Optional[Dict[str, Any]]


# ========== 2. Define Nodes ==========
def generate_random_score(state: State) -> dict:
    """Step 1: Generate a RANDOM score (non-deterministic)

    This node produces a different result every time it runs.
    After interrupt + resume, this node should NOT re-run.
    The checkpointed value should be reused.
    """
    score = random.randint(1, 100)
    print(f"[random_score] 🎲 生成随机评分: {score}  (每次运行都不同!)")
    return {"random_score": score}


CANDIDATE_POOL = [
    "Python入门", "机器学习实战", "深度学习原理", "数据结构与算法",
    "分布式系统", "微服务架构", "Kubernetes实战", "Go语言编程",
    "Rust系统编程", "前端工程化", "React高级模式", "数据库内核",
]

def generate_random_recommendations(state: State) -> dict:
    """Step 2: Generate RANDOM recommendations (non-deterministic)

    Randomly picks 3 items from a pool. Result differs every run.
    After interrupt + resume, this node should NOT re-run.
    """
    picks = random.sample(CANDIDATE_POOL, 3)
    print(f"[random_recommend] 🎲 随机推荐课程: {picks}  (每次运行都不同!)")
    return {"random_recommendations": picks}


def human_review(state: State) -> dict:
    """Step 3: Pause for human approval (interrupt)

    Shows the random score and recommendations to the human,
    then waits for approval.
    """
    score = state["random_score"]
    recs = state["random_recommendations"]

    # ★ interrupt() pauses the graph here
    answer = interrupt({
        "message": "请审批以下自动生成的结果",
        "random_score": score,
        "random_recommendations": recs,
        "instruction": "输入 'approve' 批准, 或输入拒绝原因",
    })

    print(f"[human_review] 收到人工审批: {answer}")
    return {"human_approval": answer}


def finalize(state: State) -> dict:
    """Step 4: Produce final result based on human decision"""
    approval = state.get("human_approval", "")
    score = state["random_score"]
    recs = state["random_recommendations"]

    if approval == "approve":
        result = {
            "status": "approved",
            "score": score,
            "recommendations": recs,
            "finalized_at": datetime.now().isoformat(),
        }
        print(f"[finalize] ✅ 审批通过")
    else:
        result = {
            "status": "rejected",
            "reason": approval,
            "score": score,
            "recommendations": recs,
            "finalized_at": datetime.now().isoformat(),
        }
        print(f"[finalize] ❌ 审批拒绝, 原因: {approval}")

    return {"final_result": result}


# ========== 3. Build Graph ==========
#
#  __start__ → random_score → random_recommend → human_review(interrupt) → finalize
#
def build_graph():
    checkpointer = InMemorySaver()

    builder = StateGraph(State)
    builder.add_node("random_score", generate_random_score)
    builder.add_node("random_recommend", generate_random_recommendations)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "random_score")
    builder.add_edge("random_score", "random_recommend")
    builder.add_edge("random_recommend", "human_review")
    builder.add_edge("human_review", "finalize")

    graph = builder.compile(checkpointer=checkpointer)
    return graph


# ========== 4. Demo ==========
def main():
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print("  LangGraph 持久执行示例 — 随机节点 + interrupt")
    print("  核心验证: resume 后随机节点不会重新执行, 值保持不变")
    print("=" * 70)

    # ────── Phase 1: Run until interrupt ──────
    print("\n【Phase 1】启动图, 随机节点执行后在 human_review 暂停\n")

    for chunk in graph.stream({"user_id": "test_001"}, config):
        print(f"  chunk: {chunk}")

    # Capture the random values that were generated
    state = graph.get_state(config)
    saved_score = state.values["random_score"]
    saved_recs = state.values["random_recommendations"]

    print(f"\n  ★ 图已暂停! 以下随机值已被 checkpointer 持久化保存:")
    print(f"    random_score         = {saved_score}")
    print(f"    random_recommendations = {saved_recs}")

    # ────── Phase 2: Prove randomness would differ ──────
    print("\n" + "-" * 70)
    print("【Phase 2】验证: 如果现在重新调用随机函数, 结果一定不同\n")

    # Call the random functions directly to show they produce different values
    new_score = random.randint(1, 100)
    new_recs = random.sample(CANDIDATE_POOL, 3)
    print(f"  重新生成 random_score         = {new_score}  {'(相同-巧合)' if new_score == saved_score else '(不同!)'}")
    print(f"  重新生成 random_recommendations = {new_recs}  {'(相同-巧合)' if new_recs == saved_recs else '(不同!)'}")
    print(f"\n  → 随机函数每次结果都不同, 但 resume 后应该使用保存的值")

    # ────── Phase 3: Resume and verify persistence ──────
    print("\n" + "-" * 70)
    print("【Phase 3】恢复执行, 验证随机值是否保持不变\n")

    print("  发送 Command(resume='approve') 恢复图...\n")

    for chunk in graph.stream(Command(resume="approve"), config):
        print(f"  chunk: {chunk}")

    # Check final state
    final_state = graph.get_state(config)
    final_score = final_state.values["random_score"]
    final_recs = final_state.values["random_recommendations"]

    print(f"\n  ★ 恢复后的状态:")
    print(f"    random_score         = {final_score}")
    print(f"    random_recommendations = {final_recs}")

    # ────── Phase 4: Verify ──────
    print("\n" + "-" * 70)
    print("【Phase 4】持久化验证\n")

    score_match = final_score == saved_score
    recs_match = final_recs == saved_recs

    print(f"  random_score 是否一致:          {'✅ YES' if score_match else '❌ NO'}  ({saved_score} → {final_score})")
    print(f"  random_recommendations 是否一致: {'✅ YES' if recs_match else '❌ NO'}  ({saved_recs} → {final_recs})")

    if score_match and recs_match:
        print(f"\n  🎉 持久执行验证成功!")
        print(f"     random_score 和 random_recommend 节点在 resume 后没有重新执行,")
        print(f"     checkpointer 保存的随机值被完整保留并传递给后续节点。")
    else:
        print(f"\n  ⚠️ 值发生了变化, 请检查 checkpointer 配置。")

    # ────── Checkpoint history ──────
    print(f"\n  检查点历史:")
    for i, snapshot in enumerate(graph.get_state_history(config)):
        step = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "?")
        next_nodes = snapshot.next
        vals = snapshot.values
        score_in_snapshot = vals.get("random_score", "-")
        print(f"    [{i}] step={step}, source={source}, next={next_nodes}, score={score_in_snapshot}")

    print("\n" + "=" * 70)
    print("  关键结论:")
    print("    1. random_score / random_recommend 节点含有随机性")
    print("    2. interrupt() 暂停图后, 已完成节点的结果被 checkpointer 保存")
    print("    3. Command(resume=...) 恢复后, 图从 human_review 继续执行")
    print("    4. 随机节点不会重新执行, 随机值保持不变 → 这就是持久执行")
    print("=" * 70)


if __name__ == "__main__":
    main()