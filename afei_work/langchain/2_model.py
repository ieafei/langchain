from langchain_openai import ChatOpenAI
from langchain.tools import tool
import dotenv

dotenv.load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    return f"{location} 天气晴朗。"

# inputs = [
#     "为什么鹦鹉有五颜六色的羽毛？",
#     "飞机是如何飞行的？",
#     "什么是量子计算？"
# ]

# # 用字典收集结果，key 为原始索引
# results = {}

# for idx, response in llm.batch_as_completed(inputs):
#     print(f"[已完成] 索引 {idx}：{inputs[idx]!r}")
#     results[idx] = response

# print("\n===== 按原始顺序输出 =====")
# for i in range(len(inputs)):
#     print(f"\n[{i}] 问题：{inputs[i]}")
#     print(f"    回答：{results[i].content}")

# 将（可能多个）工具绑定到模型
model_with_tools = model.bind_tools([get_weather])

# 步骤 1：模型生成工具调用
messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 步骤 2：执行工具并收集结果
for tool_call in ai_msg.tool_calls:
    # 使用生成的参数执行工具
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 步骤 3：将结果传递回模型以获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿当前天气为 72°F，晴朗。"
