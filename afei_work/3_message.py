from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = ChatOpenAI(
    model="qwen3.6-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

system_msg = SystemMessage("You are a helpful assistant.")
# human_msg = HumanMessage("Hello, how are you?")

# 从 URL
human_msg = HumanMessage(
        "Describe the content of this image, url is https://unsplash.com/photos/a-close-up-view-of-the-earth-from-space-mV0dC0GzKIY")
    

# 与聊天模型一起使用
messages = [system_msg, human_msg]
response = model.invoke(messages)  # 返回 AIMessage
print(response.content)