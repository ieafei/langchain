from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen3.6-flash",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 自定义API地址
    api_key="sk-fbd837c733ba4ce5a15d4a5555c27f41", # 自定义API Key
)

response = llm.invoke("Hello, who are you?")
print(response)
