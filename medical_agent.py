import os
import requests
import json
from openai import OpenAI

# 1. 初始化客户端 (DeepSeek 兼容 OpenAI 格式)
client = OpenAI(
    api_key="sk-1b8258e7a04348aabc004d8e7783e089", 
    base_url="https://api.deepseek.com"
)

# 2. 定义调用你本地 BERT NER 的函数
def extract_medical_entities(text):
    """调用本地 Flask 服务提取医疗实体"""
    url = "http://127.0.0.1:8080/predict"  # 确保你的 app.py 正在运行
    try:
        response = requests.post(url, json={"text": text}, timeout=5)
        if response.status_code == 200:
            return response.json().get("entities", [])
    except Exception as e:
        return f"工具调用失败: {str(e)}"
    return []

# 3. 配置工具描述 (让 DeepSeek 知道怎么用它)
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_medical_entities",
            "description": "精确提取文本中的医疗实体，如疾病(DISE)、药物(DRUG)、身体部位(BODY)、症状(SYMP)等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要分析的原始病历或患者描述文本"}
                },
                "required": ["text"]
            }
        }
    }
]

# 4. Agent 主循环
def run_medical_agent(user_query):
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手。在回答用户问题前，必须先调用实体提取工具分析病情。"},
        {"role": "user", "content": user_query}
    ]

    # 第一轮：DeepSeek 决定调用工具
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    
    msg = response.choices[0].message
    
    # 如果 DeepSeek 决定调用工具
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            # 解析参数并执行本地 BERT 推理
            args = json.loads(tool_call.function.arguments)
            print(f"--- Agent 正在调用 BERT NER 提取实体: {args['text']} ---")
            
            entities = extract_medical_entities(args['text'])
            print(f"--- BERT 返回结果: {entities} ---")
            
            # 将工具结果反馈给 DeepSeek
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(entities)
            })
        
        # 第二轮：DeepSeek 结合实体识别结果给出最终建议
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return final_response.choices[0].message.content

# 5. 测试运行
if __name__ == "__main__":
    query = "我最近左侧腹部剧烈疼痛，吃了阿司匹林好像也没什么用，是不是得阑尾炎了？"
    result = run_medical_agent(query)
    print("\n--- 医疗 Agent 最终回复 ---")
    print(result)