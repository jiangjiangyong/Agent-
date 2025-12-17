import os
import torch
import requests  
from flask import Flask, request, jsonify
from transformers import BertTokenizer

from src.config import Config
from src.model import Bert_CRF_Model

app = Flask(__name__)

# --- 1. 初始化 ---
cfg = Config()
cfg.load_labels()
tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
model = Bert_CRF_Model(num_labels=cfg.num_labels).to(cfg.device)

# 加载权重
MODEL_PATH = os.path.join(cfg.model_save_dir, "best_model.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.device))
    model.eval()
    print("✅ BERT-CRF 系统就绪")

# --- 2. 核心 NER 提取逻辑（内部复用） ---
def get_ner_entities(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=cfg.max_len, 
                       truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(cfg.device)
    mask = inputs['attention_mask'].to(cfg.device)

    with torch.no_grad():
        best_path = model(input_ids, mask)
        valid_len = min(len(text), cfg.max_len - 2)
        tags = [cfg.id2label[idx] for idx in best_path[0][1:valid_len+1]]

    entities = []
    curr_ent, curr_type = "", ""
    for char, tag in zip(text[:valid_len], tags):
        if tag.startswith('B-'):
            if curr_ent: entities.append({"entity": curr_ent, "type": curr_type})
            curr_ent, curr_type = char, tag.split('-')[1]
        elif tag.startswith('I-') and curr_type == tag.split('-')[1]:
            curr_ent += char
        else:
            if curr_ent: entities.append({"entity": curr_ent, "type": curr_type})
            curr_ent, curr_type = "", ""
    if curr_ent: entities.append({"entity": curr_ent, "type": curr_type})
    return entities

# --- 3. 新增 Agent 对话接口 ---
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('text', '')
    if not user_input:
        return jsonify({"status": "error", "message": "内容为空"})

    # 第一步：提取实体
    entities = get_ner_entities(user_input)
    
    # 第二步：构造 Prompt
    prompt = f"""你是一个专业的医疗AI助手。
用户咨询内容："{user_input}"
系统从文中提取出的关键医疗实体：{entities}

请根据上述实体和用户描述，结合医学常识给出建议。
要求：
1. 逻辑校验：如果症状部位与疾病常识不符（如左侧腹痛怀疑阑尾炎），必须专业地予以纠正。
2. 用药提醒：对文中提到的药物给出必要的风险提示。
3. 风险警示：如果症状描述危急，提醒用户立即线下就医。
"""

    # 第三步：请求 DeepSeek API
    try:
        # 修改点 1：使用最新的 API 地址
        api_url = "https://api.deepseek.com/chat/completions"
        # 修改点 2：填入你真实的 API KEY
        api_key = "sk-1b8258e7a04348aabc004d8e7783e089" 

        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个严谨、专业的医疗AI助手。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,  # 适度的创造力
                "stream": False
            },
            timeout=30  # 修改点 3：增加超时设置，防止请求卡死
        )

        # 调试用：如果 API 返回非 200，打印报错详情到控制台
        if response.status_code != 200:
            print(f"DeepSeek API 报错: {response.text}")
            return jsonify({
                "status": "partial_success",
                "entities": entities,
                "reply": f"对话引擎响应异常(Code: {response.status_code})，请检查 API Key 余额。"
            })

        agent_reply = response.json()['choices'][0]['message']['content']

    except Exception as e:
        print(f"代码逻辑异常: {str(e)}") # 后台打印具体报错
        agent_reply = f"抱歉，系统处理超时或网络连接失败。提取到的实体有：{entities}"

    return jsonify({
        "status": "success",
        "entities": entities,
        "reply": agent_reply
    })
# 保留原有的纯预测接口，方便调试
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    return jsonify({"entities": get_ner_entities(text)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)