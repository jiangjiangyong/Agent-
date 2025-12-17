# 文件名称: src/inference.py
import torch
import json
import os
from transformers import BertTokenizer
from src.config import Config
from src.model import BertForNER

class MedicalNERPredictor:
    def __init__(self):
        self.cfg = Config()
        
        # 1. 加载标签映射
        label_path = os.path.join(self.cfg.data_dir, 'labels.json')
        with open(label_path, 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        self.cfg.update_labels(label_list)
        
        # 2. 初始化模型与权重
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model_name)
        self.model = BertForNER.from_pretrained(self.cfg.model_name, num_labels=self.cfg.num_labels)
        
        model_path = os.path.join(self.cfg.model_save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.cfg.device))
        self.model.to(self.cfg.device)
        self.model.eval()

    def predict(self, text):
        tokens = list(text)
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.cfg.max_len, 
                                 truncation=True, padding='max_length')
        
        input_ids = inputs["input_ids"].to(self.cfg.device)
        attention_mask = inputs["attention_mask"].to(self.cfg.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2).cpu().numpy()[0]

        # 将 ID 映射回标签
        # 注意：BERT 的输入包含 [CLS]，所以预测索引要对应
        results = []
        for i, char in enumerate(tokens):
            if i + 1 < len(preds): # 避开 [CLS]
                label = self.cfg.id2label[preds[i+1]]
                if label != 'O':
                    results.append((char, label))
        return results

if __name__ == "__main__":
    predictor = MedicalNERPredictor()
    
    # 压力测试文本
    stress_test_text = (
        "患者于2023年5月在北京市协和医院确诊为慢性阻塞性肺疾病急性加重期，伴随呼吸衰竭症状。"
        "入院后给予静脉注射甲泼尼龙琥珀酸钠治疗，并行支气管镜肺泡灌洗术。"
        "患者既往有高血压病史十年，长期服用氨氯地平片，目前神志清，双肺呼吸音粗。"
    )
    
    print(f"\n--- 压力测试开始 ---")
    print(f"输入文本: {stress_test_text}\n")
    
    res = predictor.predict(stress_test_text)
    
    # 增强版实体聚合逻辑（支持打印所有标签）
    entities = []
    current_entity = ""
    current_type = ""
    
    for char, tag in res:
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = tag.split('-')[1]
        elif tag.startswith('I-'):
            if current_type == tag.split('-')[1]:
                current_entity += char
            else:
                # 处理可能出现的标注不连续
                entities.append((current_entity, current_type))
                current_entity = char
                current_type = tag.split('-')[1]
                
    if current_entity:
        entities.append((current_entity, current_type))

    # 打印结果
    for ent, t in entities:
        print(f"[{t.ljust(6)}] : {ent}")