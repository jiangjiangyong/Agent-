import torch
import json
import os

class Config:
    def __init__(self):
        # 路径设置
        self.base_dir = "/hy-tmp/bert_ner_project"
        self.data_dir = os.path.join(self.base_dir, "data")
        self.model_save_dir = os.path.join(self.base_dir, "models")
        
        # 训练参数
        self.model_name = "bert-base-chinese"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.epochs = 5
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        
        # 标签映射
        self.label2id = {}
        self.id2label = {}
        self.num_labels = 0

    def load_labels(self):
        label_path = os.path.join(self.data_dir, 'labels.json')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
                self.label2id = {l: i for i, l in enumerate(labels)}
                self.id2label = {i: l for i, l in enumerate(labels)}
                self.num_labels = len(labels)
            print(f"✅ 成功加载 {self.num_labels} 个标签")
        else:
            print(f"❌ 找不到标签文件: {label_path}")