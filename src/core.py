import torch
import torch.nn as nn
import json, os
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

class Bert_CRF_Model(nn.Module):
    def __init__(self, num_labels):
        super(Bert_CRF_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        # batch_first=True 匹配我们的数据格式
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        # CRF 需要 mask 是 bool 类型
        mask = attention_mask.to(torch.uint8).bool()

        if labels is not None:
            # 训练模式：返回负对数似然损失
            log_likelihood = self.crf(logits, labels, mask=mask, reduction='mean')
            return -log_likelihood
        else:
            # 推理模式：返回最优路径序列
            prediction = self.crf.decode(logits, mask=mask)
            return prediction

class Config:
    def __init__(self):
        self.model_name = "bert-base-chinese"
        self.data_dir = "data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 128
        self.batch_size = 32
        self.lr = 2e-5
        self.epochs = 5
        self.label2id = {}
        self.id2label = {}

    def load_labels(self):
        with open(os.path.join(self.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            labels = json.load(f)
            self.label2id = {l: i for i, l in enumerate(labels)}
            self.id2label = {i: l for i, l in enumerate(labels)}
            self.num_labels = len(labels)

class NERDataset(Dataset):
    def __init__(self, path, tokenizer, cfg):
        self.data = []
        # 确保 'O' 标签存在，CRF 填充必须使用有效 ID
        o_label_id = cfg.label2id.get('O', 0) 
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                encoding = tokenizer(item['character'], 
                                     is_split_into_words=True, 
                                     max_length=cfg.max_len, 
                                     truncation=True, 
                                     padding='max_length',
                                     return_tensors='pt')
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)

                # 标签切片并转换为 ID
                labels = [cfg.label2id.get(t, o_label_id) for t in item['character_label']][:cfg.max_len-2]
                
                # 构造全长度标签序列 (不含 -100)
                full_labels = [o_label_id] + labels + [o_label_id]
                padding_len = cfg.max_len - len(full_labels)
                full_labels = full_labels + [o_label_id] * padding_len
                
                self.data.append({
                    'input_ids': input_ids, 
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(full_labels)
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]