import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset
from transformers import BertModel
from torchcrf import CRF

class Bert_CRF_Model(nn.Module):
    def __init__(self, num_labels):
        super(Bert_CRF_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        mask = attention_mask.to(torch.uint8).bool()

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=mask, reduction='mean')
            return -log_likelihood
        else:
            return self.crf.decode(logits, mask=mask)

class NERDataset(Dataset):
    def __init__(self, path, tokenizer, cfg):
        self.data = []
        o_id = cfg.label2id.get('O', 0)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                encoding = tokenizer(item['character'], is_split_into_words=True, 
                                     max_length=cfg.max_len, truncation=True, 
                                     padding='max_length', return_tensors='pt')
                
                labels = [cfg.label2id.get(t, o_id) for t in item['character_label']][:cfg.max_len-2]
                full_labels = [o_id] + labels + [o_id]
                full_labels += [o_id] * (cfg.max_len - len(full_labels))
                
                self.data.append({
                    'input_ids': encoding['input_ids'].squeeze(0), 
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': torch.tensor(full_labels)
                })
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]