# 文件名称: src/dataset.py
import torch
from torch.utils.data import Dataset
import json

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, label2id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.data = self._load_data(file_path)
        
    def _load_data(self, path):
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                
                # 精准映射 HealthNER 的字段
                # tokens -> character, labels -> character_label
                tokens = item['character']
                tags = item['character_label']
                
                tag_ids = [self.label2id.get(t, self.label2id.get('O', 0)) for t in tags]
                data_list.append({'tokens': tokens, 'ner_tags': tag_ids})
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = item['tokens']     
        label_ids = item['ner_tags'] 
        
        # 编码逻辑 (保持严谨对齐)
        token_ids = [self.tokenizer.cls_token_id]
        target_labels = [-100]
        
        for t, l in zip(tokens, label_ids):
            w_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t))
            if not w_ids: continue
            token_ids.append(w_ids[0])
            target_labels.append(l)
            # 处理多出来的 subwords
            for extra in w_ids[1:]:
                token_ids.append(extra)
                target_labels.append(-100)
                
        # 截断与 Padding
        token_ids = token_ids[:self.max_len-1] + [self.tokenizer.sep_token_id]
        target_labels = target_labels[:self.max_len-1] + [-100]
        
        seq_len = len(token_ids)
        mask = [1] * seq_len + [0] * (self.max_len - seq_len)
        token_ids += [self.tokenizer.pad_token_id] * (self.max_len - seq_len)
        target_labels += [-100] * (self.max_len - seq_len)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(target_labels, dtype=torch.long)
        }