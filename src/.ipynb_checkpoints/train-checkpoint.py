import os
import torch
import torch.optim as optim
import json
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.config import Config
from src.utils import setup_logger, set_seed
from src.model import Bert_CRF_Model, NERDataset

def compute_metrics(preds, labels, id2label):
    true_list = []
    pred_list = []
    for i in range(len(labels)):
        true_seq = [id2label[l] for l in labels[i]]
        pred_seq = [id2label[p] for p in preds[i]]
        true_list.append(true_seq)
        pred_list.append(pred_seq)
    
    return {
        "f1": f1_score(true_list, pred_list),
        "precision": precision_score(true_list, pred_list),
        "recall": recall_score(true_list, pred_list),
        "report": classification_report(true_list, pred_list)
    }

def train():
    cfg = Config()
    logger = setup_logger(cfg)
    set_seed(cfg.seed)
    
    label_path = os.path.join(cfg.data_dir, 'labels.json')
    with open(label_path, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    cfg.update_labels(label_list)
    
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    
    # 【注意】这里只传 3 个参数，匹配 model.py 中的定义
    train_dataset = NERDataset(os.path.join(cfg.data_dir, 'train.json'), tokenizer, cfg)
    val_dataset = NERDataset(os.path.join(cfg.data_dir, 'test.json'), tokenizer, cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    logger.info("初始化 Bert_CRF_Model...")
    model = Bert_CRF_Model(num_labels=cfg.num_labels)
    model.to(cfg.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    logger.info(f"开始训练...")
    best_f1 = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)
            
            model.zero_grad()
            # CRF 模型直接返回负对数似然 Loss
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        for batch in tqdm(val_loader, desc="评估"):
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            labels = batch['labels']
            
            with torch.no_grad():
                preds = model(input_ids, attention_mask=attention_mask)
            
            for i in range(len(preds)):
                # 根据 mask 长度截取有效标签，去掉 CLS 和 SEP
                true_len = attention_mask[i].sum().item()
                all_preds.append(preds[i][1:true_len-1])
                all_labels.append(labels[i][1:true_len-1].tolist())
            
        metrics = compute_metrics(all_preds, all_labels, cfg.id2label)
        logger.info(f"Epoch {epoch+1} F1: {metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(cfg.model_save_dir, 'best_model.pth'))
            logger.info("保存了最佳模型！")

if __name__ == "__main__":
    train()