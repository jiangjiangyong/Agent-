# 文件名称: src/utils.py
import logging
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

def setup_logger(config):
    """
    配置全局Logger，同时输出到控制台和文件
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("BERT_NER")
    logger.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.info(f"日志系统初始化完成。日志文件路径: {log_file}")
    logger.info(f"硬件环境: {torch.cuda.get_device_name(0)} (显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    
    return logger

def set_seed(seed):
    """
    严格设定随机种子，确保实验可复现（Reproducibility）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性算法会降低一点速度，但保证严谨性
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False