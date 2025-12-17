# 文件名称: src/download_data.py
import json
import os
import requests
from config import Config
from utils import setup_logger
from tqdm import tqdm

# 定义多个镜像源，增加严谨性
MIRRORS = [
    "https://raw.gitmirror.com/",
    "https://ghproxy.net/https://raw.githubusercontent.com/",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/"
]

def download_with_retry(github_path, save_path, logger):
    """
    带重试和源切换的下载逻辑
    github_path: 用户名/仓库名/分支/路径
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for mirror in MIRRORS:
        url = f"{mirror}{github_path}"
        logger.info(f"正在尝试从镜像源下载: {url}")
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=20)
            if resp.status_code == 200:
                content = resp.content
                # 严谨性检查：判断是否为合法的 JSON 起始符
                first_char = content.strip()[:1]
                if first_char not in [b'[', b'{']:
                    logger.warning(f"镜像源 {mirror} 返回了非法内容（可能是HTML），尝试下一个...")
                    continue
                
                with open(save_path, 'wb') as f:
                    f.write(content)
                logger.info(f"文件下载成功: {os.path.basename(save_path)}")
                return True
        except Exception as e:
            logger.warning(f"源 {mirror} 请求异常: {e}")
            continue
            
    return False

def process_cmeee_json(raw_path, target_path, logger):
    """
    清洗 CMeEE-V2 原始数据
    """
    logger.info(f"正在清洗数据: {os.path.basename(raw_path)}")
    try:
        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed = []
        for item in raw_data:
            text = item.get('text', '')
            tokens = list(text)
            labels = ["O"] * len(tokens)
            entities = item.get('entities', [])
            for ent in entities:
                start, end, label_type = ent['start_idx'], ent['end_idx'], ent['type']
                if start < len(labels) and end <= len(labels):
                    labels[start] = f"B-{label_type}"
                    for i in range(start + 1, end):
                        labels[i] = f"I-{label_type}"
            processed.append({"tokens": tokens, "ner_tags": labels})
        
        with open(target_path, 'w', encoding='utf-8') as f:
            for entry in processed:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"解析/转换失败: {e}")
        raise e

def main():
    cfg = Config()
    logger = setup_logger(cfg)
    
    # 目标文件的 Github 相对路径
    repo_base = "hs3434/CMeEE-V2/master/"
    files = {
        "CMeEE-V2_train.json": "train.json",
        "CMeEE-V2_dev.json": "validation.json",
        "CMeEE-V2_test.json": "test.json"
    }
    
    os.makedirs(cfg.data_dir, exist_ok=True)
    temp_dir = os.path.join(cfg.data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    all_tags = set()
    for remote_name, local_name in files.items():
        temp_path = os.path.join(temp_dir, remote_name)
        final_path = os.path.join(cfg.data_dir, local_name)
        
        if download_with_retry(repo_base + remote_name, temp_path, logger):
            process_cmeee_json(temp_path, final_path, logger)
            with open(final_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_tags.update(json.loads(line)['ner_tags'])
        else:
            logger.error("所有镜像源下载均失败，请检查网络环境或手动上传数据。")
            return

    # 生成标签文件
    label_list = sorted(list(all_tags))
    if 'O' in label_list: label_list.remove('O')
    label_list.insert(0, 'O')
    
    with open(os.path.join(cfg.data_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(label_list, f, indent=2)
    logger.info(f"数据准备完成！标签集: {label_list}")

if __name__ == "__main__":
    main()