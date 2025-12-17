# 文件名称: src/generate_labels.py
import json
import os

def generate():
    data_path = "data/train.json"
    all_tags = set()
    
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        # 逐行读取 JSON 对象 (适配 JSONL 格式)
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 提取字符级标签 (character_label)
            all_tags.update(item['character_label'])
            
    # 严谨排序：确保 O 在索引 0
    label_list = sorted(list(all_tags))
    if 'O' in label_list:
        label_list.remove('O')
        label_list.insert(0, 'O')
    else:
        label_list.insert(0, 'O')
        
    os.makedirs("data", exist_ok=True)
    with open("data/labels.json", "w", encoding='utf-8') as f:
        json.dump(label_list, f, indent=2, ensure_ascii=False)
        
    print(f"标签提取成功！共 {len(label_list)} 类。")
    print(f"标签清单: {label_list}")

if __name__ == "__main__":
    generate()