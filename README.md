Medical-NER-Agent: 医疗命名实体识别与智能问诊系统
本项目是一个结合了 本地轻量级感知模型 (BERT-CRF) 与 云端大语言模型 (DeepSeek) 的医疗 AI 智能体系统。通过自研微调的实体识别模型精准锁定病灶与药物信息，并联动 LLM 给出逻辑严密的医学建议。
核心架构亮点
BERT-CRF 架构：在 BERT 基础上增加 CRF (条件随机场) 层，利用 Viterbi 算法进行全局标签解码。相比纯 BERT 方案，解决了医疗词汇识别不完整、标签非法跳转等问题，显著提升了实体的逻辑性。
大小模型协同：本地部署的 BERT-CRF 负责毫秒级实体提取（隐私、高效）；云端 DeepSeek 负责深度推理（逻辑纠错、常识补全）。
医学常识校准：Agent 具备基础逻辑纠错能力。例如，当用户描述“左腹痛”但怀疑“阑尾炎”时，系统能敏锐捕捉部位与疾病的逻辑矛盾并予以专业提醒。

目录结构说明
.
├── src/
│   ├── app.py           # Flask 后端服务 (集成 BERT 与 DeepSeek 接口)
│   ├── model.py         # BERT-CRF 模型架构定义
│   ├── config.py        # 训练及推理超参数配置
│   ├── train.py         # 模型微调训练脚本
│   └── utils.py         # 日志与种子工具
├── data/                # 训练数据及标签定义 (labels.json)
├── models/              # 存放训练好的 best_model.pth 权重
├── medical_agent.py     # 离线测试 Agent 脚本
└── README.md            # 项目文档

快速开始
环境准备
git clone https://github.com/jiangjiangyong/Agent-.git
cd Agent-
pip install torch transformers flask requests pytorch-crf seqeval
transformers建议4.51.3

配置 API Key
在 src/app.py 中填入你的 DeepSeek API Key：
api_key = "sk-xxxxxxxxxxxxxxxxxxxx"

启动问诊后端
python -m src.app
服务默认运行在 http://localhost:8080

/chat请求格式post请求
{
  "text": "我最近左侧腹部剧烈疼痛，吃了阿司匹林好像也没什么用，是不是得阑尾炎了？"
}

训练表现
项目在 CCKS 医疗数据集上进行了微调：

基座模型: bert-base-chinese
优化器: AdamW (LR=2e-5)
最佳 F1 Score: 0.7444

后续规划
接入 RAG (检索增强生成)：结合权威医学百科知识库。
增加 多轮对话上下文 记忆功能。
导出为 ONNX 格式以提升推理速度。

免责声明：本项目生成内容仅供参考，不构成任何医疗诊断建议。

























































































