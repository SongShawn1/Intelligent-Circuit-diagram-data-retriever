---
title: 电路图资料导航
emoji: 🔌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.0
app_file: main.py
pinned: false
license: mit
---

# 🔌 智能车辆电路图资料导航 Chatbot

基于导航树检索 + BGE Reranker + LLM Query Rewriter 的智能电路图资料检索系统。

## 🌐 在线体验

**Hugging Face Spaces**: [https://huggingface.co/spaces/lengyanbb/Intelligent_circuit_diagram_data_retriever](https://huggingface.co/spaces/lengyanbb/Intelligent_circuit_diagram_data_retriever)

## ✨ 功能特点

- 🤖 **智能对话**: 自然语言查询，交互式导航
- 🌲 **导航树检索**: 基于层级目录结构的高效检索（1944 节点，4229 文件）
- 📊 **BGE Reranker**: 精排模型提升结果相关性
- 🔄 **LLM Query Rewriter**: 智能查询改写，理解用户意图
- 💡 **交互式引导**: 结果过多时提供筛选选项，帮助用户精确定位
- 🌐 **Gradio 界面**: 现代化聊天界面，支持选项点选和返回导航

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web UI                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 聊天界面    │  │ 选项点选    │  │ 返回导航    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                  Navigation Chatbot 对话层                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Query        │  │ BGE          │  │ Context      │      │
│  │ Rewriter     │  │ Reranker     │  │ Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    导航树检索引擎                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │ NavigationTree (1944 节点) + 关键词匹配           │      │
│  └──────────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    数据层                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │ structured_data_llm.json (4229 电路图资料)        │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/SongShawn1/Intelligent-Circuit-diagram-data-retriever.git
cd Intelligent-Circuit-diagram-data-retriever
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的智谱 API Key
```

### 4. 启动应用

```bash
python run.py
```

或直接运行 Gradio 界面：

```bash
python app/gradio_ui.py
```

访问 http://localhost:7861 即可使用。

## 📁 项目结构

```
电路图检索/
├── main.py                     # 🚀 HF Spaces 入口
├── run.py                      # 🚀 本地启动脚本
├── requirements.txt            # 📦 依赖列表
├── README.md                   # 📖 项目说明
├── .env.example                # ⚙️ 环境变量模板
├── Dockerfile                  # 🐳 Docker 配置
│
├── app/                        # 🖥️ 应用层
│   ├── gradio_ui.py            #    Gradio Web 界面
│   └── chatbot.py              #    导航 Chatbot 核心
│
├── core/                       # ⚙️ 核心模块
│   ├── navigation_tree.py      #    导航树构建与检索
│   ├── query_rewriter.py       #    LLM 查询改写器
│   ├── reranker.py             #    BGE 重排序模型
│   ├── cache.py                #    结果缓存
│   └── logger.py               #    日志工具
│
├── data/                       # 📊 数据层
│   ├── structured_data_llm.json#    结构化电路图数据
│   └── 资料清单.csv            #    原始数据
│
├── cache/                      # 💾 缓存目录
└── logs/                       # 📝 日志目录
```

## 🔧 核心技术

### 🌲 导航树检索
基于电路图资料的层级目录结构构建导航树：
- **高效遍历**: 支持按品牌、车系、模块快速定位
- **关键词匹配**: 多关键词模糊匹配
- **路径导航**: 支持返回上级、重新选择

### 🔄 LLM Query Rewriter
使用智谱 GLM-4-flash 智能改写用户查询：
- **意图理解**: 识别用户真实需求
- **关键词提取**: 提取品牌、车型、系统等关键信息
- **查询优化**: 生成更精准的检索关键词

### 📊 BGE Reranker
使用 BGE-Reranker 模型对检索结果精排：
- **语义相关性**: 深度理解查询与文档的相关性
- **结果优化**: 将最相关的结果排在前面

### 💡 交互式引导
当结果过多时，系统会：
- 分析结果中的关键词分布
- 提供筛选选项供用户点选
- 支持多轮交互逐步缩小范围

## 💬 查询示例

| 查询 | 说明 |
|------|------|
| 东风天龙整车电路图 | 品牌 + 车型 + 类型 |
| 博世EDC17 | 发动机ECU型号 |
| 解放J6P仪表 | 品牌 + 车型 + 模块 |
| 潍柴发动机线路图 | 发动机品牌 + 类型 |
| 后处理系统 | 系统模块查询 |

## ⚙️ 环境变量

| 变量名 | 说明 | 必填 |
|--------|------|------|
| `ZHIPU_API_KEY` | 智谱 AI API Key（用于 LLM Query Rewriter） | 是 |

## 📄 许可证

MIT License
