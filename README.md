---
title: 电路图资料导航
emoji: 🔌
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🔌 智能车辆电路图资料导航 Chatbot

基于向量检索的智能电路图资料导航系统，支持自然语言查询。

## 功能特点

- 🤖 **智能对话**: 自然语言查询，支持模糊搜索和错别字纠正
- 🔍 **语义检索**: 基于 ChromaDB + 智谱 Embeddings 的向量检索
- 🎯 **Self-Querying**: LLM 解析查询为语义+结构化过滤
- 📊 **Reranker**: BGE 精排模型提升结果相关性
- 💡 **交互式兜底**: 低置信度时主动引导，帮助用户找到资料
- 🌐 **Web 界面**: Streamlit 构建的现代化聊天界面

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
├─────────────────────────────────────────────────────────────┤
│                  Industrial Chatbot 对话层                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Self-Query   │  │ Reranker     │  │ Decision     │      │
│  │ Parser       │  │ (BGE)        │  │ Engine       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    查询预处理器                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 错别字纠正   │  │ 同义词扩展   │  │ 领域术语扩展 │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    向量检索引擎                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │ ChromaDB (4229 文档) + 智谱 embedding-3          │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
python run.py
```

或直接运行：

```bash
streamlit run app/app_industrial.py --server.port 8503
```

访问 http://localhost:8503 即可使用。

### 3. 其他命令

```bash
# 运行检索测试
python run.py --test

# 重建向量数据库
python run.py --build
```

## 项目结构

```
电路图检索/
├── run.py                      # 🚀 启动脚本
├── requirements.txt            # 📦 依赖列表
├── README.md                   # 📖 项目说明
│
├── app/                        # 🖥️ 应用层
│   ├── app_industrial.py       #    Streamlit Web 界面
│   └── chatbot_industrial.py   #    工业级 Chatbot 核心
│
├── core/                       # ⚙️ 核心模块
│   ├── build_vector_db.py      #    向量数据库构建
│   ├── embedding_service.py    #    Embedding 服务
│   ├── self_query_parser.py    #    Self-Querying 解析器
│   ├── decision_engine.py      #    决策引擎 (Agentic RAG)
│   ├── reranker.py             #    重排序模型
│   ├── hybrid_search.py        #    🆕 混合检索 (Vector + BM25)
│   ├── query_expansion.py      #    🆕 查询扩展 (LLM驱动)
│   └── typo_corrector.py       #    错别字纠正
│
├── data/                       # 📊 数据层
│   ├── 资料清单.csv            #    原始数据
│   ├── structured_data_llm.json#    LLM 结构化数据
│   ├── keywords.txt            #    测试关键词
│   └── chroma_db/              #    向量数据库
│
├── tools/                      # 🔧 工具脚本
│   ├── preprocess_with_llm.py  #    LLM ETL 数据清洗
│   └── test_search.py          #    检索效果测试
│
└── config/                     # ⚙️ 配置文件
    └── 开发项目描述.pdf        #    项目需求文档
```

## 技术特性

### 🔄 Query Expansion 查询扩展
当检索结果不足时（如 LLM 解析错误导致零结果），自动触发查询扩展：
- **问题识别**：分析为何结果不足
- **查询改写**：LLM 生成更通用的查询
- **备选查询**：提供多个查询变体
- **典型场景**：用户说"东风天龙"但 LLM 错误解析为"东风柳汽"

### 🔍 Hybrid Search 混合检索
结合两种检索方式的优势：
- **向量检索**：语义理解，适合模糊查询
- **BM25 检索**：精确关键词匹配，适合品牌、型号查询
- **RRF 融合**：Reciprocal Rank Fusion 算法融合两种结果

### 🎯 Self-Querying Retrieval  
使用 LLM 自动解析用户查询为：
- 语义查询（用于向量搜索）
- 结构化过滤条件（品牌、车系、文档类型等）

### 🔄 渐进式过滤放宽
当过滤条件太严格时，自动逐步放宽条件以获得更多结果

## 查询示例

| 查询 | 说明 |
|------|------|
| 解放J6空调电路图 | 品牌 + 车型 + 系统 |
| 东风天龙保险丝盒 | 品牌 + 车型 + 部件 |
| 五十铃发动机电路图 | 品牌 + 系统 |
| 山西大运电器盒 | 子品牌精确匹配 |
| 潍柴WP10针脚图 | 发动机型号 + 文档类型 |
| VGT线路图 | 领域术语查询 |

## 优化效果

基于 22 条真实用户查询测试：

| 指标 | 原始查询 | 预处理后 | 提升 |
|------|---------|---------|------|
| 良好匹配(>0.4) | 18.2% | 40.9% | +22.7% |
| 精确匹配(>0.5) | 9.1% | 13.6% | +4.5% |

主要优化手段：
- 错别字纠正（庆龄→庆铃, 豪汉→豪瀚）
- 同义词扩展（保险丝→保险盒/熔断器）
- 品牌别名（庆铃→五十铃）
- 领域术语（VGT→涡轮增压）
- 精确查询保护（避免过度扩展）

## 配置

创建 `.env` 文件配置 OpenAI API（可选）：

```
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1
```

如果不配置 OpenAI，系统会自动使用本地 BGE 模型。

## 许可证

MIT License
