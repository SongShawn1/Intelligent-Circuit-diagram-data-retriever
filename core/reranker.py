#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reranker 模块 - 使用 Cross-Encoder 对检索结果进行精排

工作原理：
1. 向量搜索（Bi-Encoder）是粗排：独立编码 Query 和 Doc，速度快但精度有限
2. Reranker（Cross-Encoder）是精排：同时编码 Query-Doc 对，精度高但速度慢

典型流程：
    Query → Vector Search (Top-50) → Reranker 精排 → 返回 Top-10

支持的 Reranker：
1. BGE Reranker（本地模型）- 需要 GPU 或大内存
2. API Reranker（智谱/Cohere）- 轻量级，推荐
"""

import os
import logging

# 禁用 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Reranker 基类"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], k: int = 10) -> List[Dict]:
        """
        对文档列表进行重排序
        
        Args:
            query: 用户查询
            documents: 文档列表，每个文档是 dict，包含 content/metadata
            k: 返回 top-k 个结果
            
        Returns:
            重排序后的文档列表
        """
        raise NotImplementedError


class BgeReranker(BaseReranker):
    """
    BGE Reranker - 使用本地 Cross-Encoder 模型
    
    模型选择：
    - BAAI/bge-reranker-base: 中等大小，平衡性能
    - BAAI/bge-reranker-large: 更大更准确
    - BAAI/bge-reranker-v2-m3: 多语言版本
    """
    
    def __init__(self, model_path: str = 'BAAI/bge-reranker-base'):
        """
        初始化 BGE Reranker
        
        Args:
            model_path: 模型路径或 HuggingFace 模型名
        """
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = None
        self._use_fp16 = False  # 是否使用半精度加速
        
    def _ensure_loaded(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
            
    def _load_model(self):
        """加载模型和分词器"""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"🔄 加载 Reranker 模型: {self.model_path}")
        
        # 选择设备 - 对于小批量推理，CPU 可能比 MPS 更快（避免设备同步开销）
        # 强制使用 CPU 以获得更稳定的性能
        self._device = torch.device("cpu")
        self._use_fp16 = False
        print("  └─ 使用 CPU (小批量优化)")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32  # CPU 使用 FP32
        )
        self._model.to(self._device)
        self._model.eval()
        
        print("  └─ ✅ Reranker 加载完成")
        
    def rerank(self, query: str, documents: List[Dict], k: int = 10) -> List[Dict]:
        """
        使用 BGE Reranker 对文档进行精排
        
        Args:
            query: 用户查询
            documents: 文档列表
            k: 返回数量
            
        Returns:
            重排序后的 top-k 文档
        """
        import torch
        
        if not documents:
            return []
            
        self._ensure_loaded()
        
        # 构建 Query-Document 对
        # 使用 filename + 相关字段作为文档内容
        pairs = []
        for doc in documents:
            meta = doc.get('metadata', {})
            # 构建文档表示
            doc_text = self._build_doc_text(meta)
            pairs.append((query, doc_text))
        
        # 批量计算分数（优化：减少 max_length，启用编译优化）
        with torch.no_grad():
            inputs = self._tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=128  # 从512降到128，文档名通常很短
            )
            inputs = {key: val.to(self._device) for key, val in inputs.items()}
            
            # 推理
            outputs = self._model(**inputs, return_dict=True)
            scores = outputs.logits.view(-1).float().cpu().numpy()
        
        # 按分数排序
        sorted_indices = np.argsort(scores)[::-1][:k]
        
        # 返回重排序后的文档，附带 rerank_score
        reranked = []
        for idx in sorted_indices:
            doc = documents[idx].copy()
            doc['rerank_score'] = float(scores[idx])
            reranked.append(doc)
            
        return reranked
    
    def _build_doc_text(self, metadata: Dict) -> str:
        """从 metadata 构建文档文本表示"""
        parts = []
        
        # 按重要性排列字段
        if metadata.get('filename'):
            parts.append(metadata['filename'])
        if metadata.get('brand'):
            parts.append(f"品牌:{metadata['brand']}")
        if metadata.get('series'):
            parts.append(f"系列:{metadata['series']}")
        if metadata.get('doc_type'):
            parts.append(f"类型:{metadata['doc_type']}")
        if metadata.get('diagram_subtype'):
            parts.append(f"子类型:{metadata['diagram_subtype']}")
            
        return ' '.join(parts)


class ZhipuReranker(BaseReranker):
    """
    智谱 Reranker - 使用智谱 AI API
    
    轻量级选择，无需本地模型
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化智谱 Reranker
        
        Args:
            api_key: 智谱 API Key，不传则从环境变量读取
        """
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self._client = None
        
    def _ensure_client(self):
        """延迟初始化客户端"""
        if self._client is None:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=self.api_key)
            
    def rerank(self, query: str, documents: List[Dict], k: int = 10) -> List[Dict]:
        """
        使用智谱 API 进行重排序
        
        注意：智谱目前可能不提供 rerank API，此处为预留接口
        实际可用性需要确认
        """
        # 智谱目前可能没有直接的 rerank API
        # 可以使用 embedding 相似度作为 fallback
        # 或者等待智谱开放 rerank 接口
        
        # Fallback: 不做重排序，直接返回
        print("⚠️ 智谱 Reranker 暂未实现，返回原始结果")
        return documents[:k]


class SimpleReranker(BaseReranker):
    """
    简单 Reranker - 基于关键词匹配的轻量级重排序
    
    适用于：
    - 不想加载大模型
    - 快速验证
    - 作为 fallback
    """
    
    def __init__(self):
        pass
        
    def rerank(self, query: str, documents: List[Dict], k: int = 10) -> List[Dict]:
        """
        基于关键词匹配的简单重排序
        
        策略：
        1. 完全匹配 query 中的词 → 高分
        2. 连续子串匹配 → 加分
        3. 保持原有相似度作为 baseline
        """
        if not documents:
            return []
        
        # 提取查询关键词（2-4字的片段）
        query_terms = self._extract_terms(query)
        
        scored_docs = []
        for doc in documents:
            meta = doc.get('metadata', {})
            filename = meta.get('filename', '')
            
            # 计算关键词匹配分数
            match_score = self._calc_match_score(query_terms, filename, query)
            
            # 原始相似度
            original_score = doc.get('similarity', doc.get('score', 0))
            
            # 综合分数: 60% 原始相似度 + 40% 关键词匹配
            combined_score = 0.6 * original_score + 0.4 * match_score
            
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = combined_score
            scored_docs.append((combined_score, doc_copy))
        
        # 排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:k]]
    
    def _extract_terms(self, text: str) -> List[str]:
        """提取关键词（2-4字片段）"""
        terms = []
        # 按中文分词习惯，提取连续的2-4字片段
        for length in [4, 3, 2]:
            for i in range(len(text) - length + 1):
                term = text[i:i+length]
                if term not in terms:
                    terms.append(term)
        return terms
    
    def _calc_match_score(self, terms: List[str], filename: str, query: str) -> float:
        """计算匹配分数"""
        if not filename:
            return 0.0
        
        score = 0.0
        
        # 1. 完整查询在文件名中的匹配
        if query in filename:
            score += 0.5
        
        # 2. 关键词匹配
        matched_terms = [t for t in terms if t in filename]
        if terms:
            term_ratio = len(matched_terms) / len(terms)
            score += 0.3 * term_ratio
        
        # 3. 字符级别重叠
        query_chars = set(query)
        filename_chars = set(filename)
        if query_chars:
            char_overlap = len(query_chars & filename_chars) / len(query_chars)
            score += 0.2 * char_overlap
        
        return min(score, 1.0)  # 归一化到 [0, 1]


def create_reranker(
    reranker_type: str = 'simple',
    model_path: str = 'BAAI/bge-reranker-base'
) -> BaseReranker:
    """
    创建 Reranker 实例
    
    Args:
        reranker_type: 类型 - 'bge', 'zhipu', 'simple'
        model_path: BGE 模型路径
        
    Returns:
        Reranker 实例
    """
    if reranker_type == 'bge':
        return BgeReranker(model_path)
    elif reranker_type == 'zhipu':
        return ZhipuReranker()
    elif reranker_type == 'simple':
        return SimpleReranker()
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


class BM25Prefilter:
    """
    BM25 粗筛器 - 在 Reranker 前快速筛选候选文档
    
    当文档数量大时（如 > 50），先用 BM25 快速筛选出候选，
    再用 Cross-Encoder 精排，可大幅提升性能。
    
    典型流程：
        100+ 文档 → BM25 粗筛 (Top-30) → Reranker 精排 (Top-5)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 参数
        
        Args:
            k1: 词频饱和参数 (1.2-2.0)
            b: 文档长度归一化参数 (0-1)
        """
        self.k1 = k1
        self.b = b
        self._tokenizer = None
    
    def _get_tokenizer(self):
        """获取分词器（延迟加载）"""
        if self._tokenizer is None:
            try:
                import jieba
                self._tokenizer = jieba.lcut
                logger.debug("BM25 使用 jieba 分词")
            except ImportError:
                # Fallback: 字符级分词
                self._tokenizer = lambda text: list(text)
                logger.debug("BM25 使用字符分词 (jieba 未安装)")
        return self._tokenizer
    
    def prefilter(
        self, 
        query: str, 
        documents: List[Dict], 
        top_n: int = 30,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        使用 BM25 对文档进行粗筛
        
        Args:
            query: 用户查询
            documents: 文档列表
            top_n: 返回前 N 个候选
            min_score: 最低分数阈值
            
        Returns:
            粗筛后的文档列表（带 bm25_score）
        """
        if not documents:
            return []
        
        if len(documents) <= top_n:
            # 文档数量少于阈值，不需要粗筛
            return documents
        
        tokenize = self._get_tokenizer()
        
        # 分词
        query_terms = tokenize(query)
        doc_terms_list = []
        for doc in documents:
            meta = doc.get('metadata', {})
            text = self._build_doc_text(meta)
            doc_terms_list.append(tokenize(text))
        
        # 计算 IDF
        idf = self._compute_idf(query_terms, doc_terms_list)
        
        # 计算平均文档长度
        avg_dl = sum(len(terms) for terms in doc_terms_list) / len(doc_terms_list)
        
        # 计算 BM25 分数
        scored_docs = []
        for i, (doc, doc_terms) in enumerate(zip(documents, doc_terms_list)):
            score = self._compute_bm25(query_terms, doc_terms, idf, avg_dl)
            if score >= min_score:
                doc_copy = doc.copy()
                doc_copy['bm25_score'] = score
                scored_docs.append((score, doc_copy))
        
        # 排序
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:top_n]]
    
    def _build_doc_text(self, metadata: Dict) -> str:
        """从 metadata 构建文档文本"""
        parts = []
        for key in ['filename', 'brand', 'series', 'doc_type', 'diagram_subtype', 'path']:
            if metadata.get(key):
                parts.append(str(metadata[key]))
        return ' '.join(parts)
    
    def _compute_idf(self, query_terms: List[str], doc_terms_list: List[List[str]]) -> Dict[str, float]:
        """计算 IDF 值"""
        import math
        
        N = len(doc_terms_list)
        idf = {}
        
        for term in set(query_terms):
            # 包含该词的文档数
            df = sum(1 for doc_terms in doc_terms_list if term in doc_terms)
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        return idf
    
    def _compute_bm25(
        self, 
        query_terms: List[str], 
        doc_terms: List[str], 
        idf: Dict[str, float],
        avg_dl: float
    ) -> float:
        """计算单个文档的 BM25 分数"""
        score = 0.0
        dl = len(doc_terms)
        
        # 计算词频
        tf = {}
        for term in doc_terms:
            tf[term] = tf.get(term, 0) + 1
        
        for term in query_terms:
            if term in tf:
                # BM25 公式
                term_tf = tf[term]
                numerator = idf.get(term, 0) * term_tf * (self.k1 + 1)
                denominator = term_tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score += numerator / denominator
        
        return score


# 全局 BM25 实例
_bm25_prefilter: Optional[BM25Prefilter] = None


def get_bm25_prefilter() -> BM25Prefilter:
    """获取全局 BM25 粗筛器实例"""
    global _bm25_prefilter
    if _bm25_prefilter is None:
        _bm25_prefilter = BM25Prefilter()
    return _bm25_prefilter
