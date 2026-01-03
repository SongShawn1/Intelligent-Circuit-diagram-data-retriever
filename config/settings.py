#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块 - 集中管理所有配置项

使用方式:
    from config.settings import config
    print(config.RERANK_TOP_K)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AppConfig:
    """应用配置"""
    
    # ============ 路径配置 ============
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CONFIG_DIR: Path = PROJECT_ROOT / "config"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    CACHE_DIR: Path = PROJECT_ROOT / "cache"
    
    # 数据文件
    STRUCTURED_DATA_PATH: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "structured_data_llm.json")
    SYNONYMS_PATH: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "synonyms.json")
    
    # ============ 搜索配置 ============
    RERANK_TOP_K: int = 5                    # Reranker 返回数量
    MAX_DIRECT_RESULTS: int = 5              # 直接返回结果的最大数量
    MAX_DISPLAY_OPTIONS: int = 10            # 单次显示的最大选项数
    MAX_OPTIONS_PER_LEVEL: int = 15          # 每层最大选项数（超过则分组）
    NEED_FILTER_THRESHOLD: int = 20          # 需要筛选的文件数阈值
    
    # ============ Reranker 配置 ============
    RERANKER_TYPE: str = "bge"               # Reranker 类型: bge, cross_encoder, simple
    BGE_MODEL_NAME: str = "BAAI/bge-reranker-base"
    
    # ============ LLM 配置 ============
    USE_LLM_REWRITER: bool = True            # 是否启用 LLM Query Rewriting
    LLM_PROVIDER: str = "zhipu"              # LLM 提供商: zhipu, openai
    ZHIPU_API_KEY: str = field(default_factory=lambda: os.getenv("ZHIPU_API_KEY", ""))
    ZHIPU_MODEL: str = "glm-4-flash"
    
    # ============ 缓存配置 ============
    ENABLE_QUERY_CACHE: bool = True          # 是否启用查询缓存
    QUERY_CACHE_SIZE: int = 500              # 查询缓存大小
    QUERY_CACHE_TTL: int = 3600              # 缓存过期时间（秒）
    
    # ============ 日志配置 ============
    ENABLE_ACCESS_LOG: bool = True           # 是否启用访问日志
    LOG_LEVEL: str = "INFO"                  # 日志级别
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
    ACCESS_LOG_FILE: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "access.log")
    
    # ============ UI 配置 ============
    SERVER_PORT: int = 7861
    SHARE: bool = False
    DEBUG: bool = False
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        self.LOG_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """从环境变量加载配置"""
        return cls(
            RERANK_TOP_K=int(os.getenv("RERANK_TOP_K", 5)),
            MAX_DIRECT_RESULTS=int(os.getenv("MAX_DIRECT_RESULTS", 5)),
            USE_LLM_REWRITER=os.getenv("USE_LLM_REWRITER", "true").lower() == "true",
            ENABLE_QUERY_CACHE=os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true",
            ENABLE_ACCESS_LOG=os.getenv("ENABLE_ACCESS_LOG", "true").lower() == "true",
            DEBUG=os.getenv("DEBUG", "false").lower() == "true",
            SERVER_PORT=int(os.getenv("SERVER_PORT", 7861)),
        )


# 全局配置实例
config = AppConfig()
