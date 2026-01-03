#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查询缓存模块 - 缓存 LLM 查询改写结果

支持:
1. LRU 缓存策略
2. TTL 过期机制
3. 持久化到磁盘（可选）
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Dict[str, Any]
    created_at: float
    hits: int = 0


class QueryCache:
    """
    查询缓存 - LRU + TTL
    
    用于缓存 LLM 查询改写结果，避免重复调用 API
    """
    
    def __init__(
        self, 
        max_size: int = 500,
        ttl: int = 3600,
        persist_path: Optional[Path] = None,
    ):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒），0 表示永不过期
            persist_path: 持久化文件路径（可选）
        """
        self.max_size = max_size
        self.ttl = ttl
        self.persist_path = persist_path
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0}
        
        # 从磁盘加载
        if persist_path:
            self._load()
    
    def _make_key(self, query: str, context: str = "") -> str:
        """生成缓存键"""
        content = f"{query}|{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        获取缓存
        
        Args:
            query: 查询字符串
            context: 上下文（如上一次查询）
        
        Returns:
            缓存的值，如果不存在或已过期则返回 None
        """
        key = self._make_key(query, context)
        
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # 检查 TTL
            if self.ttl > 0 and time.time() - entry.created_at > self.ttl:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # 更新 LRU 顺序
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats["hits"] += 1
            
            return entry.value
    
    def set(self, query: str, value: Dict[str, Any], context: str = ""):
        """
        设置缓存
        
        Args:
            query: 查询字符串
            value: 要缓存的值
            context: 上下文
        """
        key = self._make_key(query, context)
        
        with self._lock:
            # 如果已存在，更新并移到末尾
            if key in self._cache:
                self._cache[key].value = value
                self._cache[key].created_at = time.time()
                self._cache.move_to_end(key)
                return
            
            # 检查容量
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # 删除最旧的
            
            # 添加新条目
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
            )
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0}
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
        }
    
    def _save(self):
        """持久化到磁盘"""
        if not self.persist_path:
            return
        
        try:
            # 确保目录存在
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                key: {
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "hits": entry.hits,
                }
                for key, entry in self._cache.items()
            }
            # 使用 Path.write_text 避免 open 在某些上下文中不可用
            import json as json_module
            self.persist_path.write_text(
                json_module.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            pass  # 静默失败，不影响主流程
    
    def _load(self):
        """从磁盘加载"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            import json as json_module
            data = json_module.loads(self.persist_path.read_text(encoding='utf-8'))
            
            for key, entry_data in data.items():
                # 跳过过期的
                if self.ttl > 0 and time.time() - entry_data["created_at"] > self.ttl:
                    continue
                
                self._cache[key] = CacheEntry(
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    hits=entry_data.get("hits", 0),
                )
            
            if self._cache:
                import logging
                logging.getLogger(__name__).info(f"  └─ 已加载查询缓存: {len(self._cache)} 条")
        except Exception:
            pass  # 静默失败
    
    def __del__(self):
        """析构时保存"""
        try:
            self._save()
        except Exception:
            pass  # 静默失败，避免程序退出时的错误


# 全局缓存实例
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """获取全局查询缓存实例"""
    global _query_cache
    if _query_cache is None:
        from config.settings import config
        _query_cache = QueryCache(
            max_size=config.QUERY_CACHE_SIZE,
            ttl=config.QUERY_CACHE_TTL,
            persist_path=config.CACHE_DIR / "query_cache.json" if config.ENABLE_QUERY_CACHE else None,
        )
    return _query_cache
