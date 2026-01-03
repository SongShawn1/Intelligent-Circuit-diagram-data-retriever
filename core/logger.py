#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
访问日志模块 - 记录用户查询和系统响应

用途:
1. 分析用户搜索习惯
2. 优化搜索算法
3. 排查问题
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler


@dataclass
class AccessLogEntry:
    """访问日志条目"""
    timestamp: str
    session_id: str
    query: str
    rewritten_query: Optional[str]
    result_count: int
    state: str
    response_time_ms: float
    used_llm: bool
    cache_hit: bool
    extra: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AccessLogger:
    """
    访问日志记录器
    
    支持:
    1. 文件日志（带轮转）
    2. JSON 格式便于分析
    3. 按天分割
    """
    
    def __init__(
        self,
        log_dir: Path,
        log_file: str = "access.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if not enabled:
            return
        
        # 配置日志器
        self.logger = logging.getLogger("access")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 不传递给根日志器
        
        # 清除已有处理器
        self.logger.handlers.clear()
        
        # 文件处理器（带轮转）
        file_handler = RotatingFileHandler(
            self.log_dir / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        
        # 统计
        self._stats = {
            "total_queries": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "no_result_queries": 0,
        }
    
    def log(
        self,
        session_id: str,
        query: str,
        rewritten_query: Optional[str],
        result_count: int,
        state: str,
        response_time_ms: float,
        used_llm: bool = False,
        cache_hit: bool = False,
        **extra,
    ):
        """记录一条访问日志"""
        if not self.enabled:
            return
        
        entry = AccessLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            query=query,
            rewritten_query=rewritten_query if rewritten_query != query else None,
            result_count=result_count,
            state=state,
            response_time_ms=round(response_time_ms, 2),
            used_llm=used_llm,
            cache_hit=cache_hit,
            extra=extra if extra else None,
        )
        
        # 写入日志
        self.logger.info(entry.to_json())
        
        # 更新统计
        self._stats["total_queries"] += 1
        if used_llm:
            self._stats["llm_calls"] += 1
        if cache_hit:
            self._stats["cache_hits"] += 1
        if result_count == 0:
            self._stats["no_result_queries"] += 1
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["total_queries"]
        return {
            **self._stats,
            "cache_hit_rate": f"{self._stats['cache_hits'] / total:.1%}" if total > 0 else "N/A",
            "no_result_rate": f"{self._stats['no_result_queries'] / total:.1%}" if total > 0 else "N/A",
        }
    
    def get_recent_queries(self, n: int = 100) -> List[Dict]:
        """获取最近 n 条查询（从日志文件读取）"""
        log_file = self.log_dir / "access.log"
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            entries = []
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except:
                    continue
            return entries
        except Exception as e:
            print(f"读取日志失败: {e}")
            return []
    
    def get_popular_queries(self, n: int = 20) -> List[tuple]:
        """获取热门查询"""
        from collections import Counter
        entries = self.get_recent_queries(1000)
        queries = [e.get("query", "") for e in entries]
        return Counter(queries).most_common(n)


# 全局日志实例
_access_logger: Optional[AccessLogger] = None


def get_access_logger() -> AccessLogger:
    """获取全局访问日志实例"""
    global _access_logger
    if _access_logger is None:
        from config.settings import config
        _access_logger = AccessLogger(
            log_dir=config.LOG_DIR,
            enabled=config.ENABLE_ACCESS_LOG,
        )
    return _access_logger
