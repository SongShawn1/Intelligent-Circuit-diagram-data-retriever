#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版智能导航 Chatbot - 基于导航树的多轮对话

核心理念：
1. 用户输入 → LLM 预处理（纠错、扩展）→ 导航树搜索 → 找到匹配节点
2. 多个匹配 → 让用户选择
3. 唯一匹配 → 如果文件少直接返回，否则继续导航或 Rerank
4. 支持返回上一级

与原 chatbot.py 的区别：
- 不再使用复杂的 DecisionEngine 判断用哪个字段分面
- 直接利用层级路径的自然结构进行导航
- 新增 LLM Query Rewriting 功能（纠错、同义词扩展）
- 代码量减少 60%，逻辑更清晰
"""

import sys
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入配置
from config.settings import config

from core.navigation_tree import NavigationTree, NavigationResult
from core.reranker import create_reranker, BaseReranker, get_bm25_prefilter
from core.query_rewriter import LLMQueryRewriter, QueryRewriteResult
from core.cache import get_query_cache
from core.logger import get_access_logger


class ConversationState(Enum):
    """对话状态"""
    IDLE = auto()
    AWAITING_SELECTION = auto()
    AWAITING_FILTER = auto()       # 等待用户筛选（叶子节点文件较多）
    COMPLETED = auto()


@dataclass
class ConversationTurn:
    """对话轮次"""
    role: str           # 'user' 或 'assistant'
    content: str        # 消息内容
    query: Optional[str] = None           # 用户的原始/改写后的查询
    rewritten_query: Optional[str] = None # 改写后的查询（如果有）


@dataclass
class NavContext:
    """导航上下文"""
    current_path: str = ""                              # 当前导航路径
    pending_options: List[tuple] = field(default_factory=list)  # 待选择的选项 [(path, name, count), ...]
    pending_files: List[Dict] = field(default_factory=list)     # 待筛选的文件（用于 AWAITING_FILTER 状态）
    pending_filter_options: List[tuple] = field(default_factory=list)  # 待筛选的关键词选项 [(keyword, count), ...]
    state: ConversationState = ConversationState.IDLE
    history: List[Dict] = field(default_factory=list)   # 导航历史，每项包含 {path, options, state}
    original_query: str = ""                            # 原始查询
    
    # 记忆窗口：对话历史
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    last_search_query: str = ""                         # 上一次的检索查询
    
    def add_user_turn(self, content: str, query: str = None, rewritten_query: str = None):
        """添加用户对话轮次"""
        self.conversation_history.append(ConversationTurn(
            role='user',
            content=content,
            query=query or content,
            rewritten_query=rewritten_query,
        ))
        # 限制历史长度（最近 10 轮）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def add_assistant_turn(self, content: str):
        """添加助手对话轮次"""
        self.conversation_history.append(ConversationTurn(
            role='assistant',
            content=content[:500],  # 限制长度
        ))
    
    def get_recent_history(self, n: int = 6) -> List[ConversationTurn]:
        """获取最近 n 条对话"""
        return self.conversation_history[-n:]
    
    def can_go_back(self) -> bool:
        """是否可以返回上一级"""
        return len(self.history) > 0
    
    def reset(self, keep_history: bool = True):
        """重置上下文（可选保留对话历史）"""
        self.current_path = ""
        self.pending_options = []
        self.pending_files = []
        self.pending_filter_options = []
        self.state = ConversationState.IDLE
        self.history = []
        self.original_query = ""
        if not keep_history:
            self.conversation_history = []
            self.last_search_query = ""


@dataclass
class ChatResponse:
    """对话响应"""
    message: str
    results: List[Dict] = field(default_factory=list)   # 最终返回的文件
    options: List[tuple] = field(default_factory=list)  # 选项 [(path, name, count), ...]
    filter_options: List[tuple] = field(default_factory=list)  # 筛选选项 [(keyword, count), ...]
    state: ConversationState = ConversationState.IDLE
    can_go_back: bool = False


class NavigationChatbot:
    """
    基于导航树的 Chatbot
    
    核心流程：
    1. 用户输入查询 → LLM 预处理（纠错、扩展）→ 导航树搜索匹配节点
    2. 根据匹配结果决定下一步：
       - 无匹配 → 提示没有结果
       - 单个匹配 → 导航到该节点（返回子节点或文件）
       - 多个匹配 → 让用户选择
    3. 如果节点下文件太多 → 使用 Reranker 选 Top 5
    """
    
    # BM25 粗筛阈值：文档超过此数量时启用 BM25 预筛选
    BM25_THRESHOLD = 50
    BM25_PREFILTER_TOP_N = 30
    
    def __init__(self, debug: bool = False, use_llm_rewriter: bool = None):
        self.debug = debug
        self.use_llm_rewriter = use_llm_rewriter if use_llm_rewriter is not None else config.USE_LLM_REWRITER
        
        # 获取缓存和日志记录器
        self.cache = get_query_cache()
        self.logger = get_access_logger()
        self.session_id = str(uuid.uuid4())[:8]  # 会话 ID
        
        logger.info("🤖 初始化导航 Chatbot...")
        
        # 加载导航树
        self.tree = NavigationTree(str(config.STRUCTURED_DATA_PATH))
        
        # Reranker（带 fallback）
        try:
            self.reranker = create_reranker(config.RERANKER_TYPE)
            logger.info(f"  └─ Reranker: {config.RERANKER_TYPE}")
        except Exception as e:
            logger.warning(f"  └─ Reranker 加载失败: {e}，使用 SimpleReranker")
            self.reranker = create_reranker('simple')
        
        # BM25 粗筛器
        self.bm25_prefilter = get_bm25_prefilter()
        
        # LLM Query Rewriter（带 fallback）
        self.query_rewriter = None
        if self.use_llm_rewriter:
            try:
                self.query_rewriter = LLMQueryRewriter(use_llm=True, debug=debug)
                logger.info("  └─ LLM Query Rewriter: 已启用")
            except Exception as e:
                logger.warning(f"  └─ LLM Query Rewriter: 初始化失败 ({e})，将使用无 LLM 模式")
                try:
                    self.query_rewriter = LLMQueryRewriter(use_llm=False, debug=debug)
                except:
                    self.query_rewriter = None
        else:
            logger.info("  └─ LLM Query Rewriter: 未启用")
        
        logger.info("✅ 导航 Chatbot 初始化完成")
    
    def chat(self, user_input: str, context: NavContext) -> ChatResponse:
        """
        处理用户输入
        
        输入类型：
        1. __SELECT__:X - 选择第 X 个选项（UI 命令）
        2. __BACK__ - 返回上一级（UI 命令）
        3. 自然语言 - 新查询、追问、选择等（由 LLM 判断意图）
        """
        user_input = user_input.strip()
        
        if not user_input:
            return ChatResponse(message="请输入您要查找的资料。")
        
        # 处理 UI 命令：返回
        if user_input == "__BACK__":
            return self._handle_back(context)
        
        # 处理 UI 命令：选择
        if user_input.startswith("__SELECT__:"):
            option_idx = user_input.replace("__SELECT__:", "").strip()
            return self._handle_selection(option_idx, context)
        
        # ========== 处理筛选状态 ==========
        if context.state == ConversationState.AWAITING_FILTER:
            return self._handle_filter_input(user_input, context)
        
        # ========== 记忆窗口：上下文感知改写 ==========
        from core.query_rewriter import IntentType, QueryRewriteResult
        from dataclasses import asdict
        
        rewrite_result = None
        search_query = user_input
        search_message_prefix = ""
        start_time = time.time()  # 记录开始时间
        cached_result = None
        
        if self.query_rewriter:
            # 检查缓存
            cached_dict = self.cache.get(user_input)
            if cached_dict:
                # 从字典重建 QueryRewriteResult
                cached_dict['intent'] = IntentType[cached_dict.get('intent', 'NEW_SEARCH')]
                rewrite_result = QueryRewriteResult(**cached_dict)
                cached_result = rewrite_result  # 标记缓存命中
                if self.debug:
                    logger.debug(f"[Cache] 命中缓存: '{user_input}'")
            else:
                # 构建对话历史
                history = [
                    {"role": t.role, "content": t.content}
                    for t in context.get_recent_history(6)
                ]
                
                # 上下文感知改写
                rewrite_result = self.query_rewriter.rewrite_with_context(
                    query=user_input,
                    last_query=context.last_search_query,
                    conversation_history=history,
                    pending_options=context.pending_options,
                )
                
                # 存入缓存（只缓存 NEW_SEARCH 和 FOLLOW_UP 类型）
                if rewrite_result.intent in (IntentType.NEW_SEARCH, IntentType.FOLLOW_UP):
                    # 转换为字典存储
                    cache_dict = asdict(rewrite_result)
                    cache_dict['intent'] = rewrite_result.intent.name
                    self.cache.set(user_input, cache_dict)
            
            if self.debug:
                logger.debug(f"[Intent] {rewrite_result.intent.name}")
                logger.debug(f"[Rewrite] '{user_input}' → '{rewrite_result.corrected_query}'")
            
            # 根据意图处理
            if rewrite_result.intent == IntentType.SELECT and rewrite_result.select_index is not None:
                # 自然语言选择（如"第一个"、"A"）
                return self._handle_selection(str(rewrite_result.select_index), context, is_index=True)
            
            if rewrite_result.intent == IntentType.BACK:
                return self._handle_back(context)
            
            if rewrite_result.intent == IntentType.CONTINUE:
                # 继续：TODO 可以实现分页
                return ChatResponse(
                    message="📋 当前显示的是所有匹配结果。您可以选择一个选项，或输入新的查询。",
                    options=context.pending_options,
                    state=context.state,
                    can_go_back=context.can_go_back(),
                )
            
            # 新查询或追问：使用改写后的查询
            search_query = rewrite_result.corrected_query
            
            if rewrite_result.corrected_query != user_input:
                if rewrite_result.is_followup:
                    search_message_prefix = f"*理解为：{rewrite_result.corrected_query}*\n\n"
                else:
                    search_message_prefix = f"*已自动纠正：{user_input} → {rewrite_result.corrected_query}*\n\n"
        
        # 如果是新查询，重置导航状态（但保留对话历史）
        if not rewrite_result or not rewrite_result.is_followup:
            context.reset(keep_history=True)
        
        context.original_query = user_input
        context.last_search_query = search_query
        
        # 记录用户对话
        context.add_user_turn(
            content=user_input,
            query=user_input,
            rewritten_query=search_query if search_query != user_input else None,
        )
        
        # 使用导航树搜索
        result = self.tree.navigate(search_query)
        
        # 如果没找到，尝试用扩展关键词
        if result.status == 'no_match' and rewrite_result and rewrite_result.expanded_keywords:
            for exp_kw in rewrite_result.expanded_keywords:
                alt_query = f"{search_query} {exp_kw}"
                alt_result = self.tree.navigate(alt_query)
                if alt_result.status != 'no_match':
                    result = alt_result
                    if self.debug:
                        logger.debug(f"[扩展搜索] 使用 '{alt_query}' 找到结果")
                    break
        
        response = self._process_nav_result(result, search_query, context)
        
        # 添加提示信息
        if search_message_prefix and response.message:
            response.message = search_message_prefix + response.message
        
        # 记录助手回复
        context.add_assistant_turn(response.message)
        
        # 记录访问日志
        elapsed_time = time.time() - start_time
        cache_hit = cached_result is not None if self.query_rewriter else False
        self.logger.log(
            query=user_input,
            rewritten_query=search_query if search_query != user_input else None,
            result_count=len(response.results),
            response_time_ms=elapsed_time * 1000,
            session_id=self.session_id,
            state=response.state.name,
            used_llm=self.query_rewriter is not None and not cache_hit,
            cache_hit=cache_hit,
        )
        
        return response
    
    def _handle_back(self, context: NavContext) -> ChatResponse:
        """处理返回上一级"""
        if not context.history:
            context.reset()
            return ChatResponse(
                message="已返回初始状态，请输入新的查询。",
                state=ConversationState.IDLE,
            )
        
        # 恢复上一个状态
        prev_state = context.history.pop()
        context.current_path = prev_state.get('path', '')
        context.pending_options = prev_state.get('options', [])
        context.state = ConversationState(prev_state.get('state', ConversationState.AWAITING_SELECTION.value))
        
        # 构建响应消息
        if context.pending_options:
            message = "请选择：\n\n"
            for i, (path, name, count) in enumerate(context.pending_options[:config.MAX_DISPLAY_OPTIONS]):
                letter = chr(ord('A') + i)
                message += f"**{letter}.** {name}（{count}个结果）\n"
            message += "\n*请输入选项字母（如 A）或直接描述您的需求*"
            
            return ChatResponse(
                message=message,
                options=context.pending_options,
                state=context.state,
                can_go_back=len(context.history) > 0,
            )
        
        # 如果没有选项，返回初始状态
        context.reset()
        return ChatResponse(
            message="已返回初始状态，请输入新的查询。",
            state=ConversationState.IDLE,
        )
    
    def _handle_selection(self, option_str: str, context: NavContext, is_index: bool = False) -> ChatResponse:
        """
        处理用户选择
        
        Args:
            option_str: 选项字符串（A/B/C 或 1/2/3）
            context: 导航上下文
            is_index: 如果为 True，option_str 是 0-based 索引
        """
        if context.state != ConversationState.AWAITING_SELECTION:
            return ChatResponse(message="当前没有可选择的选项，请输入新的查询。")
        
        # 解析选项索引
        try:
            if is_index:
                idx = int(option_str)  # 已经是 0-based 索引
            elif option_str.isalpha():
                idx = ord(option_str.upper()) - ord('A')
            else:
                idx = int(option_str) - 1  # 1-based 转 0-based
        except ValueError:
            return ChatResponse(message=f"无效的选项: {option_str}")
        
        if idx < 0 or idx >= len(context.pending_options):
            return ChatResponse(message=f"选项超出范围，请选择 1-{len(context.pending_options)}。")
        
        # 获取选中的路径
        selected_path, selected_name, _ = context.pending_options[idx]
        
        # 保存当前状态到历史（在选择之前）
        # 这样返回时可以恢复到选择前的状态
        if context.current_path or context.pending_options:
            context.history.append({
                'path': context.current_path,
                'options': context.pending_options,
                'state': context.state.value,
            })
        
        # 更新当前路径
        context.current_path = selected_path
        
        # 导航到选中的节点
        node = self.tree.get_node(selected_path)
        if node is None:
            return ChatResponse(message=f"找不到路径: {selected_path}")
        
        result = self.tree._node_to_result(node)
        return self._process_nav_result(result, context.original_query, context, is_selection=True)
    
    def _process_nav_result(
        self, 
        result: NavigationResult, 
        query: str, 
        context: NavContext,
        is_selection: bool = False
    ) -> ChatResponse:
        """处理导航结果"""
        
        if result.status == 'no_match':
            return ChatResponse(
                message=result.message,
                state=ConversationState.IDLE,
            )
        
        if result.status == 'files':
            # 直接返回文件（数量少）
            context.state = ConversationState.COMPLETED
            return ChatResponse(
                message=result.message,
                results=result.files,
                state=ConversationState.COMPLETED,
                can_go_back=context.can_go_back(),
            )
        
        if result.status == 'need_rerank':
            # 文件太多，使用 Reranker
            return self._rerank_and_return(result.files, query, context)
        
        if result.status == 'need_filter':
            # 文件数量适中，提取关键词作为筛选选项
            context.pending_files = result.files
            context.state = ConversationState.AWAITING_FILTER
            
            # 更新当前路径（如果不是从选择操作来的）
            if not is_selection and result.current_path:
                context.current_path = result.current_path
            
            # 提取筛选关键词选项
            filter_options = self._extract_filter_keywords(result.files, query)
            context.pending_filter_options = filter_options  # 保存供后续使用
            
            message = result.message
            message += "\n\n请选择筛选关键词（同一文件可能匹配多个关键词）："
            
            return ChatResponse(
                message=message,
                filter_options=filter_options,
                state=ConversationState.AWAITING_FILTER,
                can_go_back=len(context.history) > 0,
            )
        
        if result.status in ('navigate', 'multiple_matches'):
            # 需要用户选择
            context.pending_options = result.options
            context.state = ConversationState.AWAITING_SELECTION
            
            # 更新当前路径（如果不是从选择操作来的）
            if not is_selection and result.current_path:
                context.current_path = result.current_path
            
            # 格式化选项消息
            message = result.message + "\n\n"
            for i, (path, name, count) in enumerate(result.options[:config.MAX_DISPLAY_OPTIONS]):
                letter = chr(ord('A') + i)
                message += f"**{letter}.** {name}（{count}个结果）\n"
            
            message += "\n*请输入选项字母（如 A）或直接描述您的需求*"
            
            return ChatResponse(
                message=message,
                options=result.options,
                state=ConversationState.AWAITING_SELECTION,
                can_go_back=len(context.history) > 0,
            )
        
        return ChatResponse(message="未知的导航状态")
    
    def _handle_filter_input(self, user_input: str, context: NavContext) -> ChatResponse:
        """
        处理筛选状态下的用户输入
        
        用户可以：
        1. 输入 "全部" 或 "查看全部" → 使用 Reranker 返回 Top K
        2. 输入关键词 → 在 pending_files 中筛选
        """
        user_input_lower = user_input.strip().lower()
        
        # 用户想直接查看结果
        if user_input_lower in ('全部', '查看全部', '所有', 'all', '直接看'):
            return self._rerank_and_return(
                context.pending_files, 
                context.original_query, 
                context
            )
        
        # 用户输入关键词进行筛选
        keyword = user_input.strip()
        filtered = []
        
        for f in context.pending_files:
            # 在 filename 和 page_content 中搜索关键词
            filename = f.get('filename', '')
            content = f.get('page_content', '')
            if keyword in filename or keyword in content:
                filtered.append(f)
        
        if not filtered:
            # 没有匹配，提示用户
            # 重新生成筛选选项
            filter_options = self._extract_filter_keywords(context.pending_files, context.original_query)
            return ChatResponse(
                message=f"未找到包含 **{keyword}** 的文件。\n\n请选择其他筛选条件：",
                filter_options=filter_options,
                state=ConversationState.AWAITING_FILTER,
                can_go_back=context.can_go_back(),
            )
        
        if len(filtered) <= 5:
            # 筛选后数量少，直接返回
            context.state = ConversationState.COMPLETED
            return ChatResponse(
                message=f"根据关键词 **{keyword}** 筛选后，找到 **{len(filtered)}** 个结果：",
                results=filtered,
                state=ConversationState.COMPLETED,
                can_go_back=context.can_go_back(),
            )
        
        # 筛选后仍然较多（>5），继续提供筛选选项
        context.pending_files = filtered  # 更新待筛选文件
        combined_query = f"{context.original_query} {keyword}"
        filter_options = self._extract_filter_keywords(filtered, combined_query)
        context.pending_filter_options = filter_options
        
        return ChatResponse(
            message=f"根据 **{keyword}** 筛选后，还有 {len(filtered)} 个文件，请继续选择：",
            filter_options=filter_options,
            state=ConversationState.AWAITING_FILTER,
            can_go_back=context.can_go_back(),
        )
    
    def _rerank_and_return(
        self, 
        files: List[Dict], 
        query: str, 
        context: NavContext
    ) -> ChatResponse:
        """使用 Reranker 选出 Top K（大量文档时先用 BM25 粗筛）"""
        t_start = time.time()
        rerank_query = query if query else context.original_query
        
        # 转换为 reranker 需要的格式
        docs = [{'metadata': f} for f in files]
        
        # 如果文档数量大，先用 BM25 粗筛
        if len(docs) > self.BM25_THRESHOLD:
            logger.debug(f"🔍 BM25 粗筛: {len(docs)} → Top {self.BM25_PREFILTER_TOP_N}")
            try:
                docs = self.bm25_prefilter.prefilter(
                    rerank_query, 
                    docs, 
                    top_n=self.BM25_PREFILTER_TOP_N
                )
            except Exception as e:
                logger.warning(f"BM25 粗筛失败: {e}，跳过粗筛")
        
        # 使用 Reranker 精排
        logger.debug(f"🔄 Rerank: {len(docs)} 个文件 → Top {config.RERANK_TOP_K}")
        
        try:
            reranked = self.reranker.rerank(rerank_query, docs, k=config.RERANK_TOP_K)
        except Exception as e:
            logger.error(f"Reranker 失败: {e}，返回前 {config.RERANK_TOP_K} 个结果")
            reranked = docs[:config.RERANK_TOP_K]
        
        elapsed = (time.time() - t_start) * 1000
        logger.debug(f"  └─ Rerank 耗时: {elapsed:.0f}ms")
        
        # 提取结果
        results = [doc['metadata'] for doc in reranked]
        
        context.state = ConversationState.COMPLETED
        
        return ChatResponse(
            message=f"从 {len(files)} 个匹配中为您找到最相关的 **{len(results)}** 个结果：",
            results=results,
            state=ConversationState.COMPLETED,
            can_go_back=context.can_go_back(),
        )
    
    def _extract_filter_keywords(
        self, 
        files: List[Dict], 
        query: str,
        max_options: int = 5
    ) -> List[tuple]:
        """
        从文件列表中提取筛选关键词选项
        
        策略：
        1. 分析文件名中的共同特征词（型号、年份、版本等）
        2. 统计每个关键词出现的文件数
        3. 返回区分度高的关键词作为筛选选项
        
        Returns:
            [(keyword, count), ...] 筛选选项列表
        """
        import re
        from collections import Counter
        
        # 已经在查询中的词不作为选项
        query_words = set(query.lower().split())
        
        # 提取常见的区分性关键词模式
        keyword_patterns = [
            r'国[四五六]',           # 排放标准
            r'[12][90]\d{2}',        # 年份
            r'[A-Z]+\d+[A-Z]*',      # 型号如 DDi11, EDC17, CM2670
            r'高配|低配',            # 配置
            r'新能源|燃油',          # 能源类型
            r'牵引车|载货车|自卸车|搅拌车|消防车',  # 车型
            r'VECU\d*|BCM',          # 控制器类型
            r'整车电路图|仪表模块',   # 文档类型
        ]
        
        keyword_counter = Counter()
        keyword_files = {}  # keyword -> set of file ids
        
        for f in files:
            filename = f.get('filename', '')
            content = f.get('page_content', '')
            file_id = f.get('id', id(f))
            text = f"{filename} {content}"
            
            # 提取匹配的关键词
            for pattern in keyword_patterns:
                matches = re.findall(pattern, text)
                for m in matches:
                    if m.lower() not in query_words and len(m) >= 2:
                        keyword_counter[m] += 1
                        if m not in keyword_files:
                            keyword_files[m] = set()
                        keyword_files[m].add(file_id)
        
        # 筛选有区分度的关键词
        # 不能太少（至少2个文件），也不能太多（不能覆盖所有文件）
        total_files = len(files)
        valid_keywords = []
        
        for kw, count in keyword_counter.most_common(20):
            file_count = len(keyword_files.get(kw, set()))
            # 至少2个文件，且不超过总文件的 80%
            if 2 <= file_count < total_files * 0.9:
                valid_keywords.append((kw, file_count))
        
        # 如果没有找到好的区分关键词，尝试从文件名提取特征词
        if len(valid_keywords) < 2:
            # 提取文件名中的特征词
            word_counter = Counter()
            for f in files:
                filename = f.get('filename', '')
                # 分割文件名
                words = re.split(r'[_\-\s\[\]【】]', filename)
                for word in words:
                    word = word.strip()
                    if len(word) >= 2 and word.lower() not in query_words:
                        if not re.match(r'^[a-z]+$', word.lower()):  # 排除纯小写英文
                            word_counter[word] += 1
            
            for word, count in word_counter.most_common(10):
                if 2 <= count < total_files * 0.9 and word not in [k for k, c in valid_keywords]:
                    valid_keywords.append((word, count))
        
        # 添加"查看全部"选项
        valid_keywords = valid_keywords[:max_options - 1]
        valid_keywords.append(("全部（智能筛选Top5）", total_files))
        
        return valid_keywords
