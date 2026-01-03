#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Query Rewriter - 智能查询预处理

功能：
1. 错别字纠正：天隆 → 天龙
2. 同义词扩展：发动机图纸 → 发动机电路图
3. 口语化处理：解放那个J6P的图 → 解放 J6P
4. 简写补全：EDC17 → EDC17C53
5. 提取关键词：返回结构化的关键词列表

设计原则：
- 模块独立，可单独测试
- 使用智谱AI (glm-4-flash) 进行预处理
- 如果 LLM 不可用，降级到原始输入
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from enum import Enum, auto


class IntentType(Enum):
    """用户意图类型"""
    NEW_SEARCH = auto()      # 全新搜索
    FOLLOW_UP = auto()       # 追问（在当前结果上补充条件）
    SWITCH = auto()          # 切换（换成另一个品牌/系列）
    SELECT = auto()          # 选择（第一个、A、就这个）
    CONTINUE = auto()        # 继续（还有吗、其他的）
    BACK = auto()            # 返回


@dataclass
class QueryRewriteResult:
    """查询改写结果"""
    original_query: str          # 原始查询
    corrected_query: str         # 纠错后的查询
    keywords: List[str]          # 提取的关键词
    expanded_keywords: List[str] # 扩展的关键词（同义词）
    reasoning: str               # 改写理由
    used_llm: bool               # 是否使用了 LLM
    confidence: float            # 置信度 (0-1)
    
    # 上下文理解结果
    intent: IntentType = IntentType.NEW_SEARCH  # 用户意图
    is_followup: bool = False                    # 是否为追问
    select_index: Optional[int] = None           # 如果是选择，选择的索引


@dataclass
class ConversationContext:
    """对话上下文（用于记忆窗口）"""
    last_query: str = ""
    last_options: List[Tuple[str, str, int]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user/assistant", "content": "..."}]


class LLMQueryRewriter:
    """
    LLM 查询改写器
    
    使用 LLM 对用户查询进行预处理：
    1. 纠正错别字
    2. 提取关键词
    3. 扩展同义词
    4. 上下文理解（记忆窗口）
    """
    
    # 常见错别字映射（LLM 降级时使用）
    TYPO_MAP = {
        '天隆': '天龙',
        '天井': '天锦',
        '福天': '福田',
        '欧慢': '欧曼',
        '豪握': '豪沃',
        '凯锐800': '凯锐800',
        '博士': '博世',
        '得尔福': '德尔福',
        '康明斯': '康明斯',
        '电陆图': '电路图',
        '线速图': '线束图',
    }
    
    # 同义词映射（LLM 降级时使用）
    SYNONYM_MAP = {
        '图纸': ['电路图'],
        '电气图': ['电路图'],
        '接线图': ['电路图', '线束图'],
        '原理图': ['电路图'],
        '引擎': ['发动机'],
        '马达': ['发动机'],
        '仪表盘': ['仪表'],
        '变速器': ['变速箱'],
        '空调系统': ['空调'],
        '整车': ['整车电路图'],
    }
    
    def __init__(
        self, 
        use_llm: bool = True, 
        debug: bool = False,
        domain_knowledge_path: Optional[str] = None,
        synonyms_path: Optional[str] = None,
    ):
        """
        初始化查询改写器
        
        Args:
            use_llm: 是否启用 LLM（推荐启用）
            debug: 是否输出调试信息
            domain_knowledge_path: 领域知识文件路径（可选）
            synonyms_path: 同义词表路径（可选，默认加载 config/synonyms.json）
        """
        self.use_llm = use_llm
        self.debug = debug
        self._llm_client = None
        self._domain_terms = set()
        
        # 加载领域词汇
        self._load_domain_knowledge(domain_knowledge_path)
        
        # 加载同义词表
        self._load_synonyms(synonyms_path)
    
    def _load_domain_knowledge(self, path: Optional[str] = None):
        """加载领域词汇（用于构建 prompt）"""
        # 默认词汇
        self._domain_terms = {
            # 品牌
            '东风', '福田', '江淮', '江铃', '解放', '重汽', '陕汽', '北汽',
            '柳汽', '华菱', '三一', '徐工', '临工', '雷沃', '中联',
            # 系列
            '天龙', '天锦', '欧曼', '欧辉', '凯运', '凯锐', '奥铃', '骏铃',
            '康铃', '帅铃', '豪沃', '斯太尔', '德龙', '乘龙', 'J6', 'J6P', 'J7',
            # 电路类型
            '整车电路图', 'ECU电路图', '发动机电路图', '仪表电路图', '空调电路图',
            '线束图', '电器盒', '启动原理图', '充电原理图',
            # ECU 型号
            'EDC17', 'EDC17C53', 'EDC7', 'CM2150', 'DCU', 'VECU',
            '博世', '德尔福', '康明斯', '威孚',
        }
        
        # 从文件加载（如果提供）
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self._domain_terms.add(term)
            except Exception as e:
                if self.debug:
                    print(f"加载领域词汇失败: {e}")
    
    def _load_synonyms(self, path: Optional[str] = None):
        """加载同义词表"""
        # 确定同义词文件路径
        if path is None:
            synonyms_path = Path(__file__).parent.parent / "config" / "synonyms.json"
        else:
            synonyms_path = Path(path)
        
        # 构建双向映射：别名 -> 标准词
        self._synonym_to_standard = {}
        
        if synonyms_path.exists():
            try:
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理各类同义词
                for category in ['品牌别名', '系列别名', '电路类型同义词', '部件同义词', 'ECU型号别名', '车辆类型同义词']:
                    if category in data:
                        for standard, aliases in data[category].items():
                            for alias in aliases:
                                self._synonym_to_standard[alias.lower()] = standard
                
                # 处理错别字映射
                if '常见错别字' in data:
                    for typo, correct in data['常见错别字'].items():
                        self.TYPO_MAP[typo] = correct
                
                if self.debug:
                    print(f"  └─ 已加载同义词表: {len(self._synonym_to_standard)} 条映射")
                    
            except Exception as e:
                if self.debug:
                    print(f"加载同义词表失败: {e}")
                self._synonym_to_standard = {}
        else:
            self._synonym_to_standard = {}
    
    def normalize_query(self, query: str) -> str:
        """
        使用同义词表标准化查询
        将别名替换为标准术语
        """
        result = query
        for alias, standard in self._synonym_to_standard.items():
            if alias in result.lower():
                # 保持原始大小写进行替换
                import re
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                result = pattern.sub(standard, result)
        return result

    @property
    def llm_client(self):
        """延迟加载 LLM 客户端"""
        if self._llm_client is None and self.use_llm:
            try:
                from zhipuai import ZhipuAI
                api_key = os.getenv(
                    "ZHIPU_API_KEY", 
                    "0b77d446617f46d1b2288285d18481d3.1W9lkeXeQfMQw4Ik"
                )
                if api_key:
                    self._llm_client = ZhipuAI(api_key=api_key)
            except ImportError:
                if self.debug:
                    print("zhipuai 库未安装，LLM 功能不可用")
        return self._llm_client
    
    def rewrite(self, query: str) -> QueryRewriteResult:
        """
        改写用户查询
        
        Args:
            query: 原始用户查询
        
        Returns:
            QueryRewriteResult 改写结果
        """
        query = query.strip()
        
        if not query:
            return QueryRewriteResult(
                original_query=query,
                corrected_query=query,
                keywords=[],
                expanded_keywords=[],
                reasoning="空查询",
                used_llm=False,
                confidence=1.0,
            )
        
        # 尝试使用 LLM
        if self.use_llm and self.llm_client:
            result = self._llm_rewrite(query)
            if result:
                if self.debug:
                    print(f"[LLM] '{query}' → '{result.corrected_query}'")
                    print(f"      关键词: {result.keywords}")
                    print(f"      扩展词: {result.expanded_keywords}")
                return result
        
        # 降级到规则方法
        return self._rule_based_rewrite(query)
    
    def _llm_rewrite(self, query: str) -> Optional[QueryRewriteResult]:
        """使用 LLM 进行查询改写"""
        
        # 构建提示词
        domain_terms_sample = ', '.join(list(self._domain_terms)[:30])
        
        prompt = f"""你是电路图检索系统的查询预处理助手。请对用户查询进行处理。

## 领域词汇参考
{domain_terms_sample}

## 用户查询
{query}

## 处理任务
1. **错别字纠正**：修正常见错误（如"天隆"→"天龙"，"博士"→"博世"）
2. **关键词提取**：提取用于检索的核心词汇
3. **同义词扩展**：添加相关术语以提高召回（如"图纸"可扩展为"电路图"）
4. **口语规范化**：将口语化表达转为标准术语

## 输出要求
返回 JSON 格式（不要 markdown 代码块）：
{{"corrected_query": "纠错后的完整查询", "keywords": ["关键词1", "关键词2"], "expanded_keywords": ["扩展词1"], "reasoning": "处理说明"}}

## 示例
输入: "天隆的ECU图"
输出: {{"corrected_query": "天龙ECU电路图", "keywords": ["天龙", "ECU电路图"], "expanded_keywords": ["东风", "ECU"], "reasoning": "纠正'天隆'为'天龙'，补全'图'为'电路图'"}}

输入: "解放那个J6P的发动机图纸"
输出: {{"corrected_query": "解放J6P发动机电路图", "keywords": ["解放", "J6P", "发动机电路图"], "expanded_keywords": ["发动机"], "reasoning": "移除口语词'那个'，'图纸'规范为'电路图'"}}

请处理用户查询："""

        try:
            response = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 处理可能的 markdown 代码块
            if '```' in result_text:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            result = json.loads(result_text)
            
            return QueryRewriteResult(
                original_query=query,
                corrected_query=result.get('corrected_query', query),
                keywords=result.get('keywords', []),
                expanded_keywords=result.get('expanded_keywords', []),
                reasoning=result.get('reasoning', 'LLM 处理'),
                used_llm=True,
                confidence=0.95,
            )
            
        except Exception as e:
            if self.debug:
                print(f"LLM 改写失败: {e}")
            return None
    
    def _rule_based_rewrite(self, query: str) -> QueryRewriteResult:
        """
        基于规则的查询改写（LLM 降级方案）
        """
        corrected = query
        keywords = []
        expanded = []
        reasons = []
        
        # 1. 错别字纠正
        for typo, correct in self.TYPO_MAP.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correct)
                reasons.append(f"纠正 '{typo}' → '{correct}'")
        
        # 2. 同义词扩展
        for term, synonyms in self.SYNONYM_MAP.items():
            if term in corrected:
                expanded.extend(synonyms)
                reasons.append(f"扩展 '{term}' → {synonyms}")
        
        # 3. 使用 jieba 提取关键词
        try:
            import jieba
            
            # 添加领域词
            for term in self._domain_terms:
                jieba.add_word(term)
            
            words = list(jieba.cut(corrected))
            
            stop_words = {'的', '了', '和', '与', '或', '在', '是', '有', 
                          '我', '要', '找', '查', '搜索', '请', '帮', '看',
                          '那个', '这个', '什么', '哪个', '吗', '呢', '吧'}
            
            for w in words:
                w = w.strip()
                if w and w not in stop_words and len(w) >= 1:
                    if w not in keywords:
                        keywords.append(w)
        except ImportError:
            # jieba 不可用，简单按空格分词
            keywords = [w.strip() for w in corrected.split() if w.strip()]
        
        reasoning = '; '.join(reasons) if reasons else "规则处理（无改动）"
        
        return QueryRewriteResult(
            original_query=query,
            corrected_query=corrected,
            keywords=keywords,
            expanded_keywords=list(set(expanded)),
            reasoning=reasoning,
            used_llm=False,
            confidence=0.7,
        )
    
    # ============ 上下文感知改写（记忆窗口） ============
    
    # 汉字数字映射
    CN_NUM_MAP = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
        '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    }
    
    # 选择类模式（确定性高，用规则处理）
    SELECT_PATTERNS = [
        (r'^第([一二三四五六七八九十\d]+)[个条]?$', 'number'),  # 第一个、第3个
        (r'^([一二三四五六七八九十\d]+)号?$', 'number'),        # 3号 / 三号
        (r'^就([这那])个$', 'this'),            # 就这个
        (r'^选([这那])个$', 'this'),            # 选这个
        (r'^([A-Ja-j])$', 'letter'),            # A, B, C...
        (r'^([A-Ja-j])[.。,，]?$', 'letter'),   # A. / A。
    ]
    
    # 继续类模式
    CONTINUE_PATTERNS = [
        r'^还有[吗么]?[？?]?$',                 # 还有吗
        r'^还有其他的?[吗么]?[？?]?$',          # 还有其他的吗
        r'^其他的?呢?[？?]?$',                  # 其他的呢
        r'^更多$',                              # 更多
        r'^继续$',                              # 继续
        r'^下一[页个]$',                        # 下一页
    ]
    
    def rewrite_with_context(
        self,
        query: str,
        last_query: str = "",
        conversation_history: List[Dict[str, str]] = None,
        pending_options: List[Tuple[str, str, int]] = None,
    ) -> QueryRewriteResult:
        """
        上下文感知的查询改写（记忆窗口）
        
        Args:
            query: 当前用户输入
            last_query: 上一次的检索查询
            conversation_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            pending_options: 当前待选择的选项 [(path, name, count), ...]
        
        Returns:
            QueryRewriteResult（包含意图判断）
        """
        query = query.strip()
        
        if not query:
            return QueryRewriteResult(
                original_query=query,
                corrected_query=query,
                keywords=[],
                expanded_keywords=[],
                reasoning="空查询",
                used_llm=False,
                confidence=1.0,
                intent=IntentType.NEW_SEARCH,
            )
        
        # 1. 先用规则处理确定性高的情况
        rule_result = self._try_context_rules(query, last_query, pending_options)
        if rule_result:
            if self.debug:
                print(f"[规则] '{query}' → intent={rule_result.intent.name}")
            return rule_result
        
        # 2. 如果没有上下文，直接做普通改写
        if not last_query and not conversation_history:
            return self.rewrite(query)
        
        # 3. 使用 LLM 理解上下文
        if self.use_llm and self.llm_client:
            llm_result = self._llm_rewrite_with_context(
                query, last_query, conversation_history
            )
            if llm_result:
                if self.debug:
                    print(f"[LLM Context] '{query}' → '{llm_result.corrected_query}' (intent={llm_result.intent.name})")
                return llm_result
        
        # 4. 降级：当作新查询处理
        return self.rewrite(query)
    
    def _try_context_rules(
        self,
        query: str,
        last_query: str,
        pending_options: List[Tuple[str, str, int]] = None,
    ) -> Optional[QueryRewriteResult]:
        """尝试用规则处理确定性高的情况"""
        
        # 辅助函数：解析汉字/阿拉伯数字
        def parse_number(s: str) -> int:
            if s.isdigit():
                return int(s)
            # 汉字数字
            total = 0
            for c in s:
                if c in self.CN_NUM_MAP:
                    total = total * 10 + self.CN_NUM_MAP[c] if total else self.CN_NUM_MAP[c]
            return total if total else 1
        
        # 1. 检查选择类（A/B/C、第N个）
        for pattern, ptype in self.SELECT_PATTERNS:
            match = re.match(pattern, query)
            if match:
                if ptype == 'number':
                    idx = parse_number(match.group(1)) - 1  # 转 0-based
                elif ptype == 'letter':
                    idx = ord(match.group(1).upper()) - ord('A')
                else:
                    idx = 0  # "就这个" 默认第一个
                
                return QueryRewriteResult(
                    original_query=query,
                    corrected_query=last_query or query,
                    keywords=[],
                    expanded_keywords=[],
                    reasoning=f"选择第 {idx + 1} 个选项",
                    used_llm=False,
                    confidence=0.99,
                    intent=IntentType.SELECT,
                    is_followup=True,
                    select_index=idx,
                )
        
        # 2. 检查继续类
        for pattern in self.CONTINUE_PATTERNS:
            if re.match(pattern, query):
                return QueryRewriteResult(
                    original_query=query,
                    corrected_query=last_query or query,
                    keywords=[],
                    expanded_keywords=[],
                    reasoning="继续查看更多结果",
                    used_llm=False,
                    confidence=0.99,
                    intent=IntentType.CONTINUE,
                    is_followup=True,
                )
        
        # 3. 检查返回
        if query in ('返回', '上一级', '后退', '回去'):
            return QueryRewriteResult(
                original_query=query,
                corrected_query=last_query or query,
                keywords=[],
                expanded_keywords=[],
                reasoning="返回上一级",
                used_llm=False,
                confidence=0.99,
                intent=IntentType.BACK,
                is_followup=True,
            )
        
        # 4. 检查简单的替换类（换成XX的）
        replace_match = re.match(r'^换成(.+?)的?$', query)
        if replace_match and last_query:
            new_brand = replace_match.group(1).strip()
            # 尝试构建新查询
            rewritten = new_brand + '电路图'  # 简单处理
            return QueryRewriteResult(
                original_query=query,
                corrected_query=rewritten,
                keywords=[new_brand],
                expanded_keywords=['电路图'],
                reasoning=f"替换为 {new_brand}",
                used_llm=False,
                confidence=0.85,
                intent=IntentType.SWITCH,
                is_followup=True,
            )
        
        return None
    
    def _llm_rewrite_with_context(
        self,
        query: str,
        last_query: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Optional[QueryRewriteResult]:
        """使用 LLM 进行上下文感知的改写"""
        
        # 构建对话历史字符串
        history_str = ""
        if conversation_history:
            for turn in conversation_history[-6:]:  # 最近3轮
                role = "用户" if turn.get('role') == 'user' else "助手"
                content = turn.get('content', '')[:150]
                history_str += f"{role}: {content}\n"
        
        prompt = f"""你是电路图检索助手。请分析用户输入，判断意图并改写查询。

## 对话历史
{history_str if history_str else "（无）"}

## 上一次查询
{last_query if last_query else "（无）"}

## 当前用户输入
{query}

## 意图类型
- NEW_SEARCH: 全新搜索（与上文无关的新查询）
- FOLLOW_UP: 追问（在上次结果基础上补充条件，如"仪表的呢？"）
- SWITCH: 切换（换成另一个品牌/系列，如"福田的呢？"）
- SELECT: 选择（如"第一个"、"A"）
- CONTINUE: 继续（如"还有吗"）

## 改写规则
1. 新查询 → 纠错并提取关键词
2. 追问 → 结合上次查询生成完整查询
3. 切换 → 保留电路类型，替换品牌/系列

## 示例
| 上次查询 | 用户输入 | 意图 | 改写结果 |
|---------|---------|------|---------|
| 东风天龙整车电路图 | 仪表电路图呢 | FOLLOW_UP | 东风天龙仪表电路图 |
| 东风天龙整车电路图 | 福田的呢 | SWITCH | 福田整车电路图 |
| 东风天龙整车电路图 | 解放J6P发动机 | NEW_SEARCH | 解放J6P发动机电路图 |
| （无） | 天隆电路图 | NEW_SEARCH | 天龙电路图 |

## 输出JSON（不要markdown代码块）
{{"intent": "NEW_SEARCH/FOLLOW_UP/SWITCH", "corrected_query": "改写后的完整查询", "keywords": ["关键词"], "reasoning": "理由"}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 处理 markdown 代码块
            if '```' in result_text:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            result = json.loads(result_text)
            
            # 解析意图
            intent_str = result.get('intent', 'NEW_SEARCH').upper()
            intent_map = {
                'NEW_SEARCH': IntentType.NEW_SEARCH,
                'FOLLOW_UP': IntentType.FOLLOW_UP,
                'SWITCH': IntentType.SWITCH,
                'SELECT': IntentType.SELECT,
                'CONTINUE': IntentType.CONTINUE,
            }
            intent = intent_map.get(intent_str, IntentType.NEW_SEARCH)
            is_followup = intent != IntentType.NEW_SEARCH
            
            return QueryRewriteResult(
                original_query=query,
                corrected_query=result.get('corrected_query', query),
                keywords=result.get('keywords', []),
                expanded_keywords=result.get('expanded_keywords', []),
                reasoning=result.get('reasoning', 'LLM 上下文处理'),
                used_llm=True,
                confidence=0.90,
                intent=intent,
                is_followup=is_followup,
            )
            
        except Exception as e:
            if self.debug:
                print(f"LLM 上下文改写失败: {e}")
            return None
