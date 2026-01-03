#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导航树引擎 - 基于层级路径的树状导航

核心思想：
资料清单的层级路径 "电路图->整车电路图->东风->天龙" 天然形成了一棵树。
用户可以逐层选择导航，也可以直接搜索跳到匹配的节点。

树结构示例：
    电路图
    ├── ECU电路图
    │   ├── 工程机械
    │   │   ├── 三一
    │   │   │   ├── SY60 [3个文件]
    │   │   │   └── SY115C9 [2个文件]
    │   │   └── 徐工
    │   │       └── XE135G [3个文件]
    │   └── 商用车
    │       └── ...
    └── 整车电路图
        ├── 东风
        │   ├── 天龙 [63个文件]
        │   └── 天锦 [30个文件]
        └── 福田
            └── ...

工作流程：
1. 用户输入 "天龙电路图"
2. 搜索匹配的节点 → 找到 "电路图->整车电路图->东风->天龙"
3. 如果只有一个节点匹配 → 返回该节点下的所有文件 (或 Top 5)
4. 如果多个节点匹配 → 让用户选择是哪个节点
5. 如果叶子节点文件过多 → 使用 Reranker 选 Top 5
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TreeNode:
    """导航树节点"""
    name: str                           # 节点名称（如 "东风"）
    path: str                           # 完整路径（如 "电路图->整车电路图->东风"）
    depth: int                          # 深度（从 0 开始）
    parent: Optional['TreeNode'] = None # 父节点
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)  # 子节点
    files: List[Dict] = field(default_factory=list)  # 直属文件（叶子节点才有）
    
    @property
    def is_leaf(self) -> bool:
        """是否是叶子节点（没有子节点）"""
        return len(self.children) == 0
    
    @property
    def total_files(self) -> int:
        """子树下的总文件数"""
        count = len(self.files)
        for child in self.children.values():
            count += child.total_files
        return count
    
    def get_all_files(self) -> List[Dict]:
        """获取子树下的所有文件"""
        all_files = list(self.files)
        for child in self.children.values():
            all_files.extend(child.get_all_files())
        return all_files
    
    def get_children_summary(self) -> List[Tuple[str, int]]:
        """获取子节点摘要 [(name, file_count), ...]"""
        return [(name, child.total_files) for name, child in self.children.items()]
    
    def to_dict(self) -> Dict:
        """序列化（不含 parent 避免循环）"""
        return {
            'name': self.name,
            'path': self.path,
            'depth': self.depth,
            'children': {k: v.to_dict() for k, v in self.children.items()},
            'file_count': len(self.files),
            'total_files': self.total_files,
        }


class NavigationTree:
    """
    导航树 - 管理资料的层级结构
    
    支持：
    1. 从 JSON 数据构建树
    2. 按路径导航（逐层选择）
    3. 模糊搜索匹配节点
    4. 获取节点下的文件
    """
    
    # 配置
    MAX_DIRECT_RESULTS = 5    # 直接返回的最大文件数
    MAX_OPTIONS_PER_LEVEL = 5 # 每层最多显示的选项数
    
    def __init__(self, data_path: Optional[str] = None):
        """
        初始化导航树
        
        Args:
            data_path: JSON 数据文件路径（可选）
        """
        self.root = TreeNode(name="root", path="", depth=-1)
        self._node_index: Dict[str, TreeNode] = {}  # path -> node 索引
        self._keyword_index: Dict[str, List[TreeNode]] = defaultdict(list)  # 关键词索引
        
        if data_path:
            self.load_from_json(data_path)
    
    def load_from_json(self, path: str):
        """从 JSON 文件加载数据并构建树"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            self._add_item(item)
        
        # 构建关键词索引
        self._build_keyword_index()
        
        print(f"📂 导航树构建完成: {len(self._node_index)} 个节点, {self.root.total_files} 个文件")
    
    def _add_item(self, item: Dict):
        """将一个数据项添加到树中"""
        path = item.get('path', '')
        if not path:
            return
        
        segments = path.split('->')
        current = self.root
        current_path = ""
        
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            
            current_path = f"{current_path}->{segment}" if current_path else segment
            
            if segment not in current.children:
                node = TreeNode(
                    name=segment,
                    path=current_path,
                    depth=i,
                    parent=current,
                )
                current.children[segment] = node
                self._node_index[current_path] = node
            
            current = current.children[segment]
        
        # 将文件添加到叶子节点
        file_info = {
            'id': item.get('id'),
            'filename': item.get('filename', ''),
            'path': path,
            'brand': item.get('brand', ''),
            'series': item.get('series', ''),
            'doc_type': item.get('doc_type', ''),
            'page_content': item.get('page_content', ''),
        }
        current.files.append(file_info)
    
    def _build_keyword_index(self):
        """构建关键词到节点的索引"""
        for path, node in self._node_index.items():
            # 节点名本身
            self._keyword_index[node.name.lower()].append(node)
            
            # 路径中的每个片段
            for segment in path.split('->'):
                segment = segment.strip().lower()
                if segment and segment != node.name.lower():
                    self._keyword_index[segment].append(node)
    
    def get_node(self, path: str) -> Optional[TreeNode]:
        """按路径获取节点"""
        return self._node_index.get(path)
    
    def get_children(self, path: str = "") -> List[Tuple[str, int]]:
        """
        获取某路径下的子节点列表
        
        Args:
            path: 节点路径，空字符串表示根节点
        
        Returns:
            [(子节点名, 文件数), ...]
        """
        if not path:
            node = self.root
        else:
            node = self._node_index.get(path)
            if not node:
                return []
        
        return sorted(node.get_children_summary(), key=lambda x: -x[1])
    
    def search_nodes(self, query: str, max_results: int = 20) -> List[TreeNode]:
        """
        搜索匹配的节点
        
        策略：
        1. 多关键词联合匹配 → 匹配越多分数越高
        2. 精确匹配节点名 → 高优先级
        3. 优先返回有文件的节点
        4. 优先返回层级较深的具体节点
        
        Args:
            query: 搜索词
            max_results: 最大返回数量
        
        Returns:
            匹配的节点列表（按相关性排序）
        """
        # 提取查询中的关键词
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # 评分：节点 -> (匹配关键词数, 总分)
        scores: Dict[str, Tuple[int, float]] = {}
        
        for path, node in self._node_index.items():
            path_lower = path.lower()
            node_name_lower = node.name.lower()
            
            matched_keywords = 0
            total_score = 0.0
            
            for kw in keywords:
                kw_lower = kw.lower()
                
                # 精确匹配节点名
                if node_name_lower == kw_lower:
                    matched_keywords += 1
                    total_score += 20.0
                # 节点名包含关键词
                elif kw_lower in node_name_lower:
                    matched_keywords += 1
                    total_score += 10.0
                # 路径包含关键词（任意位置）
                elif kw_lower in path_lower:
                    matched_keywords += 1
                    total_score += 5.0
            
            if matched_keywords > 0:
                # 加分：匹配的关键词越多越好
                total_score += matched_keywords * 15.0
                
                # 加分：有文件的节点，文件越多分数越高
                if node.total_files > 0:
                    total_score += min(node.total_files / 10, 10)  # 最多加10分
                
                # 加分：层级适中（太浅信息少，太深太具体）
                if 2 <= node.depth <= 4:
                    total_score += 2.0
                
                scores[path] = (matched_keywords, total_score)
        
        # 按 (匹配关键词数 desc, 总分 desc) 排序
        sorted_paths = sorted(
            scores.keys(), 
            key=lambda p: (-scores[p][0], -scores[p][1])
        )
        
        return [self._node_index[p] for p in sorted_paths[:max_results]]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词（使用 jieba 分词）"""
        import jieba
        
        # 添加自定义词汇（品牌、系列名）
        custom_words = [
            '整车电路图', 'ECU电路图', '东风', '福田', '江淮', '江铃', '解放', '重汽', 
            '陕汽', '北汽', '天龙', '天锦', '欧曼', '欧辉', '凯运', '凯锐', '奥铃',
            '豪沃', '斯太尔', '骏铃', '康铃', '帅铃', '德龙', '乘龙', '柳汽',
        ]
        for w in custom_words:
            jieba.add_word(w)
        
        # 分词
        words = list(jieba.cut(query))
        
        # 保留有意义的技术词
        preserve_words = {'电路图', '整车电路图', 'ECU电路图', '发动机', '仪表', '变速箱', 
                          '整车', '新能源', '传感器', '电器盒', '线束'}
        
        # 移除无意义词
        stop_words = {'的', '了', '和', '与', '或', '在', '是', '有', '我', '要', '找', '查', '搜索', '请', '帮', '看'}
        
        # 过滤
        keywords = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            if w in preserve_words:
                keywords.append(w)
            elif w.lower() not in stop_words and len(w) >= 2:
                keywords.append(w)
        
        # 去重保持顺序
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        # 按长度排序（优先匹配长词）
        unique_keywords.sort(key=len, reverse=True)
        
        return unique_keywords
    
    def navigate(self, query: str, current_path: str = "") -> 'NavigationResult':
        """
        智能导航：根据查询和当前位置，决定下一步
        
        Args:
            query: 用户查询（可以是搜索词或选项字母）
            current_path: 当前所在路径
        
        Returns:
            NavigationResult
        """
        # 如果当前在某个节点，先看用户是否选择了子节点
        if current_path:
            current_node = self.get_node(current_path)
            if current_node:
                # 检查是否是选择某个子节点（精确或模糊匹配）
                best_match = None
                best_score = 0
                for name, child in current_node.children.items():
                    name_lower = name.lower()
                    query_lower = query.lower()
                    if name_lower == query_lower:
                        best_match = child
                        best_score = 100
                        break
                    elif query_lower in name_lower:
                        score = len(query_lower) / len(name_lower) * 50
                        if score > best_score:
                            best_match = child
                            best_score = score
                    elif name_lower in query_lower:
                        score = len(name_lower) / len(query_lower) * 40
                        if score > best_score:
                            best_match = child
                            best_score = score
                
                if best_match and best_score >= 30:
                    return self._node_to_result(best_match)
        
        # 搜索匹配的节点
        matches = self.search_nodes(query)
        
        if not matches:
            return self._build_no_match_response(query)
        
        # 过滤掉文件数为0的节点
        matches = [m for m in matches if m.total_files > 0]
        
        if len(matches) == 0:
            return self._build_no_match_response(query)
        
        # 如果只有一个匹配，直接返回
        if len(matches) == 1:
            return self._node_to_result(matches[0])
        
        # 多个匹配：检查是否来自不同的顶层分类
        # 如果匹配分布在不同大类（如"仪表模块" vs "整车电路图"），让用户先选择大类
        top_categories = self._get_distinct_categories(matches)
        
        if len(top_categories) > 1:
            # 匹配来自多个不同大类，让用户先选择
            return self._present_category_choices(matches, top_categories, query)
        
        # 同一大类下的多个匹配
        keywords = self._extract_keywords(query)
        if keywords:
            first_kw_count = sum(1 for kw in keywords if kw.lower() in matches[0].path.lower())
            second_kw_count = sum(1 for kw in keywords if kw.lower() in matches[1].path.lower()) if len(matches) > 1 else 0
            
            # 只有在第一个匹配明显更好时才直接返回
            if first_kw_count > second_kw_count + 1:
                return self._node_to_result(matches[0])
        
        # 多个匹配且没有明显赢家：如果选项超过阈值，先分层
        if len(matches) > self.MAX_OPTIONS_PER_LEVEL:
            return self._group_matches_hierarchically(matches, query)
        
        # 选项数量适中，直接展示
        return NavigationResult(
            status='multiple_matches',
            message=f"找到 {len(matches)} 个相关分类，请选择：",
            options=[(n.path, self._get_display_name(n), n.total_files) for n in matches],
        )
    
    def _group_matches_hierarchically(self, matches: List['TreeNode'], query: str) -> 'NavigationResult':
        """
        当匹配过多时，将匹配按上层分类分组，让用户先选择大类
        
        策略：找到匹配节点的共同祖先层级，按该层级分组
        """
        from collections import defaultdict
        
        # 分析所有匹配的路径结构
        # 找到合适的分组层级
        
        # 尝试按不同层级分组，找到分组数量在 2-5 之间的层级
        best_level = 1
        best_groups = {}
        
        for level in range(1, 4):  # 尝试第1、2、3层
            groups = defaultdict(list)
            for node in matches:
                parts = node.path.split('->')
                if len(parts) >= level:
                    group_key = '->'.join(parts[:level])
                    groups[group_key].append(node)
                else:
                    groups[node.path].append(node)
            
            num_groups = len(groups)
            if 2 <= num_groups <= self.MAX_OPTIONS_PER_LEVEL:
                best_level = level
                best_groups = dict(groups)
                break
            elif num_groups > self.MAX_OPTIONS_PER_LEVEL:
                # 分组太多，保留这一层但只取前几个
                best_level = level
                best_groups = dict(groups)
                break
            else:
                # 分组太少，继续尝试更深层级
                best_level = level
                best_groups = dict(groups)
        
        if len(best_groups) <= 1:
            # 无法有效分组，直接返回前几个匹配
            return NavigationResult(
                status='multiple_matches',
                message=f"找到 {len(matches)} 个相关分类，请选择：",
                options=[(n.path, self._get_display_name(n), n.total_files) 
                         for n in matches[:self.MAX_OPTIONS_PER_LEVEL]],
            )
        
        # 构建分组选项
        options = []
        for group_path, nodes in sorted(best_groups.items(), 
                                         key=lambda x: sum(n.total_files for n in x[1]), 
                                         reverse=True):
            total_files = sum(n.total_files for n in nodes)
            # 获取分组显示名
            group_node = self.get_node(group_path)
            if group_node:
                display_name = group_node.name
            else:
                display_name = group_path.split('->')[-1] if '->' in group_path else group_path
            
            options.append((group_path, display_name, total_files))
        
        # 限制选项数量
        options = options[:self.MAX_OPTIONS_PER_LEVEL]
        
        return NavigationResult(
            status='navigate',
            message=f"找到 {len(matches)} 个相关分类，请先选择大类：",
            options=options,
        )
    
    def _get_distinct_categories(self, matches: List['TreeNode']) -> Dict[str, List['TreeNode']]:
        """
        将匹配节点按顶层分类分组
        
        例如：
        - 电路图->仪表模块->东风->天龙
        - 电路图->整车电路图->东风->天龙
        
        会分成 "仪表模块" 和 "整车电路图" 两个分类
        """
        from collections import defaultdict
        
        categories = defaultdict(list)
        
        for node in matches:
            parts = node.path.split('->')
            # 取第二层作为分类（跳过"电路图"这种顶层）
            if len(parts) >= 2:
                # 常见顶层：电路图、整车，跳过取第二层
                if parts[0] in ('电路图', '整车'):
                    category = parts[1] if len(parts) > 1 else parts[0]
                else:
                    category = parts[0]
            else:
                category = parts[0]
            
            categories[category].append(node)
        
        return dict(categories)
    
    def _present_category_choices(
        self, 
        matches: List['TreeNode'], 
        categories: Dict[str, List['TreeNode']], 
        query: str
    ) -> 'NavigationResult':
        """
        当匹配来自多个不同大类时，让用户先选择大类
        """
        options = []
        
        for category, nodes in sorted(categories.items(), 
                                       key=lambda x: sum(n.total_files for n in x[1]), 
                                       reverse=True):
            total_files = sum(n.total_files for n in nodes)
            
            # 找到这个分类的代表节点（文件最多的）
            best_node = max(nodes, key=lambda n: n.total_files)
            
            # 显示名称：分类名 + 子结构信息
            if best_node.children:
                sub_info = f"（含 {len(best_node.children)} 个子分类）"
            else:
                sub_info = ""
            
            display_name = f"{category}{sub_info}"
            
            # 使用代表节点的路径
            options.append((best_node.path, display_name, total_files))
        
        # 限制选项数量
        options = options[:self.MAX_OPTIONS_PER_LEVEL]
        
        total_files = sum(n.total_files for n in matches)
        
        return NavigationResult(
            status='multiple_matches',
            message=f"找到 {total_files} 个相关资料，分布在 {len(categories)} 个分类中，请选择：",
            options=options,
        )
    
    def _get_display_name(self, node: TreeNode) -> str:
        """获取节点的显示名称（带上下文，用于区分同名节点）"""
        # 构建面包屑路径（最多显示3级）
        parts = node.path.split('->')
        if len(parts) <= 2:
            return node.path
        # 显示最后3级
        return ' > '.join(parts[-3:])
    
    def _build_no_match_response(self, query: str) -> 'NavigationResult':
        """
        构建友好的无结果响应，包含建议和热门推荐
        """
        # 提取查询关键词用于建议
        keywords = self._extract_keywords(query)
        
        # 构建友好提示消息
        lines = [f"😔 抱歉，没有找到与 **{query}** 相关的资料。"]
        lines.append("")
        
        # 提供建议
        lines.append("💡 **建议您尝试：**")
        suggestions = []
        
        # 检查是否可能是错别字
        typo_hints = self._check_possible_typos(query)
        if typo_hints:
            suggestions.append(f"检查拼写，您是否想搜索：**{typo_hints[0]}**？")
        
        suggestions.extend([
            "使用更简短的关键词，如品牌名或系列名",
            "尝试同义词，如 '电路图' 也可搜 '线路图'、'接线图'",
            "分开搜索多个关键词",
        ])
        
        for i, sug in enumerate(suggestions[:4], 1):
            lines.append(f"  {i}. {sug}")
        
        lines.append("")
        
        # 添加热门资料推荐
        popular_categories = self._get_popular_categories()
        if popular_categories:
            lines.append("📂 **热门资料分类：**")
            for name, count in popular_categories[:5]:
                lines.append(f"  • {name}（{count}个文件）")
        
        return NavigationResult(
            status='no_match',
            message='\n'.join(lines),
        )
    
    def _check_possible_typos(self, query: str) -> List[str]:
        """检查可能的错别字，返回可能的正确词"""
        # 常见错别字映射
        typo_map = {
            '天隆': '天龙', '天井': '天锦', '福天': '福田',
            '欧慢': '欧曼', '豪握': '豪沃', '博士': '博世',
            '得尔福': '德尔福', '电陆图': '电路图', '线速图': '线束图',
            '奥玲': '奥铃', '程龙': '乘龙', '斯泰尔': '斯太尔',
        }
        
        hints = []
        for typo, correct in typo_map.items():
            if typo in query:
                corrected = query.replace(typo, correct)
                hints.append(corrected)
        
        return hints
    
    def _get_popular_categories(self) -> List[Tuple[str, int]]:
        """获取热门资料分类"""
        # 从根节点的子节点中获取文件数最多的分类
        if not self.root or not self.root.children:
            return []
        
        categories = []
        for child in self.root.children.values():
            # 获取第二层（更具体的分类）
            if child.children:
                for grandchild in child.children.values():
                    categories.append((
                        f"{child.name} > {grandchild.name}",
                        grandchild.total_files
                    ))
            else:
                categories.append((child.name, child.total_files))
        
        # 按文件数排序
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories[:8]
    
    def _find_common_prefix(self, paths: List[str]) -> str:
        """找到路径列表的共同前缀"""
        if not paths:
            return ""
        
        # 分割成段
        segments_list = [p.split('->') for p in paths]
        min_len = min(len(s) for s in segments_list)
        
        common = []
        for i in range(min_len):
            seg = segments_list[0][i]
            if all(s[i] == seg for s in segments_list):
                common.append(seg)
            else:
                break
        
        return '->'.join(common)
    
    def _node_to_result(self, node: TreeNode) -> 'NavigationResult':
        """将节点转换为导航结果"""
        total_files = node.total_files
        
        # Case 1: 有子节点 → 优先让用户选择子节点（即使文件数量少）
        if len(node.children) > 0:
            children = node.get_children_summary()
            # 如果只有一个子节点且文件很少，可以考虑直接进入
            if len(children) == 1 and total_files <= self.MAX_DIRECT_RESULTS:
                # 只有一个子分类且文件少，直接返回所有文件
                files = node.get_all_files()
                return NavigationResult(
                    status='files',
                    current_path=node.path,
                    message=f"在 **{node.name}** 下找到 {len(files)} 个文件：",
                    files=files,
                )
            # 多个子节点或文件较多，让用户选择
            return NavigationResult(
                status='navigate',
                current_path=node.path,
                message=f"在 **{node.name}** 下找到 {total_files} 个相关资料，请选择分类：",
                options=[(f"{node.path}->{name}", name, count) for name, count in children],
            )
        
        # Case 2: 叶子节点，文件数量少 → 直接返回
        if total_files <= self.MAX_DIRECT_RESULTS:
            files = node.get_all_files()
            return NavigationResult(
                status='files',
                current_path=node.path,
                message=f"在 **{node.name}** 下找到 {len(files)} 个文件：",
                files=files,
            )
        
        # Case 3: 叶子节点，文件超过阈值 → 让用户进一步筛选
        return NavigationResult(
            status='need_filter',
            current_path=node.path,
            message=f"在 **{node.name}** 下找到 {total_files} 个文件",
            files=node.get_all_files(),
        )


@dataclass
class NavigationResult:
    """导航结果"""
    status: str  # 'files', 'navigate', 'multiple_matches', 'need_filter', 'no_match'
    message: str = ""
    current_path: str = ""
    options: List[Tuple[str, str, int]] = field(default_factory=list)
    files: List[Dict] = field(default_factory=list)
