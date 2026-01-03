#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio UI v2 - 简洁版电路图资料导航
"""

import sys
import logging
from pathlib import Path
import gradio as gr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.chatbot import NavigationChatbot, NavContext, ConversationState


# 全局 Chatbot - 启动时预加载
logger.info("🚀 正在初始化 Chatbot 和模型...")
chatbot = NavigationChatbot(debug=False)
logger.info("✅ 初始化完成，服务就绪！")

def get_chatbot():
    return chatbot


def format_results(results: list) -> str:
    """格式化搜索结果 - 清晰显示ID和目录路径"""
    if not results:
        return ""
    
    lines = ["\n---", f"**找到 {len(results)} 个匹配结果：**\n"]
    
    for i, f in enumerate(results, 1):
        file_id = f.get('id', '?')
        filename = f.get('filename', '未知')
        path = f.get('path', '')
        
        # 显示标题行：序号、文件名、ID
        lines.append(f"**{i}. {filename}**  `ID: {file_id}`")
        
        # 显示目录路径
        if path:
            # 将 "电路图->ECU电路图->工程机械->三一" 格式化为更易读的形式
            path_display = path.replace('->', ' / ')
            lines.append(f"   {path_display}")
        
        lines.append("")
    
    return '\n'.join(lines)


def save_context(ctx: NavContext) -> str:
    import json
    return json.dumps({
        'current_path': ctx.current_path,
        'pending_options': ctx.pending_options,
        'pending_files': ctx.pending_files,
        'pending_filter_options': ctx.pending_filter_options if hasattr(ctx, 'pending_filter_options') else [],
        'state': ctx.state.value,
        'history': ctx.history,  # 现在是 list of dict
        'original_query': ctx.original_query,
    }, ensure_ascii=False)


def restore_context(json_str: str) -> NavContext:
    import json
    if not json_str:
        return NavContext()
    try:
        data = json.loads(json_str)
        ctx = NavContext()
        ctx.current_path = data.get('current_path', '')
        ctx.pending_options = [tuple(opt) for opt in data.get('pending_options', [])]
        ctx.pending_files = data.get('pending_files', [])
        ctx.pending_filter_options = [tuple(opt) for opt in data.get('pending_filter_options', [])]
        ctx.state = ConversationState(data.get('state', 1))
        # history 现在是 list of dict，每个 dict 包含 path, options, state
        history_data = data.get('history', [])
        ctx.history = []
        for item in history_data:
            if isinstance(item, dict):
                # 新格式：保持原样
                ctx.history.append(item)
            else:
                # 旧格式（兼容）：只有路径字符串
                ctx.history.append({'path': item, 'options': [], 'state': 1})
        ctx.original_query = data.get('original_query', '')
        return ctx
    except:
        return NavContext()


# ============ 输入校验配置 ============
MAX_INPUT_LENGTH = 200  # 最大输入长度
MIN_INPUT_LENGTH = 1    # 最小输入长度


def validate_input(user_msg: str) -> tuple[bool, str]:
    """
    校验用户输入
    
    Returns:
        (is_valid, error_message)
    """
    if not user_msg or not user_msg.strip():
        return False, ""
    
    msg = user_msg.strip()
    
    # 长度校验
    if len(msg) > MAX_INPUT_LENGTH:
        return False, f"输入内容过长，请限制在 {MAX_INPUT_LENGTH} 字符以内。"
    
    if len(msg) < MIN_INPUT_LENGTH:
        return False, "请输入查询内容。"
    
    return True, ""


def process_input(user_msg: str, history: list, ctx_json: str, display_msg: str = None):
    """
    处理用户输入（使用生成器实现加载状态）
    
    Args:
        user_msg: 发送给 chatbot 的消息（可能是内部命令）
        history: 对话历史
        ctx_json: 上下文 JSON
        display_msg: 在对话框中显示的用户消息（如果为 None，使用 user_msg）
    
    Yields:
        (history, ctx_json, options_update, back_update)
    """
    # 输入校验（内部命令除外）
    if not user_msg.startswith("__"):
        is_valid, error_msg = validate_input(user_msg)
        if not is_valid:
            if error_msg:
                history = history or []
                history.append({"role": "assistant", "content": error_msg})
                yield history, ctx_json, gr.update(visible=False, choices=[]), gr.update(visible=False)
            else:
                yield history, ctx_json, gr.update(visible=False, choices=[]), gr.update(visible=False)
            return
    
    if not user_msg.strip():
        yield history, ctx_json, gr.update(visible=False, choices=[]), gr.update(visible=False)
        return
    
    # 更新历史（Gradio 6.0 使用 dict 格式）
    history = history or []
    
    # 添加用户消息（使用显示消息或原始消息）
    show_msg = display_msg if display_msg else user_msg
    if show_msg and not show_msg.startswith("__"):
        history.append({"role": "user", "content": show_msg})
    
    # 显示加载状态
    history.append({"role": "assistant", "content": "正在搜索中，请稍候..."})
    yield history, ctx_json, gr.update(visible=False, choices=[]), gr.update(visible=False)
    
    # 错误处理
    try:
        bot = get_chatbot()
        ctx = restore_context(ctx_json)
        
        # 调用 chatbot
        response = bot.chat(user_msg, ctx)
        
        # 移除加载提示
        history = history[:-1]
        
        # 构建回复
        reply = response.message
        if response.results:
            reply += format_results(response.results)
        
        # 添加助手回复
        history.append({"role": "assistant", "content": reply})
        
        # 生成选项（导航选项或筛选选项）
        if response.options:
            # 导航选项
            choices = [f"{chr(ord('A')+i)}. {name} ({count}个)" 
                       for i, (path, name, count) in enumerate(response.options[:10])]
            options_update = gr.update(visible=True, choices=choices, value=None, label="请选择目录")
        elif response.filter_options:
            # 筛选选项
            choices = [f"{chr(ord('A')+i)}. {keyword} ({count}个)" 
                       for i, (keyword, count) in enumerate(response.filter_options)]
            options_update = gr.update(visible=True, choices=choices, value=None, label="请选择筛选条件")
        else:
            options_update = gr.update(visible=False, choices=[])
        
        # 返回按钮
        back_update = gr.update(visible=response.can_go_back)
        
        yield history, save_context(ctx), options_update, back_update
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        # 移除加载提示
        history = history[:-1]
        history.append({"role": "assistant", "content": "抱歉，处理请求时发生错误，请稍后重试。"})
        yield history, ctx_json, gr.update(visible=False, choices=[]), gr.update(visible=False)


def handle_option_select(option_value: str, history: list, ctx_json: str):
    """处理选项选择（导航选项或筛选选项）"""
    if not option_value:
        yield history, ctx_json, gr.update(), gr.update()
        return
    
    ctx = restore_context(ctx_json)
    
    # 提取选项字母和内容 (e.g., "A. 天龙*系列 (112个)" -> letter="A", content="天龙*系列")
    letter = option_value[0] if option_value else ""
    
    # 显示用户选择的完整选项文本
    display_msg = f"选择：{option_value}"
    
    # 根据当前状态决定如何处理
    if ctx.state == ConversationState.AWAITING_FILTER:
        # 筛选状态：提取关键词并作为筛选输入
        # 解析选项索引
        try:
            idx = ord(letter.upper()) - ord('A')
        except:
            idx = 0
        
        if idx < len(ctx.pending_filter_options):
            keyword, count = ctx.pending_filter_options[idx]
            # 检查是否是"全部"选项
            if "全部" in keyword:
                yield from process_input("全部", history, ctx_json, display_msg=display_msg)
            else:
                # 用关键词进行筛选
                yield from process_input(keyword, history, ctx_json, display_msg=display_msg)
        else:
            yield from process_input(f"__SELECT__:{letter}", history, ctx_json, display_msg=display_msg)
    else:
        # 导航状态
        yield from process_input(f"__SELECT__:{letter}", history, ctx_json, display_msg=display_msg)


def handle_back(history: list, ctx_json: str):
    """处理返回"""
    # 显示返回操作
    yield from process_input("__BACK__", history, ctx_json, display_msg="⬅️ 返回上一级")


def handle_reset():
    """重置对话"""
    return [], save_context(NavContext()), gr.update(visible=False, choices=[]), gr.update(visible=False)


def show_loading(history: list):
    """显示加载状态"""
    history = history or []
    history.append({"role": "assistant", "content": "⏳ 正在搜索中..."})
    return history


def remove_loading(history: list):
    """移除加载状态（如果最后一条是加载提示）"""
    if history and history[-1].get("content", "").startswith("⏳"):
        return history[:-1]
    return history


# 自定义 CSS - 简洁风格
CUSTOM_CSS = """
    /* 隐藏页脚 */
    footer { display: none !important; }
    
    /* 标题样式 */
    .main-title {
        text-align: center;
        margin-bottom: 0.3rem !important;
        font-weight: 500 !important;
    }
    .sub-title {
        text-align: center;
        color: #666 !important;
        font-size: 0.9rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* 聊天区域 */
    .chat-container {
        border-radius: 8px !important;
    }
    
    /* 选项样式 */
    .options-section label {
        font-size: 0.95rem !important;
        padding: 6px 12px !important;
    }
    
    /* 输入框容器 - 模拟输入框内嵌按钮 */
    .input-wrapper {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        background: var(--input-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: 8px !important;
        padding-right: 4px !important;
    }
    .input-wrapper:focus-within {
        border-color: var(--color-accent) !important;
        box-shadow: 0 0 0 2px var(--color-accent-soft) !important;
    }
    .input-wrapper textarea {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .input-wrapper textarea:focus {
        box-shadow: none !important;
    }
    
    /* 发送按钮 - 正圆形图标按钮，无背景 */
    .send-btn {
        min-width: 32px !important;
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.2rem !important;
        flex-shrink: 0 !important;
        background: transparent !important;
        border: none !important;
        color: #666 !important;
        cursor: pointer !important;
        box-shadow: none !important;
    }
    .send-btn:hover {
        background: rgba(0, 0, 0, 0.05) !important;
        color: #333 !important;
    }
    
    /* 按钮样式 */
    .action-buttons button {
        border-radius: 6px !important;
        font-size: 0.9rem !important;
    }
    
    /* 加载动画 */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading-msg { animation: pulse 1.5s infinite; }
"""

# 创建 UI - 简洁风格
with gr.Blocks(title="电路图资料检索", analytics_enabled=False) as demo:
    gr.Markdown("# 电路图资料检索", elem_classes="main-title")
    gr.Markdown("输入关键词搜索，如：东风天龙整车电路图、博世EDC17、解放J6P", elem_classes="sub-title")
    
    # 状态
    ctx_state = gr.State(save_context(NavContext()))
    
    # 主体区域
    with gr.Column():
        # 聊天区域
        chat = gr.Chatbot(
            label="",
            height=400,
            show_label=False,
            elem_classes="chat-container",
            type="messages",  # 使用 OpenAI 风格的 messages 格式
        )
        
        # 选项区域
        options_radio = gr.Radio(
            choices=[],
            label="请选择",
            visible=False,
            elem_classes="options-section",
        )
        
        # 输入区域 - 输入框内嵌发送按钮
        with gr.Group(elem_classes="input-wrapper"):
            with gr.Row():
                txt_input = gr.Textbox(
                    label="",
                    placeholder="描述您需要查找的电路图资料...",
                    scale=10,
                    show_label=False,
                    container=False,
                    lines=1,
                )
                send_btn = gr.Button("➤", variant="primary", elem_classes="send-btn", scale=0)
        
        # 功能按钮
        with gr.Row(elem_classes="action-buttons"):
            back_btn = gr.Button("返回上一级", visible=False)
            reset_btn = gr.Button("新对话")
    
    # 事件绑定 - 全部禁用 API 暴露以避免 schema 生成问题
    send_btn.click(
        process_input,
        inputs=[txt_input, chat, ctx_state],
        outputs=[chat, ctx_state, options_radio, back_btn],
        api_name=False,
    ).then(lambda: "", outputs=txt_input, api_name=False)
    
    txt_input.submit(
        process_input,
        inputs=[txt_input, chat, ctx_state],
        outputs=[chat, ctx_state, options_radio, back_btn],
        api_name=False,
    ).then(lambda: "", outputs=txt_input, api_name=False)
    
    options_radio.change(
        handle_option_select,
        inputs=[options_radio, chat, ctx_state],
        outputs=[chat, ctx_state, options_radio, back_btn],
        api_name=False,
    )
    
    back_btn.click(
        handle_back,
        inputs=[chat, ctx_state],
        outputs=[chat, ctx_state, options_radio, back_btn],
        api_name=False,
    )
    
    reset_btn.click(
        handle_reset,
        outputs=[chat, ctx_state, options_radio, back_btn],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
