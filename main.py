#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces 入口文件
"""

import sys
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入并启动应用
from app.gradio_ui import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,  # 禁用 API 文档以避免 schema 生成问题
        ssr_mode=False   # 禁用 SSR
    )
