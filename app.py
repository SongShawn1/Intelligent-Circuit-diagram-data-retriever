#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces 入口文件
"""

import os
import sys
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 从环境变量加载配置（HF Spaces 通过 Secrets 设置）
# 如果没有设置，使用空值（某些功能可能受限）

# 导入并启动应用
from app.gradio_ui import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
