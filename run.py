#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能车辆电路图资料导航 - 启动脚本

使用方法:
    python run.py           # 启动 Gradio Web 应用
    python run.py --check   # 仅检查配置和健康状态
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def check_environment() -> bool:
    """检查环境配置"""
    from dotenv import load_dotenv
    load_dotenv()
    
    errors = []
    warnings = []
    
    # 检查必要的环境变量
    zhipu_key = os.getenv("ZHIPU_API_KEY", "")
    if not zhipu_key:
        warnings.append("ZHIPU_API_KEY 未配置，LLM 查询改写功能将不可用")
    
    # 检查数据文件
    data_file = PROJECT_ROOT / "data" / "structured_data_llm.json"
    if not data_file.exists():
        errors.append(f"数据文件不存在: {data_file}")
    
    # 检查必要目录
    for dir_name in ["logs", "cache"]:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"  [创建目录] {dir_name}/")
    
    # 输出结果
    print("\n========== 环境检查 ==========")
    
    if errors:
        print("\n[错误]")
        for err in errors:
            print(f"  ✗ {err}")
    
    if warnings:
        print("\n[警告]")
        for warn in warnings:
            print(f"  ! {warn}")
    
    if not errors and not warnings:
        print("  ✓ 所有检查通过")
    
    print("================================\n")
    
    return len(errors) == 0


def health_check() -> dict:
    """健康检查"""
    status = {
        "status": "healthy",
        "checks": {}
    }
    
    # 检查数据文件
    data_file = PROJECT_ROOT / "data" / "structured_data_llm.json"
    status["checks"]["data_file"] = data_file.exists()
    
    # 检查模块可导入
    try:
        from app.chatbot import NavigationChatbot
        status["checks"]["chatbot_module"] = True
    except Exception as e:
        status["checks"]["chatbot_module"] = False
        status["status"] = "unhealthy"
    
    # 检查 API Key
    from dotenv import load_dotenv
    load_dotenv()
    status["checks"]["llm_configured"] = bool(os.getenv("ZHIPU_API_KEY", ""))
    
    return status


def run_app():
    """启动 Gradio 应用"""
    import subprocess
    app_path = PROJECT_ROOT / "app" / "gradio_ui.py"
    subprocess.run([sys.executable, str(app_path)])


def main():
    parser = argparse.ArgumentParser(description="电路图资料检索系统")
    parser.add_argument("--check", action="store_true", help="仅检查环境配置")
    parser.add_argument("--health", action="store_true", help="输出健康检查 JSON")
    args = parser.parse_args()
    
    if args.health:
        import json
        print(json.dumps(health_check(), ensure_ascii=False, indent=2))
        return
    
    if args.check:
        success = check_environment()
        sys.exit(0 if success else 1)
    
    # 正常启动前先检查
    if not check_environment():
        print("环境检查失败，请修复上述错误后重试。")
        sys.exit(1)
    
    run_app()


if __name__ == "__main__":
    main()
