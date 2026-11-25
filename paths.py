#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一路径解析工具。

默认从项目根目录读取 models/、data/ 等资源。
如果把大模型和数据集移到项目外，可设置环境变量 SIGN_LANGUAGE_ASSETS_DIR
指向外部资源目录（例如 E:/sign_language_assets），代码会优先在那里查找；
找不到时再回退到项目内的同名路径。
"""

import os
from pathlib import Path

# 项目根目录（代码所在位置，始终随仓库走）
APP_ROOT = Path(__file__).resolve().parent

# 外部资源目录，默认就是项目根目录
ASSETS_DIR = Path(os.environ.get("SIGN_LANGUAGE_ASSETS_DIR", str(APP_ROOT))).expanduser()


def app_path(*parts):
    """返回项目内的文件路径（配置、模板、静态资源等始终随仓库走）。"""
    return str(APP_ROOT.joinpath(*parts))


def resolve(*parts):
    """
    返回资源路径：外部资源目录优先，项目内路径兜底。

    这样即使大模型/数据集被移出仓库，只要设置 SIGN_LANGUAGE_ASSETS_DIR，
    代码仍能找到它们；不设置时则按项目内默认目录查找。
    """
    external = ASSETS_DIR.joinpath(*parts)
    if external.exists():
        return str(external)
    return str(APP_ROOT.joinpath(*parts))


def resolve_config_path(path):
    """解析配置文件里的路径：绝对路径原样返回，相对路径按 resolve() 规则处理。"""
    if not path:
        return path
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return resolve(*candidate.parts)


def find_chinese_font():
    """跨平台查找可用的中文字体，找不到返回 None。"""
    candidates = [
        os.environ.get("SIGN_LANGUAGE_FONT", ""),
        app_path("fonts", "simhei.ttf"),
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None
