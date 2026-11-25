#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手语识别系统启动脚本
"""

import os
import argparse
import yaml
from app import app

from paths import app_path, resolve_config_path

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='手语识别系统')
    parser.add_argument('--host', type=str, help='主机地址')
    parser.add_argument('--port', type=int, help='端口号')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    args = parser.parse_args()
    
    # 加载配置（配置文件始终位于项目根目录）
    with open(app_path('config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置参数
    host = args.host if args.host is not None else config['app']['host']
    port = args.port if args.port is not None else config['app']['port']
    debug = args.debug if args.debug is not None else config['app']['debug']
    
    # 创建必要的目录（上传目录固定在项目内，大文件目录支持外置）
    os.makedirs(app_path('data'), exist_ok=True)
    os.makedirs(resolve_config_path('models/yolo'), exist_ok=True)
    os.makedirs(resolve_config_path('models/openpose'), exist_ok=True)
    os.makedirs(app_path('static', 'uploads'), exist_ok=True)
    
    # 检查词汇表是否存在
    vocab_path = resolve_config_path(config['recognition']['vocabulary_path'])
    if not os.path.exists(vocab_path):
        print(f"警告: 词汇表文件不存在: {vocab_path}")
        print("请确保词汇表文件存在，或者使用src/utils/data_utils.py中的create_vocabulary函数创建")
    
    # 检查模型文件是否存在
    yolo_model_path = resolve_config_path(config['models']['yolo']['model_path'])
    if not os.path.exists(yolo_model_path):
        print(f"警告: YOLOv8模型文件不存在: {yolo_model_path}")
        print("请运行download_models.py下载预训练模型")
    
    openpose_model_path = resolve_config_path(config['models']['openpose']['model_path'])
    if not os.path.exists(openpose_model_path):
        print(f"警告: OpenPose模型文件不存在: {openpose_model_path}")
        print("请运行download_models.py下载预训练模型")
    
    # 启动应用
    print(f"启动手语识别系统，访问 http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main() 