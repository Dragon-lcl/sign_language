#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理工具模块
用于数据处理和管理
"""

import os
import json
import numpy as np
import cv2
from collections import defaultdict

from paths import resolve_config_path

def create_vocabulary(data_dir, output_path='data/vocabulary.json'):
    """
    创建手语词汇表
    
    参数:
        data_dir: 数据目录，包含标记好的手语视频
        output_path: 输出路径
        
    返回:
        词汇表字典
    """
    data_dir = resolve_config_path(data_dir)
    output_path = resolve_config_path(output_path)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取所有标签
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            labels.append(label)
    
    # 创建词汇表
    vocabulary = {
        'labels': labels,
        'label_to_id': {label: i for i, label in enumerate(labels)},
        'id_to_label': {str(i): label for i, label in enumerate(labels)}
    }
    
    # 保存词汇表
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)
    
    print(f"词汇表已保存到: {output_path}")
    print(f"共有 {len(labels)} 个手语标签")
    
    return vocabulary

def load_vocabulary(vocab_path='data/vocabulary.json'):
    """
    加载手语词汇表
    
    参数:
        vocab_path: 词汇表路径
        
    返回:
        词汇表字典
    """
    vocab_path = resolve_config_path(vocab_path)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    
    return vocabulary

def extract_frames_from_video(video_path, output_dir=None, frame_interval=1):
    """
    从视频中提取帧
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录，如果为None则不保存
        frame_interval: 帧间隔，每隔多少帧提取一帧
        
    返回:
        提取的帧列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 创建输出目录
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔提取帧
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
            # 保存帧
            if output_dir is not None:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    
    return frames

def prepare_dataset(data_dir, output_dir='data/processed', frame_interval=1):
    """
    准备数据集
    
    参数:
        data_dir: 数据目录，包含标记好的手语视频
        output_dir: 输出目录
        frame_interval: 帧间隔，每隔多少帧提取一帧
        
    返回:
        处理后的数据集信息
    """
    data_dir = resolve_config_path(data_dir)
    output_dir = resolve_config_path(output_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 创建标签输出目录
            label_output_dir = os.path.join(output_dir, label)
            os.makedirs(label_output_dir, exist_ok=True)
            
            for file in os.listdir(label_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(label_dir, file)
                    video_files.append(video_path)
                    labels.append(label)
                    
                    # 提取帧
                    video_name = os.path.splitext(file)[0]
                    video_output_dir = os.path.join(label_output_dir, video_name)
                    os.makedirs(video_output_dir, exist_ok=True)
                    
                    extract_frames_from_video(video_path, video_output_dir, frame_interval)
    
    # 创建数据集信息
    dataset_info = {
        'num_videos': len(video_files),
        'num_labels': len(set(labels)),
        'labels': list(set(labels)),
        'videos': [{'path': path, 'label': label} for path, label in zip(video_files, labels)]
    }
    
    # 保存数据集信息
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)
    
    print(f"数据集已处理并保存到: {output_dir}")
    print(f"共有 {len(video_files)} 个视频，{len(set(labels))} 个标签")
    
    return dataset_info

def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分数据集为训练集、验证集和测试集
    
    参数:
        data_dir: 数据目录，包含标记好的手语视频
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    返回:
        划分后的数据集信息
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 检查比例和
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集和测试集比例之和必须为1")
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 获取所有视频文件
    video_files = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(label_dir, file)
                    video_files.append(video_path)
                    labels.append(label)
    
    # 按标签分组
    label_to_videos = defaultdict(list)
    for video, label in zip(video_files, labels):
        label_to_videos[label].append(video)
    
    # 划分数据集
    train_videos = []
    val_videos = []
    test_videos = []
    train_labels = []
    val_labels = []
    test_labels = []
    
    for label, videos in label_to_videos.items():
        # 打乱视频顺序
        np.random.shuffle(videos)
        
        # 计算划分点
        n_videos = len(videos)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        
        # 划分
        train_videos.extend(videos[:n_train])
        train_labels.extend([label] * n_train)
        
        val_videos.extend(videos[n_train:n_train+n_val])
        val_labels.extend([label] * n_val)
        
        test_videos.extend(videos[n_train+n_val:])
        test_labels.extend([label] * (n_videos - n_train - n_val))
    
    # 创建数据集信息
    dataset_split = {
        'train': [{'path': path, 'label': label} for path, label in zip(train_videos, train_labels)],
        'val': [{'path': path, 'label': label} for path, label in zip(val_videos, val_labels)],
        'test': [{'path': path, 'label': label} for path, label in zip(test_videos, test_labels)],
        'num_train': len(train_videos),
        'num_val': len(val_videos),
        'num_test': len(test_videos),
        'num_labels': len(label_to_videos)
    }
    
    # 保存数据集划分信息
    with open(os.path.join(data_dir, 'dataset_split.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_split, f, ensure_ascii=False, indent=4)
    
    print(f"数据集划分已保存到: {os.path.join(data_dir, 'dataset_split.json')}")
    print(f"训练集: {len(train_videos)} 个视频")
    print(f"验证集: {len(val_videos)} 个视频")
    print(f"测试集: {len(test_videos)} 个视频")
    
    return dataset_split 