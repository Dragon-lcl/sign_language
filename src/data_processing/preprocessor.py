#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理模块
用于图像和视频的预处理，包括调整大小、归一化等
"""

import cv2
import numpy as np
import yaml
import os

class Preprocessor:
    """
    数据预处理类
    用于图像和视频的预处理
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化预处理器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取图像大小配置
        self.image_size = tuple(self.config['data']['image_size'])
        self.normalize = self.config['data']['normalize']
    
    def preprocess_image(self, image):
        """
        预处理单张图像
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            
        返回:
            预处理后的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 调整图像大小
        resized_image = cv2.resize(image, self.image_size)
        
        # 归一化
        if self.normalize:
            resized_image = resized_image.astype(np.float32) / 255.0
        
        return resized_image
    
    def preprocess_video_frame(self, frame):
        """
        预处理视频帧
        
        参数:
            frame: 视频帧，numpy数组，BGR格式
            
        返回:
            预处理后的视频帧
        """
        return self.preprocess_image(frame)
    
    def preprocess_video(self, video_path):
        """
        预处理视频，生成帧序列
        
        参数:
            video_path: 视频文件路径
            
        返回:
            预处理后的视频帧序列
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预处理帧
            processed_frame = self.preprocess_video_frame(frame)
            frames.append(processed_frame)
        
        cap.release()
        
        return np.array(frames)
    
    def extract_frames(self, video_path, output_dir=None, frame_interval=1):
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
                # 预处理帧
                processed_frame = self.preprocess_video_frame(frame)
                frames.append(processed_frame)
                
                # 保存帧
                if output_dir is not None:
                    frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()
        
        return np.array(frames) 