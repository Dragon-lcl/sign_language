#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征提取模块
用于从图像和视频中提取手语特征
"""

import cv2
import numpy as np
import yaml
import os
import json

from paths import resolve_config_path

class FeatureExtractor:
    """
    特征提取类
    用于从图像和视频中提取手语特征
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化特征提取器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取特征提取方法
        self.feature_method = self.config['recognition']['feature_method']
        
        # 加载词汇表（相对路径基于项目根/外部资源目录解析）
        vocab_path = resolve_config_path(self.config['recognition']['vocabulary_path'])
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
        else:
            self.vocabulary = {}
            print(f"警告: 词汇表文件不存在: {vocab_path}")
    
    def extract_keypoints(self, pose_keypoints, hand_keypoints):
        """
        从姿态关键点和手部关键点提取特征
        
        参数:
            pose_keypoints: 姿态关键点，numpy数组
            hand_keypoints: 手部关键点，numpy数组
            
        返回:
            提取的特征向量
        """
        # 合并关键点
        features = []
        
        # 添加姿态关键点（只关注上半身）
        if pose_keypoints is not None and len(pose_keypoints) > 0:
            # 上半身关键点索引（头部、肩膀、手臂等）
            upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7]
            for idx in upper_body_indices:
                if idx < len(pose_keypoints):
                    features.extend(pose_keypoints[idx])
        
        # 添加手部关键点
        if hand_keypoints is not None and len(hand_keypoints) > 0:
            for hand in hand_keypoints:
                features.extend(hand.flatten())
        
        return np.array(features)
    
    def extract_heatmap_features(self, heatmaps):
        """
        从热图中提取特征
        
        参数:
            heatmaps: 热图，numpy数组
            
        返回:
            提取的特征向量
        """
        # 降维并归一化热图
        features = []
        for heatmap in heatmaps:
            # 降采样热图
            resized_heatmap = cv2.resize(heatmap, (32, 32))
            # 展平并添加到特征
            features.extend(resized_heatmap.flatten())
        
        return np.array(features)
    
    def extract_combined_features(self, pose_keypoints, hand_keypoints, heatmaps):
        """
        结合关键点和热图提取特征
        
        参数:
            pose_keypoints: 姿态关键点，numpy数组
            hand_keypoints: 手部关键点，numpy数组
            heatmaps: 热图，numpy数组
            
        返回:
            提取的特征向量
        """
        # 提取关键点特征
        keypoint_features = self.extract_keypoints(pose_keypoints, hand_keypoints)
        
        # 提取热图特征
        heatmap_features = self.extract_heatmap_features(heatmaps)
        
        # 合并特征
        combined_features = np.concatenate([keypoint_features, heatmap_features])
        
        return combined_features
    
    def extract_features(self, pose_keypoints, hand_keypoints, heatmaps=None):
        """
        根据配置的方法提取特征
        
        参数:
            pose_keypoints: 姿态关键点，numpy数组
            hand_keypoints: 手部关键点，numpy数组
            heatmaps: 热图，numpy数组，可选
            
        返回:
            提取的特征向量
        """
        if self.feature_method == "keypoints":
            return self.extract_keypoints(pose_keypoints, hand_keypoints)
        elif self.feature_method == "heatmap":
            if heatmaps is None:
                raise ValueError("热图特征提取方法需要提供热图")
            return self.extract_heatmap_features(heatmaps)
        elif self.feature_method == "combined":
            if heatmaps is None:
                raise ValueError("组合特征提取方法需要提供热图")
            return self.extract_combined_features(pose_keypoints, hand_keypoints, heatmaps)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.feature_method}")
    
    def normalize_features(self, features):
        """
        归一化特征
        
        参数:
            features: 特征向量，numpy数组
            
        返回:
            归一化后的特征向量
        """
        # 简单的最小-最大归一化
        min_val = np.min(features)
        max_val = np.max(features)
        
        if max_val == min_val:
            return np.zeros_like(features)
        
        normalized_features = (features - min_val) / (max_val - min_val)
        
        return normalized_features
    
    def extract_sequence_features(self, pose_keypoints_seq, hand_keypoints_seq, heatmaps_seq=None):
        """
        从序列中提取特征
        
        参数:
            pose_keypoints_seq: 姿态关键点序列，numpy数组
            hand_keypoints_seq: 手部关键点序列，numpy数组
            heatmaps_seq: 热图序列，numpy数组，可选
            
        返回:
            提取的特征序列
        """
        sequence_features = []
        
        for i in range(len(pose_keypoints_seq)):
            pose_keypoints = pose_keypoints_seq[i]
            hand_keypoints = hand_keypoints_seq[i]
            
            heatmaps = None
            if heatmaps_seq is not None:
                heatmaps = heatmaps_seq[i]
            
            # 提取单帧特征
            features = self.extract_features(pose_keypoints, hand_keypoints, heatmaps)
            
            # 归一化特征
            normalized_features = self.normalize_features(features)
            
            sequence_features.append(normalized_features)
        
        return np.array(sequence_features) 