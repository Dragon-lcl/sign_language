#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手语识别器模块
用于整合检测和分类功能，实现手语识别
"""

import os
import yaml
import numpy as np
import cv2
import json
from collections import Counter

from ..detection import YOLODetector, OpenPoseDetector
from ..data_processing import Preprocessor, FeatureExtractor
from .sign_classifier import SignClassifier
from paths import resolve_config_path

class SignRecognizer:
    """
    手语识别器类
    用于整合检测和分类功能，实现手语识别
    """
    
    def __init__(self, config_path='config.yaml', detector_type='yolo'):
        """
        初始化手语识别器
        
        参数:
            config_path: 配置文件路径
            detector_type: 检测器类型，'yolo'或'openpose'
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化预处理器
        self.preprocessor = Preprocessor(config_path)
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor(config_path)
        
        # 初始化分类器
        self.classifier = SignClassifier(config_path)
        
        # 初始化检测器
        self.detector_type = detector_type
        if detector_type == 'yolo':
            self.detector = YOLODetector(config_path)
        elif detector_type == 'openpose':
            self.detector = OpenPoseDetector(config_path)
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")
        
        # 获取序列长度
        self.sequence_length = self.config['recognition']['sequence_length']
        
        # 加载词汇表（相对路径基于项目根/外部资源目录解析）
        vocab_path = resolve_config_path(self.config['recognition']['vocabulary_path'])
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
        else:
            self.vocabulary = {}
            print(f"警告: 词汇表文件不存在: {vocab_path}")
    
    def recognize_image(self, image):
        """
        识别单张图像中的手语
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            
        返回:
            识别结果和概率
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 预处理图像
        processed_image = self.preprocessor.preprocess_image(image)
        
        # 检测关键点
        if self.detector_type == 'yolo':
            # 使用YOLOv8检测
            results = self.detector.detect(processed_image)
            keypoints = self.detector.extract_keypoints(results)
            
            if keypoints is None or len(keypoints) == 0:
                return None, 0.0
            
            # 提取特征
            features = self.feature_extractor.extract_features(keypoints, None)
        else:
            # 使用OpenPose检测
            pose_keypoints, left_hand_keypoints, right_hand_keypoints, heatmaps, keypoint_list = self.detector.detect(processed_image)
            
            if pose_keypoints is None or len(pose_keypoints) == 0:
                return None, 0.0
            
            # 提取特征
            features = self.feature_extractor.extract_features(keypoint_list, [left_hand_keypoints, right_hand_keypoints], heatmaps)
        
        # 归一化特征
        normalized_features = self.feature_extractor.normalize_features(features)
        
        # 预测
        predictions, probabilities = self.classifier.predict(np.array([normalized_features]))
        
        return predictions[0], probabilities[0]
    
    def recognize_video(self, video_path):
        """
        识别视频中的手语
        
        参数:
            video_path: 视频文件路径
            
        返回:
            识别结果序列和概率序列
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 处理视频
        if self.detector_type == 'yolo':
            # 使用YOLOv8处理视频
            keypoints_sequence = self.detector.process_video(video_path)
            
            if keypoints_sequence is None or len(keypoints_sequence) == 0:
                return None, None
            
            # 提取特征序列
            features_sequence = []
            for keypoints in keypoints_sequence:
                if keypoints is not None:
                    features = self.feature_extractor.extract_features(keypoints, None)
                    normalized_features = self.feature_extractor.normalize_features(features)
                    features_sequence.append(normalized_features)
        else:
            # 使用OpenPose处理视频
            pose_keypoints_sequence, left_hand_keypoints_sequence, right_hand_keypoints_sequence, heatmaps_sequence = self.detector.process_video(video_path)
            
            if pose_keypoints_sequence is None or len(pose_keypoints_sequence) == 0:
                return None, None
            
            # 提取特征序列
            features_sequence = []
            for i in range(len(pose_keypoints_sequence)):
                pose_keypoints = pose_keypoints_sequence[i]
                left_hand_keypoints = left_hand_keypoints_sequence[i]
                right_hand_keypoints = right_hand_keypoints_sequence[i]
                heatmaps = heatmaps_sequence[i]
                
                features = self.feature_extractor.extract_features(pose_keypoints, [left_hand_keypoints, right_hand_keypoints], heatmaps)
                normalized_features = self.feature_extractor.normalize_features(features)
                features_sequence.append(normalized_features)
        
        # 预测序列
        predictions, probabilities = self.classifier.predict_sequence(features_sequence)
        
        return predictions, probabilities
    
    def recognize_realtime(self, camera_id=0):
        """
        实时识别摄像头视频中的手语
        
        参数:
            camera_id: 摄像头ID
            
        返回:
            None
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化缓冲区
        feature_buffer = []
        prediction_buffer = []
        
        print("按 'q' 键退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预处理图像
            processed_frame = self.preprocessor.preprocess_image(frame)
            
            # 检测关键点
            if self.detector_type == 'yolo':
                # 使用YOLOv8检测
                results = self.detector.detect(processed_frame)
                keypoints = self.detector.extract_keypoints(results)
                
                # 绘制结果
                result_frame = self.detector.draw_results(frame, results)
                
                if keypoints is not None and len(keypoints) > 0:
                    # 提取特征
                    features = self.feature_extractor.extract_features(keypoints, None)
                    normalized_features = self.feature_extractor.normalize_features(features)
                    
                    # 添加到缓冲区
                    feature_buffer.append(normalized_features)
                    
                    # 保持缓冲区大小
                    if len(feature_buffer) > self.sequence_length:
                        feature_buffer.pop(0)
                    
                    # 如果缓冲区已满，进行预测
                    if len(feature_buffer) == self.sequence_length:
                        predictions, probabilities = self.classifier.predict_sequence(feature_buffer)
                        
                        # 添加到预测缓冲区
                        prediction_buffer.extend(predictions)
                        
                        # 保持预测缓冲区大小
                        if len(prediction_buffer) > 10:
                            prediction_buffer.pop(0)
                        
                        # 获取最常见的预测
                        if len(prediction_buffer) > 0:
                            most_common = Counter(prediction_buffer).most_common(1)[0]
                            prediction = most_common[0]
                            count = most_common[1]
                            confidence = count / len(prediction_buffer)
                            
                            # 显示预测结果
                            cv2.putText(result_frame, f"手语: {prediction} ({confidence:.2f})",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # 使用OpenPose检测
                pose_keypoints, left_hand_keypoints, right_hand_keypoints, heatmaps, keypoint_list = self.detector.detect(processed_frame)
                
                # 绘制结果
                result_frame = self.detector.draw_results(frame, pose_keypoints, left_hand_keypoints, right_hand_keypoints)
                
                if pose_keypoints is not None and len(pose_keypoints) > 0:
                    # 提取特征
                    features = self.feature_extractor.extract_features(keypoint_list, [left_hand_keypoints, right_hand_keypoints], heatmaps)
                    normalized_features = self.feature_extractor.normalize_features(features)
                    
                    # 添加到缓冲区
                    feature_buffer.append(normalized_features)
                    
                    # 保持缓冲区大小
                    if len(feature_buffer) > self.sequence_length:
                        feature_buffer.pop(0)
                    
                    # 如果缓冲区已满，进行预测
                    if len(feature_buffer) == self.sequence_length:
                        predictions, probabilities = self.classifier.predict_sequence(feature_buffer)
                        
                        # 添加到预测缓冲区
                        prediction_buffer.extend(predictions)
                        
                        # 保持预测缓冲区大小
                        if len(prediction_buffer) > 10:
                            prediction_buffer.pop(0)
                        
                        # 获取最常见的预测
                        if len(prediction_buffer) > 0:
                            most_common = Counter(prediction_buffer).most_common(1)[0]
                            prediction = most_common[0]
                            count = most_common[1]
                            confidence = count / len(prediction_buffer)
                            
                            # 显示预测结果
                            cv2.putText(result_frame, f"手语: {prediction} ({confidence:.2f})",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Sign Language Recognition', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
    
    def train(self, data_dir):
        """
        训练手语识别模型
        
        参数:
            data_dir: 数据目录，包含标记好的手语视频
            
        返回:
            训练后的分类器
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 获取所有视频文件
        video_files = []
        labels = []
        
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(label_dir, file))
                        labels.append(label)
        
        if len(video_files) == 0:
            raise ValueError(f"数据目录中没有视频文件: {data_dir}")
        
        print(f"找到 {len(video_files)} 个视频文件，开始处理...")
        
        # 处理视频并提取特征
        features_list = []
        
        for i, video_file in enumerate(video_files):
            print(f"处理视频 {i+1}/{len(video_files)}: {video_file}")
            
            # 处理视频
            if self.detector_type == 'yolo':
                # 使用YOLOv8处理视频
                keypoints_sequence = self.detector.process_video(video_file)
                
                if keypoints_sequence is None or len(keypoints_sequence) == 0:
                    print(f"警告: 视频中没有检测到关键点: {video_file}")
                    continue
                
                # 提取特征序列
                features_sequence = []
                for keypoints in keypoints_sequence:
                    if keypoints is not None:
                        features = self.feature_extractor.extract_features(keypoints, None)
                        normalized_features = self.feature_extractor.normalize_features(features)
                        features_sequence.append(normalized_features)
            else:
                # 使用OpenPose处理视频
                pose_keypoints_sequence, left_hand_keypoints_sequence, right_hand_keypoints_sequence, heatmaps_sequence = self.detector.process_video(video_file)
                
                if pose_keypoints_sequence is None or len(pose_keypoints_sequence) == 0:
                    print(f"警告: 视频中没有检测到关键点: {video_file}")
                    continue
                
                # 提取特征序列
                features_sequence = []
                for i in range(len(pose_keypoints_sequence)):
                    pose_keypoints = pose_keypoints_sequence[i]
                    left_hand_keypoints = left_hand_keypoints_sequence[i]
                    right_hand_keypoints = right_hand_keypoints_sequence[i]
                    heatmaps = heatmaps_sequence[i]
                    
                    features = self.feature_extractor.extract_features(pose_keypoints, [left_hand_keypoints, right_hand_keypoints], heatmaps)
                    normalized_features = self.feature_extractor.normalize_features(features)
                    features_sequence.append(normalized_features)
            
            # 添加到特征列表
            features_list.append(features_sequence)
        
        # 训练分类器
        print("开始训练分类器...")
        self.classifier.train(np.array(features_list), np.array(labels))
        
        print("训练完成！")
        
        return self.classifier 