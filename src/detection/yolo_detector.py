#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8检测器模块
用于手部和姿态检测
"""

import os
import yaml
import numpy as np
import cv2
from ultralytics import YOLO

from paths import resolve_config_path

class YOLODetector:
    """
    YOLOv8检测器类
    用于手部和姿态检测
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化YOLOv8检测器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取模型路径（相对路径基于项目根/外部资源目录解析）
        model_path = resolve_config_path(self.config['models']['yolo']['model_path'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8模型文件不存在: {model_path}")
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 获取置信度阈值和IoU阈值
        self.conf_threshold = self.config['models']['yolo']['confidence_threshold']
        self.iou_threshold = self.config['models']['yolo']['iou_threshold']
    
    def detect(self, image):
        """
        检测图像中的人体姿态
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            
        返回:
            检测结果，包含关键点信息
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 使用YOLOv8进行检测
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        return results[0]  # 返回第一个结果
    
    def extract_keypoints(self, results):
        """
        从检测结果中提取关键点
        
        参数:
            results: YOLOv8检测结果
            
        返回:
            关键点数组
        """
        if results is None or not hasattr(results, 'keypoints') or results.keypoints is None:
            return None
        
        # 获取关键点
        keypoints = results.keypoints.data.cpu().numpy()
        
        return keypoints
    
    def draw_results(self, image, results):
        """
        在图像上绘制检测结果
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            results: YOLOv8检测结果
            
        返回:
            绘制了检测结果的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        if results is None:
            return image.copy()
        
        # 创建图像副本
        result_image = image.copy()
        
        # 绘制检测框和关键点
        annotated_image = results.plot()
        
        return annotated_image
    
    def process_video(self, video_path, output_path=None, show=False):
        """
        处理视频
        
        参数:
            video_path: 视频文件路径
            output_path: 输出视频路径，如果为None则不保存
            show: 是否显示处理过程
            
        返回:
            关键点序列
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        keypoints_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用YOLOv8进行检测
            results = self.detect(frame)
            
            # 提取关键点
            keypoints = self.extract_keypoints(results)
            keypoints_sequence.append(keypoints)
            
            # 绘制结果
            result_frame = self.draw_results(frame, results)
            
            # 保存结果帧
            if output_path is not None:
                out.write(result_frame)
            
            # 显示结果
            if show:
                cv2.imshow('YOLOv8 Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            print(f"\r处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.2f}%)", end="")
        
        print()  # 换行
        
        # 释放资源
        cap.release()
        if output_path is not None:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        return keypoints_sequence
    
    def process_realtime(self, camera_id=0, output_path=None):
        """
        实时处理摄像头视频
        
        参数:
            camera_id: 摄像头ID
            output_path: 输出视频路径，如果为None则不保存
            
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
        fps = self.config['app']['fps']
        
        # 创建视频写入器
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("按 'q' 键退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用YOLOv8进行检测
            results = self.detect(frame)
            
            # 绘制结果
            result_frame = self.draw_results(frame, results)
            
            # 保存结果帧
            if output_path is not None:
                out.write(result_frame)
            
            # 显示结果
            cv2.imshow('YOLOv8 Real-time Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        if output_path is not None:
            out.release()
        cv2.destroyAllWindows() 