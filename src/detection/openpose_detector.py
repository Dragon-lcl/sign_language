#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenPose检测器模块
用于手部和姿态检测
"""

import os
import yaml
import numpy as np
import cv2

from paths import resolve_config_path

class OpenPoseDetector:
    """
    OpenPose检测器类
    用于手部和姿态检测
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化OpenPose检测器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取模型路径（相对路径基于项目根/外部资源目录解析）
        model_path = resolve_config_path(self.config['models']['openpose']['model_path'])
        proto_path = resolve_config_path(self.config['models']['openpose']['proto_path'])
        hand_model_path = resolve_config_path(self.config['models']['openpose']['hand_model_path'])
        hand_proto_path = resolve_config_path(self.config['models']['openpose']['hand_proto_path'])
        
        # 检查文件是否存在
        for path in [model_path, proto_path, hand_model_path, hand_proto_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"OpenPose模型文件不存在: {path}")
        
        # 加载姿态估计模型
        self.pose_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        
        # 加载手部检测模型
        self.hand_net = cv2.dnn.readNetFromCaffe(hand_proto_path, hand_model_path)
        
        # 设置后端和目标
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # 如果有CUDA设备，使用CUDA
            self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # COCO数据集的关键点
        self.pose_pairs = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
            (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
        ]
        
        # 手部关键点对
        self.hand_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
    
    def detect_pose(self, image):
        """
        检测图像中的人体姿态
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            
        返回:
            关键点和热图
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 准备输入
        input_blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False
        )
        
        # 设置输入
        self.pose_net.setInput(input_blob)
        
        # 前向传播
        output = self.pose_net.forward()
        
        # 处理输出
        keypoints = []
        keypoint_list = []
        
        # COCO数据集有18个关键点
        num_keypoints = 18
        
        # 遍历所有关键点
        for i in range(num_keypoints):
            # 获取热图
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (width, height))
            
            # 找到最大值的位置
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            
            # 如果概率大于阈值，添加关键点
            if prob > 0.1:
                keypoints.append((int(point[0]), int(point[1]), prob))
                keypoint_list.append([int(point[0]), int(point[1]), prob])
            else:
                keypoints.append(None)
                keypoint_list.append([0, 0, 0])
        
        return keypoints, output, keypoint_list
    
    def detect_hands(self, image, pose_keypoints):
        """
        检测图像中的手部
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            pose_keypoints: 姿态关键点
            
        返回:
            左手和右手的关键点
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        if pose_keypoints is None or len(pose_keypoints) < 18:
            return None, None
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 获取手腕位置
        left_wrist = pose_keypoints[7]  # 左手腕
        right_wrist = pose_keypoints[4]  # 右手腕
        
        left_hand_keypoints = None
        right_hand_keypoints = None
        
        # 检测左手
        if left_wrist is not None:
            # 提取左手区域
            x, y, _ = left_wrist
            hand_roi = image[max(0, y - 100):min(height, y + 100), max(0, x - 100):min(width, x + 100)]
            
            if hand_roi.size > 0:
                # 准备输入
                input_blob = cv2.dnn.blobFromImage(
                    hand_roi, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False
                )
                
                # 设置输入
                self.hand_net.setInput(input_blob)
                
                # 前向传播
                output = self.hand_net.forward()
                
                # 处理输出
                left_hand_keypoints = []
                
                # 手部有21个关键点
                num_keypoints = 21
                
                # 遍历所有关键点
                for i in range(num_keypoints):
                    # 获取热图
                    prob_map = output[0, i, :, :]
                    prob_map = cv2.resize(prob_map, (hand_roi.shape[1], hand_roi.shape[0]))
                    
                    # 找到最大值的位置
                    _, prob, _, point = cv2.minMaxLoc(prob_map)
                    
                    # 如果概率大于阈值，添加关键点
                    if prob > 0.1:
                        # 转换回原始图像坐标
                        x_coord = point[0] + max(0, x - 100)
                        y_coord = point[1] + max(0, y - 100)
                        left_hand_keypoints.append((x_coord, y_coord, prob))
                    else:
                        left_hand_keypoints.append(None)
        
        # 检测右手
        if right_wrist is not None:
            # 提取右手区域
            x, y, _ = right_wrist
            hand_roi = image[max(0, y - 100):min(height, y + 100), max(0, x - 100):min(width, x + 100)]
            
            if hand_roi.size > 0:
                # 准备输入
                input_blob = cv2.dnn.blobFromImage(
                    hand_roi, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False
                )
                
                # 设置输入
                self.hand_net.setInput(input_blob)
                
                # 前向传播
                output = self.hand_net.forward()
                
                # 处理输出
                right_hand_keypoints = []
                
                # 手部有21个关键点
                num_keypoints = 21
                
                # 遍历所有关键点
                for i in range(num_keypoints):
                    # 获取热图
                    prob_map = output[0, i, :, :]
                    prob_map = cv2.resize(prob_map, (hand_roi.shape[1], hand_roi.shape[0]))
                    
                    # 找到最大值的位置
                    _, prob, _, point = cv2.minMaxLoc(prob_map)
                    
                    # 如果概率大于阈值，添加关键点
                    if prob > 0.1:
                        # 转换回原始图像坐标
                        x_coord = point[0] + max(0, x - 100)
                        y_coord = point[1] + max(0, y - 100)
                        right_hand_keypoints.append((x_coord, y_coord, prob))
                    else:
                        right_hand_keypoints.append(None)
        
        return left_hand_keypoints, right_hand_keypoints
    
    def draw_pose(self, image, keypoints):
        """
        在图像上绘制姿态关键点
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            keypoints: 关键点列表
            
        返回:
            绘制了关键点的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 创建图像副本
        result_image = image.copy()
        
        # 绘制关键点
        for i, keypoint in enumerate(keypoints):
            if keypoint is not None:
                x, y, prob = keypoint
                cv2.circle(result_image, (x, y), 5, (0, 255, 255), -1)
        
        # 绘制骨架
        for pair in self.pose_pairs:
            part_a = pair[0]
            part_b = pair[1]
            
            if keypoints[part_a] is not None and keypoints[part_b] is not None:
                x_a, y_a, _ = keypoints[part_a]
                x_b, y_b, _ = keypoints[part_b]
                cv2.line(result_image, (x_a, y_a), (x_b, y_b), (0, 255, 0), 2)
        
        return result_image
    
    def draw_hands(self, image, left_hand_keypoints, right_hand_keypoints):
        """
        在图像上绘制手部关键点
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            left_hand_keypoints: 左手关键点
            right_hand_keypoints: 右手关键点
            
        返回:
            绘制了手部关键点的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 创建图像副本
        result_image = image.copy()
        
        # 绘制左手关键点
        if left_hand_keypoints is not None:
            for i, keypoint in enumerate(left_hand_keypoints):
                if keypoint is not None:
                    x, y, prob = keypoint
                    cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
            # 绘制左手骨架
            for pair in self.hand_pairs:
                part_a = pair[0]
                part_b = pair[1]
                
                if (part_a < len(left_hand_keypoints) and part_b < len(left_hand_keypoints) and
                    left_hand_keypoints[part_a] is not None and left_hand_keypoints[part_b] is not None):
                    x_a, y_a, _ = left_hand_keypoints[part_a]
                    x_b, y_b, _ = left_hand_keypoints[part_b]
                    cv2.line(result_image, (x_a, y_a), (x_b, y_b), (0, 0, 255), 2)
        
        # 绘制右手关键点
        if right_hand_keypoints is not None:
            for i, keypoint in enumerate(right_hand_keypoints):
                if keypoint is not None:
                    x, y, prob = keypoint
                    cv2.circle(result_image, (x, y), 3, (255, 0, 0), -1)
            
            # 绘制右手骨架
            for pair in self.hand_pairs:
                part_a = pair[0]
                part_b = pair[1]
                
                if (part_a < len(right_hand_keypoints) and part_b < len(right_hand_keypoints) and
                    right_hand_keypoints[part_a] is not None and right_hand_keypoints[part_b] is not None):
                    x_a, y_a, _ = right_hand_keypoints[part_a]
                    x_b, y_b, _ = right_hand_keypoints[part_b]
                    cv2.line(result_image, (x_a, y_a), (x_b, y_b), (255, 0, 0), 2)
        
        return result_image
    
    def detect(self, image):
        """
        检测图像中的人体姿态和手部
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            
        返回:
            姿态关键点、左手关键点、右手关键点和热图
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 检测姿态
        pose_keypoints, heatmaps, keypoint_list = self.detect_pose(image)
        
        # 检测手部
        left_hand_keypoints, right_hand_keypoints = self.detect_hands(image, pose_keypoints)
        
        return pose_keypoints, left_hand_keypoints, right_hand_keypoints, heatmaps, keypoint_list
    
    def draw_results(self, image, pose_keypoints, left_hand_keypoints, right_hand_keypoints):
        """
        在图像上绘制检测结果
        
        参数:
            image: 输入图像，numpy数组，BGR格式
            pose_keypoints: 姿态关键点
            left_hand_keypoints: 左手关键点
            right_hand_keypoints: 右手关键点
            
        返回:
            绘制了检测结果的图像
        """
        if image is None:
            raise ValueError("输入图像为空")
        
        # 绘制姿态
        result_image = self.draw_pose(image, pose_keypoints)
        
        # 绘制手部
        result_image = self.draw_hands(result_image, left_hand_keypoints, right_hand_keypoints)
        
        return result_image
    
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
        
        pose_keypoints_sequence = []
        left_hand_keypoints_sequence = []
        right_hand_keypoints_sequence = []
        heatmaps_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测姿态和手部
            pose_keypoints, left_hand_keypoints, right_hand_keypoints, heatmaps, keypoint_list = self.detect(frame)
            
            # 保存关键点
            pose_keypoints_sequence.append(keypoint_list)
            left_hand_keypoints_sequence.append(left_hand_keypoints)
            right_hand_keypoints_sequence.append(right_hand_keypoints)
            heatmaps_sequence.append(heatmaps)
            
            # 绘制结果
            result_frame = self.draw_results(frame, pose_keypoints, left_hand_keypoints, right_hand_keypoints)
            
            # 保存结果帧
            if output_path is not None:
                out.write(result_frame)
            
            # 显示结果
            if show:
                cv2.imshow('OpenPose Detection', result_frame)
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
        
        return pose_keypoints_sequence, left_hand_keypoints_sequence, right_hand_keypoints_sequence, heatmaps_sequence
    
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
            
            # 检测姿态和手部
            pose_keypoints, left_hand_keypoints, right_hand_keypoints, _, _ = self.detect(frame)
            
            # 绘制结果
            result_frame = self.draw_results(frame, pose_keypoints, left_hand_keypoints, right_hand_keypoints)
            
            # 保存结果帧
            if output_path is not None:
                out.write(result_frame)
            
            # 显示结果
            cv2.imshow('OpenPose Real-time Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        if output_path is not None:
            out.release()
        cv2.destroyAllWindows() 