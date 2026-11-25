#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化工具模块
用于可视化关键点和预测结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def visualize_keypoints(image, keypoints, connections=None, colors=None):
    """
    可视化关键点
    
    参数:
        image: 输入图像，numpy数组，BGR格式
        keypoints: 关键点列表
        connections: 关键点连接关系
        colors: 颜色列表
        
    返回:
        可视化后的图像
    """
    if image is None:
        raise ValueError("输入图像为空")
    
    if keypoints is None or len(keypoints) == 0:
        return image.copy()
    
    # 创建图像副本
    result_image = image.copy()
    
    # 默认颜色
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # 绘制关键点
    for i, keypoint in enumerate(keypoints):
        if keypoint is not None:
            if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 2:
                x, y = int(keypoint[0]), int(keypoint[1])
                color = colors[i % len(colors)]
                cv2.circle(result_image, (x, y), 5, color, -1)
                cv2.putText(result_image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 绘制连接
    if connections is not None:
        for connection in connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    keypoints[idx1] is not None and keypoints[idx2] is not None):
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    color = colors[idx1 % len(colors)]
                    cv2.line(result_image, pt1, pt2, color, 2)
    
    return result_image

def visualize_prediction(image, prediction, probability, position=(10, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    """
    可视化预测结果
    
    参数:
        image: 输入图像，numpy数组，BGR格式
        prediction: 预测结果
        probability: 预测概率
        position: 文本位置
        font_scale: 字体大小
        color: 文本颜色
        thickness: 文本粗细
        
    返回:
        可视化后的图像
    """
    if image is None:
        raise ValueError("输入图像为空")
    
    # 创建图像副本
    result_image = image.copy()
    
    # 绘制预测结果
    text = f"手语: {prediction} ({probability:.2f})"
    cv2.putText(result_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    return result_image

def plot_keypoints_3d(keypoints, figsize=(10, 8), connections=None, colors=None):
    """
    绘制3D关键点图
    
    参数:
        keypoints: 关键点列表，每个关键点包含(x, y, z)坐标
        figsize: 图像大小
        connections: 关键点连接关系
        colors: 颜色列表
        
    返回:
        matplotlib图像
    """
    if keypoints is None or len(keypoints) == 0:
        raise ValueError("关键点为空")
    
    # 创建3D图像
    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    
    # 默认颜色
    if colors is None:
        colors = ['g', 'r', 'b', 'y', 'c', 'm']
    
    # 提取坐标
    xs = [kp[0] for kp in keypoints if kp is not None]
    ys = [kp[1] for kp in keypoints if kp is not None]
    zs = [kp[2] for kp in keypoints if kp is not None]
    
    # 绘制关键点
    ax.scatter(xs, ys, zs, c='r', marker='o')
    
    # 绘制连接
    if connections is not None:
        for connection in connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    keypoints[idx1] is not None and keypoints[idx2] is not None):
                    x = [keypoints[idx1][0], keypoints[idx2][0]]
                    y = [keypoints[idx1][1], keypoints[idx2][1]]
                    z = [keypoints[idx1][2], keypoints[idx2][2]]
                    color = colors[idx1 % len(colors)]
                    ax.plot(x, y, z, color)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Keypoints')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    # 转换为图像
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    
    return image

def plot_prediction_history(predictions, probabilities, figsize=(12, 6)):
    """
    绘制预测历史
    
    参数:
        predictions: 预测结果列表
        probabilities: 预测概率列表
        figsize: 图像大小
        
    返回:
        matplotlib图像
    """
    if len(predictions) == 0 or len(probabilities) == 0:
        raise ValueError("预测结果或概率为空")
    
    if len(predictions) != len(probabilities):
        raise ValueError("预测结果和概率长度不匹配")
    
    # 创建图像
    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # 获取唯一的预测结果
    unique_predictions = list(set(predictions))
    
    # 为每个预测结果创建一个颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_predictions)))
    color_map = {pred: colors[i] for i, pred in enumerate(unique_predictions)}
    
    # 绘制预测概率
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        ax.bar(i, prob, color=color_map[pred], alpha=0.7)
        ax.text(i, prob + 0.02, pred, ha='center', rotation=90, fontsize=8)
    
    # 设置坐标轴
    ax.set_xlabel('Frame')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction History')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(predictions)))
    
    # 添加图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[pred]) for pred in unique_predictions]
    ax.legend(handles, unique_predictions, loc='upper right')
    
    # 转换为图像
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    
    return image 