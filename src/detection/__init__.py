"""
检测模块
用于手部和姿态检测
"""

from .yolo_detector import YOLODetector
from .openpose_detector import OpenPoseDetector

__all__ = ['YOLODetector', 'OpenPoseDetector'] 