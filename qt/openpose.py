import cv2
import numpy as np

class OpenPose:
    def __init__(self, model_path, proto_path, hand_model_path=None, hand_proto_path=None):
        try:
            # 加载身体姿态估计模型
            self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            if self.net.empty():
                raise Exception("Failed to load body pose model")
                
            # 设置计算后端和目标
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # 初始化手部模型
            self.hand_net = None
            if hand_model_path and hand_proto_path:
                try:
                    self.hand_net = cv2.dnn.readNetFromCaffe(hand_proto_path, hand_model_path)
                    if not self.hand_net.empty():
                        self.hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        self.hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"Warning: Failed to load hand model: {str(e)}")
                    self.hand_net = None
                    
        except Exception as e:
            print(f"Error initializing OpenPose: {str(e)}")
            raise

    def estimate(self, frame):
        if self.net is None:
            return [], []
            
        # 准备输入图像
        frame_height, frame_width = frame.shape[:2]
        in_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368),
                                      (0, 0, 0), swapRB=False, crop=False)
        
        # 设置输入
        self.net.setInput(in_blob)
        
        # 前向传播
        try:
            output = self.net.forward()
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            return [], []
            
        # 处理输出
        keypoints = []
        for i in range(output.shape[1]):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            x = (frame_width * point[0]) / output.shape[3]
            y = (frame_height * point[1]) / output.shape[2]
            
            if prob > 0.1:  # 置信度阈值
                keypoints.append((x, y))
            else:
                keypoints.append(None)
        
        # 手部关键点检测
        hand_keypoints = []
        if self.hand_net is not None:
            try:
                hand_keypoints = self.detect_hands(frame)
            except Exception as e:
                print(f"Error during hand detection: {str(e)}")
                hand_keypoints = []
        
        return keypoints, hand_keypoints

    def detect_hands(self, frame):
        if self.hand_net is None:
            return []
            
        # 准备输入图像
        in_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368),
                                      (0, 0, 0), swapRB=False, crop=False)
        
        # 设置输入
        self.hand_net.setInput(in_blob)
        
        # 前向传播
        try:
            output = self.hand_net.forward()
        except Exception as e:
            print(f"Error during hand detection forward pass: {str(e)}")
            return []
            
        # 处理输出
        hand_keypoints = []
        for i in range(output.shape[1]):
            prob_map = output[0, i, :, :]
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            x = (frame.shape[1] * point[0]) / output.shape[3]
            y = (frame.shape[0] * point[1]) / output.shape[2]
            
            if prob > 0.1:  # 置信度阈值
                hand_keypoints.append((x, y))
            else:
                hand_keypoints.append(None)
                
        return hand_keypoints 