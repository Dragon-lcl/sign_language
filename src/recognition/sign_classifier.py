#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手语分类器模块
用于对提取的特征进行分类
"""

import os
import yaml
import numpy as np
import json
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from paths import resolve_config_path

class SignClassifier:
    """
    手语分类器类
    用于对提取的特征进行分类
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        初始化手语分类器
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 获取分类器类型
        self.classifier_type = self.config['recognition']['classifier']
        
        # 加载词汇表（相对路径基于项目根/外部资源目录解析）
        vocab_path = resolve_config_path(self.config['recognition']['vocabulary_path'])
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocabulary = json.load(f)
        else:
            self.vocabulary = {}
            print(f"警告: 词汇表文件不存在: {vocab_path}")
        
        # 初始化分类器
        self.classifier = None
        self.scaler = StandardScaler()
        
        # 创建模型目录（与外部资源目录保持一致）
        os.makedirs(resolve_config_path('models'), exist_ok=True)
    
    def create_svm_classifier(self):
        """
        创建SVM分类器
        
        返回:
            SVM分类器
        """
        return SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    
    def create_random_forest_classifier(self):
        """
        创建随机森林分类器
        
        返回:
            随机森林分类器
        """
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    def create_mlp_classifier(self):
        """
        创建多层感知机分类器
        
        返回:
            多层感知机分类器
        """
        return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.0001,
                            solver='adam', random_state=42, early_stopping=True)
    
    def create_lstm_model(self, input_shape, num_classes):
        """
        创建LSTM模型
        
        参数:
            input_shape: 输入形状
            num_classes: 类别数
            
        返回:
            LSTM模型
        """
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
        
        return model
    
    def train(self, features, labels):
        """
        训练分类器
        
        参数:
            features: 特征数组
            labels: 标签数组
            
        返回:
            训练后的分类器
        """
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("特征或标签为空")
        
        if len(features) != len(labels):
            raise ValueError("特征和标签长度不匹配")
        
        # 标签编码
        label_to_id = {label: i for i, label in enumerate(set(labels))}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        # 保存标签映射
        with open(resolve_config_path('models/label_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, ensure_ascii=False, indent=4)
        
        # 转换标签为数字
        numeric_labels = np.array([label_to_id[label] for label in labels])
        
        if self.classifier_type == 'lstm':
            # 对于LSTM，需要序列数据
            if len(features.shape) < 3:
                raise ValueError("LSTM需要序列数据，形状应为(样本数, 时间步, 特征数)")
            
            # 创建LSTM模型
            num_classes = len(label_to_id)
            input_shape = (features.shape[1], features.shape[2])
            self.classifier = self.create_lstm_model(input_shape, num_classes)
            
            # 转换标签为one-hot编码
            one_hot_labels = tf.keras.utils.to_categorical(numeric_labels, num_classes=num_classes)
            
            # 训练模型
            self.classifier.fit(features, one_hot_labels, epochs=50, batch_size=32, validation_split=0.2)
            
            # 保存模型
            self.classifier.save(resolve_config_path('models/lstm_model.h5'))
        else:
            # 对于其他分类器，需要展平特征
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # 标准化特征
            features = self.scaler.fit_transform(features)
            
            # 创建分类器
            if self.classifier_type == 'svm':
                self.classifier = self.create_svm_classifier()
            elif self.classifier_type == 'random_forest':
                self.classifier = self.create_random_forest_classifier()
            elif self.classifier_type == 'mlp':
                self.classifier = self.create_mlp_classifier()
            else:
                raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
            
            # 训练分类器
            self.classifier.fit(features, numeric_labels)
            
            # 保存分类器
            with open(resolve_config_path(f'models/{self.classifier_type}_classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier, f)
            
            # 保存标准化器
            with open(resolve_config_path('models/scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        return self.classifier
    
    def load_model(self):
        """
        加载预训练模型
        
        返回:
            加载的模型
        """
        # 加载标签映射
        mapping_path = resolve_config_path('models/label_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                self.label_to_id = mapping['label_to_id']
                self.id_to_label = mapping['id_to_label']
        else:
            print(f"警告: 标签映射文件不存在: {mapping_path}")
            return None
        
        if self.classifier_type == 'lstm':
            # 加载LSTM模型
            model_path = resolve_config_path('models/lstm_model.h5')
            if os.path.exists(model_path):
                self.classifier = load_model(model_path)
            else:
                print(f"警告: LSTM模型文件不存在: {model_path}")
                return None
        else:
            # 加载其他分类器
            classifier_path = resolve_config_path(f'models/{self.classifier_type}_classifier.pkl')
            scaler_path = resolve_config_path('models/scaler.pkl')
            
            if os.path.exists(classifier_path) and os.path.exists(scaler_path):
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                print(f"警告: 分类器文件不存在: {classifier_path} 或 {scaler_path}")
                return None
        
        return self.classifier
    
    def predict(self, features):
        """
        预测手语动作
        
        参数:
            features: 特征数组
            
        返回:
            预测的标签和概率
        """
        if self.classifier is None:
            self.load_model()
            
            if self.classifier is None:
                raise ValueError("分类器未加载")
        
        if self.classifier_type == 'lstm':
            # 对于LSTM，需要序列数据
            if len(features.shape) < 3:
                raise ValueError("LSTM需要序列数据，形状应为(样本数, 时间步, 特征数)")
            
            # 预测
            predictions = self.classifier.predict(features)
            
            # 获取最可能的类别
            predicted_ids = np.argmax(predictions, axis=1)
            probabilities = np.max(predictions, axis=1)
        else:
            # 对于其他分类器，需要展平特征
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # 标准化特征
            features = self.scaler.transform(features)
            
            # 预测
            predicted_ids = self.classifier.predict(features)
            probabilities = np.max(self.classifier.predict_proba(features), axis=1)
        
        # 转换ID为标签
        predicted_labels = [self.id_to_label[str(id)] for id in predicted_ids]
        
        return predicted_labels, probabilities
    
    def predict_sequence(self, sequence_features):
        """
        预测手语动作序列
        
        参数:
            sequence_features: 序列特征数组
            
        返回:
            预测的标签序列和概率序列
        """
        if self.classifier_type == 'lstm':
            # 对于LSTM，直接使用序列
            return self.predict(np.array([sequence_features]))
        else:
            # 对于其他分类器，逐帧预测
            predictions = []
            probabilities = []
            
            for features in sequence_features:
                pred, prob = self.predict(np.array([features]))
                predictions.append(pred[0])
                probabilities.append(prob[0])
            
            return predictions, probabilities 