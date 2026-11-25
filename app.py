#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
手语识别系统Web应用程序
"""

import os
import cv2
import numpy as np
import json
import base64
import yaml
import pyttsx3
from googletrans import Translator
from flask import Flask, render_template, request, jsonify, Response, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# 统一路径解析（支持通过 SIGN_LANGUAGE_ASSETS_DIR 把大模型放到项目外）
from paths import APP_ROOT, app_path, resolve, find_chinese_font

# 创建Flask应用
app = Flask(__name__, 
    static_folder=os.path.join(APP_ROOT, 'static'),
    static_url_path=''
)
CORS(app)

# 设置响应超时时间和其他配置
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=0,
    PERMANENT_SESSION_LIFETIME=3600,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
)

# 加载配置（配置文件始终位于项目根目录，不依赖当前工作目录）
try:
    with open(app_path('config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"加载配置文件失败: {str(e)}")
    config = {'names': ["未知手语"]}

# 设置模型路径（外部资源目录优先，回退到项目内 models/）
YOLO_MODEL_PATH = resolve('models', 'yolo', 'exp', 'weights', 'best.pt')
OPENPOSE_MODEL_PATH = resolve('models', 'openpose', 'pose_iter_440000.caffemodel')
OPENPOSE_PROTO_PATH = resolve('models', 'openpose', 'pose_deploy_linevec.prototxt')
OPENPOSE_HAND_MODEL_PATH = resolve('models', 'caffe_models', 'hand', 'pose_iter_102000.caffemodel')
OPENPOSE_HAND_PROTO_PATH = resolve('models', 'openpose', 'hand_pose_deploy.prototxt')

# 初始化模型
yolo = None
openpose = None

# 初始化翻译器
try:
    translator = Translator(service_urls=['translate.google.com'])
except Exception as e:
    print(f"翻译器初始化失败: {str(e)}")
    translator = None

# 初始化TTS引擎
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    voices = tts_engine.getProperty('voices')
except Exception as e:
    print(f"TTS引擎初始化失败: {str(e)}")
    tts_engine = None
    voices = []

# 存储检测到的单词列表
detected_words = []

# 定义身体关键点连接对
body_pairs = {
    'head': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'torso': [(1, 5), (1, 8)],
    'right_arm': [(1, 2), (2, 3), (3, 4)],
    'left_arm': [(1, 5), (5, 6), (6, 7)],
    'right_leg': [(8, 9), (9, 10), (10, 11)],
    'left_leg': [(8, 12), (12, 13), (13, 14)]
}

# 定义手部关键点连接对
hand_pairs = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
}

# 定义身体部位的颜色
body_colors = {
    'head': (255, 0, 0),
    'torso': (0, 255, 0),
    'right_arm': (0, 0, 255),
    'left_arm': (255, 255, 0),
    'right_leg': (255, 0, 255),
    'left_leg': (0, 255, 255)
}

# 定义手指的颜色
hand_colors = {
    'thumb': (255, 0, 0),
    'index': (0, 255, 0),
    'middle': (0, 0, 255),
    'ring': (255, 255, 0),
    'pinky': (255, 0, 255)
}

def load_models():
    """加载所有模型"""
    global yolo, openpose
    
    try:
        # 加载YOLO模型
        print("正在加载YOLO模型...")
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO模型文件不存在: {YOLO_MODEL_PATH}")
        yolo = YOLO(YOLO_MODEL_PATH)
        print("YOLO模型加载成功")
        
        # 加载OpenPose模型
        print("正在加载OpenPose模型...")
        if not os.path.exists(OPENPOSE_MODEL_PATH):
            raise FileNotFoundError(f"OpenPose模型文件不存在: {OPENPOSE_MODEL_PATH}")
        openpose = cv2.dnn.readNetFromCaffe(OPENPOSE_PROTO_PATH, OPENPOSE_MODEL_PATH)
        print("OpenPose模型加载成功")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 初始化加载模型
load_models()

# 设置上传文件目录（运行时产物，固定在项目内）
UPLOAD_FOLDER = app_path('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_chinese_text(img, text, position, text_color=(0, 255, 0)):
    """使用PIL绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    fontpath = find_chinese_font()
    if fontpath:
        font = ImageFont.truetype(fontpath, 32)
    else:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_image(image, conf_threshold=0.5, iou_threshold=0.45):
    """处理图像并返回检测结果"""
    if yolo is None:
        return None, "YOLO模型未加载"
        
    try:
        # 创建帧的副本以避免修改原始帧
        display_frame = image.copy()
        
        # 使用YOLO进行预测
        results = yolo.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detected_classes = []
        max_conf = 0
        
        # 处理检测结果
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                if conf > max_conf:
                    max_conf = conf
                
                # 获取类别名称
                class_name = config['names'][cls] if cls < len(config['names']) else "未知"
                detected_classes.append({
                    "class": class_name,
                    "confidence": float(conf)
                })
                
                # 绘制边界框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制中文标签
                label = f"{class_name}: {conf:.2f}"
                display_frame = draw_chinese_text(display_frame, label, (x1, max(y1 - 35, 0)))
                
                # 添加到检测词列表
                if class_name not in detected_words:
                    detected_words.append(class_name)
        
        # 使用OpenPose进行姿态估计
        if openpose is not None:
            # 准备输入blob
            blob = cv2.dnn.blobFromImage(image, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            openpose.setInput(blob)
            
            # 获取输出
            output = openpose.forward()
            
            # 处理关键点
            H, W = image.shape[:2]
            points = []
            
            # 提取关键点
            for i in range(output.shape[1]):
                heatMap = output[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = int((W * point[0]) / output.shape[3])
                y = int((H * point[1]) / output.shape[2])
                
                if conf > 0.1:  # 置信度阈值
                    points.append((x, y))
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 255), -1)
            
            # 绘制骨架连接
            for part, pairs in body_pairs.items():
                color = body_colors.get(part, (0, 255, 0))
                for pair in pairs:
                    if pair[0] < len(points) and pair[1] < len(points):
                        pt1 = points[pair[0]]
                        pt2 = points[pair[1]]
                        cv2.line(display_frame, pt1, pt2, color, 2)
        
        return display_frame, {
            "detections": detected_classes,
            "max_confidence": float(max_conf) if max_conf > 0 else 0,
            "detected_words": detected_words
        }
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def speak_text(text, language='zh'):
    """语音播报文本"""
    if tts_engine is None:
        return False
        
    try:
        # 如果需要英文播报，先翻译
        if language == 'en' and translator is not None:
            try:
                translation = translator.translate(text, dest='en')
                if hasattr(translation, 'text'):
                    text = translation.text
            except Exception as e:
                print(f"翻译失败: {str(e)}")
        
        # 选择合适的声音
        for voice in voices:
            if language in voice.name.lower():
                tts_engine.setProperty('voice', voice.id)
                break
        
        tts_engine.say(text)
        tts_engine.runAndWait()
        return True
    except Exception as e:
        print(f"语音播报失败: {str(e)}")
        return False

def clear_detected_words():
    """清空检测到的单词列表"""
    global detected_words
    detected_words = []

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

@app.route('/image_recognition')
def image_recognition():
    """图像识别页面"""
    return render_template('image_recognition.html')

@app.route('/video_recognition')
def video_recognition():
    """视频识别页面"""
    return render_template('video_recognition.html')

@app.route('/realtime_recognition')
def realtime_recognition():
    """实时识别页面"""
    return render_template('realtime_recognition.html')

@app.route('/api/recognize_image', methods=['POST'])
def api_recognize_image():
    """图像识别API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 保存文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 读取图像
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': '无法读取图像'}), 400
            
            # 获取置信度阈值
            conf_threshold = float(request.form.get('confidence', 0.5))
            iou_threshold = float(request.form.get('iou', 0.45))
            
            # 处理图像
            result_image, detections = process_image(image, conf_threshold, iou_threshold)
            
            if result_image is None:
                return jsonify({'error': detections}), 500
            
            # 保存结果图像
            result_filename = 'result_' + filename
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_filepath, result_image)
            
            # 如果请求中包含语音播报标志
            if request.form.get('speak', 'false').lower() == 'true':
                if detections['detections']:
                    # 获取最高置信度的检测结果
                    best_detection = max(detections['detections'], key=lambda x: x['confidence'])
                    speak_text(best_detection['class'])
            
            return jsonify({
                'detections': detections['detections'],
                'max_confidence': detections['max_confidence'],
                'detected_words': detections['detected_words'],
                'image_url': '/static/uploads/' + result_filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/api/recognize_video', methods=['POST'])
def api_recognize_video():
    """视频识别API"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 保存文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 打开视频文件
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return jsonify({'error': '无法打开视频文件'}), 400
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建视频写入器
            result_filename = 'result_' + filename
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_filepath, fourcc, fps, (frame_width, frame_height))
            
            # 处理每一帧
            all_detections = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result_frame, detections = process_image(frame)
                if result_frame is not None:
                    out.write(result_frame)
                    all_detections.append(detections)
            
            # 释放资源
            cap.release()
            out.release()
            
            # 统计结果
            if all_detections:
                # 获取所有检测到的类别及其出现次数
                from collections import Counter
                all_classes = []
                for det in all_detections:
                    for d in det['detections']:
                        all_classes.append(d['class'])
                
                most_common = Counter(all_classes).most_common(1)
                if most_common:
                    final_prediction = most_common[0][0]
                    confidence = most_common[0][1] / len(all_detections)
                else:
                    final_prediction = "未检测到手语"
                    confidence = 0
                
                # 如果请求中包含语音播报标志
                if request.form.get('speak', 'false').lower() == 'true':
                    speak_text(final_prediction)
                
                return jsonify({
                    'video_url': '/static/uploads/' + result_filename,
                    'final_prediction': final_prediction,
                    'confidence': confidence,
                    'frame_count': len(all_detections),
                    'detected_words': detected_words
                })
            else:
                return jsonify({'error': '视频处理失败'}), 500
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': '不支持的文件类型'}), 400

def generate_frames():
    """生成实时视频帧"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        result_frame, detections = process_image(frame)
        if result_frame is None:
            continue
        
        # 转换为JPEG
        ret, buffer = cv2.imencode('.jpg', result_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """视频流"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/speak_text', methods=['POST'])
def api_speak_text():
    """语音播报API"""
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'zh')
    
    if not text:
        return jsonify({'error': '没有要播报的文本'}), 400
    
    success = speak_text(text, language)
    return jsonify({'success': success})

@app.route('/api/clear_words', methods=['POST'])
def api_clear_words():
    """清空检测到的单词列表API"""
    clear_detected_words()
    return jsonify({'success': True})

@app.route('/api/get_detected_words', methods=['GET'])
def api_get_detected_words():
    """获取检测到的单词列表API"""
    return jsonify({
        'detected_words': detected_words
    })

# 添加静态文件路由
@app.route('/static/<path:filename>')
def serve_static(filename):
    """处理静态文件请求"""
    try:
        print(f"请求静态文件: {filename}")
        response = app.send_static_file(filename)
        print(f"静态文件响应: {response.status_code}")
        return response
    except Exception as e:
        print(f"静态文件请求失败: {str(e)}")
        return str(e), 404

if __name__ == '__main__':
    # 运行Flask应用
    app.run(
        host=config.get('app', {}).get('host', '127.0.0.1'),
        port=config.get('app', {}).get('port', 5000),
        debug=config.get('app', {}).get('debug', True),
        threaded=True,
        use_reloader=False
    ) 