import sys
import os
import cv2
import numpy as np
import pyttsx3  # 添加文本转语音库
from googletrans import Translator  # 添加翻译库
from PyQt5.QtWebEngineWidgets import QWebEngineView  # 添加WebView组件
from PyQt5.QtCore import QUrl  # 添加URL支持
from PyQt5.QtWebChannel import QWebChannel

# 把项目根目录加入导入路径，统一使用根目录的 paths.py（兼容 cd qt 后直接运行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from paths import app_path, resolve, find_chinese_font

# Qt 插件路径：仅在 Conda 风格目录确实存在时设置，其他环境交给 Qt 自行查找
if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
    candidate = os.path.join(os.path.dirname(sys.executable), "Library", "plugins", "platforms")
    if os.path.isdir(candidate):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = candidate
# 禁用Qt的高DPI缩放
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# 设置Qt的缩放因子
os.environ["QT_SCALE_FACTOR"] = "1"

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog,
                           QGroupBox, QRadioButton, QButtonGroup, QSlider, QCheckBox, QFrame, QTextEdit, QComboBox)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QBrush, QPalette, QIcon
import yaml
import torch
from ultralytics import YOLO
from openpose import OpenPose
from PIL import Image, ImageDraw, ImageFont

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._run_flag = True

    def run(self):
        while self._run_flag:
            ret, frame = self.camera.read()
            if ret:
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self._run_flag = False
        self.wait()

class TranslationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手语翻译")
        self.resize(800, 600)
        
        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 初始化手语数字人WebView
        self.avatar_webview = QWebEngineView()
        self.avatar_webview.setFixedHeight(400)
        
        # 设置页面
        self.page = self.avatar_webview.page()
        
        # 添加页面加载完成的信号处理
        self.avatar_webview.loadFinished.connect(self.on_load_finished)
        
        # 设置JavaScript控制台消息处理
        self.page.setWebChannel(QWebChannel(self.page))
        self.page.profile().setHttpUserAgent("Mozilla/5.0")
        
        # 加载手语数字人HTML页面
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>手语翻译</title>
            <style>
                body { 
                    margin: 0; 
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: #f8f9fa;
                }
                #avatar-container {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background: #ffffff;
                }
            </style>
        </head>
        <body>
            <div id="avatar-container"></div>
            <script src="https://avatar.gbqr.net/yiyu.js"></script>
            <script>
                // 数字人 SDK 需要使用者自行申请凭据（token）后填入下方常量。
                // 未配置时仅显示提示，不影响识别功能。
                const YIYU_SDK_TOKEN = '';
                window.yiyuTokenReady = false;

                window.addEventListener('load', function () {
                    const container = document.getElementById('avatar-container');
                    if (!YIYU_SDK_TOKEN) {
                        container.innerHTML =
                            '<div style="padding:20px;color:#856404;background:#fff3cd;">手语数字人未启用：请先申请 yiyu SDK token 并填入 YIYU_SDK_TOKEN。</div>';
                        return;
                    }
                    // 初始化数字人
                    yiyu.app.init({
                        name: YIYU_SDK_TOKEN,
                        readLocalResource: false,
                        container: 'avatar-container'
                    });
                    window.yiyuTokenReady = true;

                    // 监听数字人准备完成事件
                    yiyu.app.onAppReady(function() {
                        console.log('数字人加载完成');
                    });

                    // 监听HTML资源加载完成事件
                    yiyu.app.onHtmlReady(function() {
                        console.log('HTML资源加载完成');
                    });
                });

                // 定义全局翻译函数
                window.translateText = function(text) {
                    if (!YIYU_SDK_TOKEN || !window.yiyuTokenReady || !window.yiyu || !window.yiyu.app) {
                        console.warn('数字人未启用或未加载完成');
                        return false;
                    }
                    try {
                        yiyu.app.startTranslate(text);
                        console.log('翻译请求已发送:', text);
                        return true;
                    } catch (error) {
                        console.error('翻译失败:', error);
                        return false;
                    }
                };
            </script>
        </body>
        </html>
        """
        
        # 加载HTML内容
        base_url = QUrl("http://localhost")  # 修改为使用 localhost
        self.avatar_webview.setHtml(html_content, base_url)
        
        # 设置WebEngine页面设置
        settings = self.page.settings()
        settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)  # 允许本地内容访问远程URL
        settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)    # 允许本地内容访问文件URL
        settings.setAttribute(settings.AllowRunningInsecureContent, True)      # 允许运行不安全内容（开发环境）
        
        # 添加手语数字人控制区域
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("输入要翻译的文本")
        self.text_input.setMaximumHeight(60)
        
        self.translate_btn = QPushButton("翻译")
        self.translate_btn.clicked.connect(self.translate_text)
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 80px;
                height: 28px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
        """)
        
        control_layout.addWidget(self.text_input)
        control_layout.addWidget(self.translate_btn)
        
        # 添加到主布局
        main_layout.addWidget(self.avatar_webview)
        main_layout.addWidget(control_container)
    
    def on_load_finished(self, ok):
        """页面加载完成的处理函数"""
        if ok:
            print("页面加载成功")
            # 检查SDK状态
            self.page.runJavaScript(
                """
                if (window.yiyu && window.yiyu.app) {
                    console.log('SDK已加载');
                    true;
                } else {
                    console.log('SDK未加载');
                    false;
                }
                """,
                self.handle_sdk_status
            )
        else:
            print("页面加载失败")
    
    def handle_sdk_status(self, loaded):
        """处理SDK状态检查结果"""
        if loaded:
            print("SDK已成功加载")
        else:
            print("SDK未能加载，请检查网络连接")
    
    def translate_text(self):
        """翻译文本并显示手语数字人"""
        text = self.text_input.toPlainText()
        if text:
            print(f"开始翻译文本: {text}")
            check_script = """
                if (window.yiyu && window.yiyu.app) {
                    try {
                        window.yiyu.app.startTranslate('%s');
                        console.log('翻译请求已发送');
                        true;
                    } catch (error) {
                        console.error('翻译出错:', error);
                        false;
                    }
                } else {
                    console.error('SDK未加载');
                    false;
                }
            """ % text
            
            self.page.runJavaScript(check_script, self.handle_translate_result)
    
    def handle_translate_result(self, success):
        """处理翻译结果"""
        if success:
            print("翻译执行成功")
        else:
            print("翻译执行失败")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv8的手语识别系统")
        self.resize(1000, 700)
        
        # 定义身体关键点连接对
        self.body_pairs = {
            'head': [(0, 1), (1, 2), (2, 3), (3, 4)],  # 头部
            'torso': [(1, 5), (1, 8)],  # 躯干
            'right_arm': [(1, 2), (2, 3), (3, 4)],  # 右臂
            'left_arm': [(1, 5), (5, 6), (6, 7)],  # 左臂
            'right_leg': [(8, 9), (9, 10), (10, 11)],  # 右腿
            'left_leg': [(8, 12), (12, 13), (13, 14)]  # 左腿
        }
        
        # 定义手部关键点连接对
        self.hand_pairs = {
            'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
            'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
            'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
            'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
            'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
        }
        
        # 定义身体部位的颜色
        self.body_colors = {
            'head': (255, 0, 0),     # 红色
            'torso': (0, 255, 0),    # 绿色
            'right_arm': (0, 0, 255), # 蓝色
            'left_arm': (255, 255, 0), # 黄色
            'right_leg': (255, 0, 255), # 紫色
            'left_leg': (0, 255, 255)   # 青色
        }
        
        # 定义手指的颜色
        self.hand_colors = {
            'thumb': (255, 0, 0),     # 红色
            'index': (0, 255, 0),     # 绿色
            'middle': (0, 0, 255),    # 蓝色
            'ring': (255, 255, 0),    # 黄色
            'pinky': (255, 0, 255)    # 紫色
        }
        
        # 初始化翻译器
        try:
            self.translator = Translator(service_urls=['translate.google.com'])
            # 测试翻译器是否正常工作
            test_result = self.translator.translate('测试', dest='en')
            print(f"翻译器测试结果: {test_result.text}")
        except Exception as e:
            print(f"翻译器初始化失败: {str(e)}")
            self.translator = None
        
        # 初始化文本转语音引擎
        self.tts_engine = None
        self.init_tts_engine()
        
        # 添加存储检测文字的列表
        self.detected_words = []
        
        # 添加TTS状态标志
        self.is_speaking = False
        
        # 定义中文语音的描述和语言标识
        self.voice_descriptions = {
            'microsoft huihui desktop - chinese (simplified)': {'name': '慧慧 (中文)', 'lang': 'zh'},
            'microsoft yaoyao desktop - chinese (simplified)': {'name': '瑶瑶 (中文)', 'lang': 'zh'},
            'microsoft kangkang desktop - chinese (simplified)': {'name': '康康 (男声)', 'lang': 'zh'},
            'microsoft zira desktop - english (united states)': {'name': 'Zira (English)', 'lang': 'en'}
        }
        
        self.current_voice_id = 0  # 默认使用第一个声音
        
        # 创建翻译窗口实例（但不显示）
        self.translation_window = None
        
        # 加载配置
        self.load_config()
        
        # 初始化UI
        self.setup_ui()
        
        # 加载模型
        self.load_models()
        
        # 初始化变量
        self.camera = None
        self.video_thread = None
        self.current_mode = "camera"  # camera, image, video
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.is_camera_running = False
        self.last_frame_time = 0
        self.last_detection_time = 0
        self.detection_interval = 0.5  # 每0.5秒进行一次检测
        self.frame_buffer = None  # 用于存储待处理的帧

    def init_tts_engine(self):
        """初始化TTS引擎"""
        try:
            if self.tts_engine is not None:
                self.tts_engine.stop()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            # 获取所有可用的声音
            self.voices = self.tts_engine.getProperty('voices')
            
            # 打印所有可用的声音信息
            print("\n系统可用的声音列表：")
            for i, voice in enumerate(self.voices):
                print(f"Voice {i}:")
                print(f"  ID: {voice.id}")
                print(f"  Name: {voice.name}")
                print(f"  Languages: {voice.languages}")
                print(f"  Gender: {voice.gender}")
                print(f"  Age: {voice.age}")
            
            # 设置默认声音
            if self.voices:
                self.current_voice_id = 0
                self.tts_engine.setProperty('voice', self.voices[0].id)
                
        except Exception as e:
            print(f"TTS引擎初始化失败: {str(e)}")
            self.tts_engine = None

    def load_config(self):
        try:
            with open(app_path('config.yaml'), 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if 'names' not in self.config:
                print("警告: 配置文件中缺少names字段，将使用默认配置")
                self.config['names'] = ["未知手语"]
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            self.config = {'names': ["未知手语"]}
        
        # 设置模型路径（外部资源目录优先，回退到项目内 models/）
        self.yolo_model_path = resolve('models', 'yolo', 'exp', 'weights', 'best.pt')
        
        # OpenPose 模型路径
        self.openpose_model_path = resolve('models', 'openpose', 'pose_iter_440000.caffemodel')
        self.openpose_proto_path = resolve('models', 'openpose', 'pose_deploy_linevec.prototxt')
        
        # 手部模型路径
        self.openpose_hand_model_path = resolve('models', 'caffe_models', 'hand', 'pose_iter_102000.caffemodel')
        self.openpose_hand_proto_path = resolve('models', 'openpose', 'hand_pose_deploy.prototxt')
        
        print("模型路径配置：")
        print(f"YOLO模型: {self.yolo_model_path}")
        print(f"OpenPose身体模型: {self.openpose_model_path}")
        print(f"OpenPose手部模型: {self.openpose_hand_model_path}")

    def setup_ui(self):
        self.setWindowTitle("基于OpenPose和YOLOv8的手语识别系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建一个半透明的遮罩层
        overlay_widget = QWidget(self)
        overlay_widget.setGeometry(0, 0, self.width(), self.height())
        overlay_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.2);  /* 20%透明度的黑色遮罩 */
            }
        """)
        overlay_widget.lower()  # 将遮罩层放到最底层
        
        # 设置窗口背景
        try:
            # 尝试多个可能的背景图片路径（均相对于当前文件，不再硬编码盘符）
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "background", "test2.jpg"),
                os.path.join(os.path.dirname(__file__), "background", "background.png"),
                os.path.join(os.path.dirname(__file__), "background", "background.jpg")
            ]
            
            background_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"找到背景图片: {path}")
                    background_image = QImage(path)
                    if not background_image.isNull():
                        print(f"成功加载背景图片: {path}")
                        # 缩放图片以适应窗口
                        scaled_image = background_image.scaled(
                            self.size(),
                            Qt.KeepAspectRatioByExpanding,
                            Qt.SmoothTransformation
                        )
                        palette = self.palette()
                        palette.setBrush(QPalette.Window, QBrush(scaled_image))
                        self.setPalette(palette)
                        self.setAutoFillBackground(True)
                        background_loaded = True
                        break
            
            if not background_loaded:
                print("未能加载任何背景图片，使用渐变背景")
                # 设置渐变背景
                self.setStyleSheet("""
                    QMainWindow {
                        background: qlineargradient(
                            x1: 0, y1: 0, x2: 1, y2: 1,
                            stop: 0 #2C3E50,
                            stop: 1 #3498DB
                        );
                    }
                """)
                
        except Exception as e:
            print(f"设置背景时出错: {str(e)}")
            print("当前工作目录:", os.getcwd())
            import traceback
            traceback.print_exc()
            # 设置默认渐变背景
            self.setStyleSheet("""
                QMainWindow {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 1,
                        stop: 0 #2C3E50,
                        stop: 1 #3498DB
                    );
                }
            """)
        
        # 设置主窗口样式
        additional_style = """
            QLabel {
                font-size: 14px;
                color: white;
                padding: 5px;
            }
            QPushButton {
                background-color: rgba(70, 130, 180, 0.5);
                color: white;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 80px;
                max-width: 80px;
                height: 28px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.4);
            }
            QGroupBox {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                margin-top: 8px;
                font-size: 12px;
                padding: 12px;
                color: white;
            }
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """
        self.setStyleSheet(self.styleSheet() + additional_style)
        
        # 窗口大小改变时更新遮罩层大小
        self.resizeEvent = lambda event: overlay_widget.setGeometry(0, 0, self.width(), self.height())

        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧控制面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(250)
        
        # 1. 检测模式组
        mode_group = QGroupBox("检测模式")
        mode_layout = QVBoxLayout(mode_group)
        
        # 创建单选按钮
        self.camera_radio = QRadioButton("摄像头检测")
        self.video_radio = QRadioButton("视频检测")
        self.image_radio = QRadioButton("图片检测")
        
        # 设置单选按钮图标
        camera_icon = QIcon("qt/img/摄像头.png")
        video_icon = QIcon("qt/img/视频.png")
        image_icon = QIcon("qt/img/图片.jpg")
        
        self.camera_radio.setIcon(camera_icon)
        self.video_radio.setIcon(video_icon)
        self.image_radio.setIcon(image_icon)
        
        # 设置图标大小
        icon_size = 20
        self.camera_radio.setIconSize(QSize(icon_size, icon_size))
        self.video_radio.setIconSize(QSize(icon_size, icon_size))
        self.image_radio.setIcon(image_icon)
        
        # 设置默认选中摄像头模式
        self.camera_radio.setChecked(True)
        
        # 添加到布局
        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.video_radio)
        mode_layout.addWidget(self.image_radio)
        
        # 创建按钮组
        mode_button_group = QButtonGroup(self)
        mode_button_group.addButton(self.camera_radio)
        mode_button_group.addButton(self.video_radio)
        mode_button_group.addButton(self.image_radio)
        
        # 连接信号
        self.camera_radio.toggled.connect(lambda: self.mode_changed("camera"))
        self.video_radio.toggled.connect(lambda: self.mode_changed("video"))
        self.image_radio.toggled.connect(lambda: self.mode_changed("image"))
        
        left_layout.addWidget(mode_group)
        
        # 2. 模型控制组
        model_control_group = QGroupBox("模型控制")
        model_control_layout = QVBoxLayout(model_control_group)
        
        # OpenPose 控制
        self.openpose_checkbox = QCheckBox("启用 OpenPose")
        self.openpose_checkbox.setChecked(True)
        self.openpose_checkbox.stateChanged.connect(self.toggle_openpose)
        self.select_openpose_model_btn = QPushButton("选择 OpenPose 模型")
        self.select_openpose_model_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 12px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 120px;
                max-width: none;
                height: 25px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.4);
            }
        """)
        self.select_openpose_model_btn.clicked.connect(self.select_openpose_model)
        
        # YOLOv8 控制
        self.yolo_checkbox = QCheckBox("启用 YOLOv8")
        self.yolo_checkbox.setChecked(True)
        self.yolo_checkbox.stateChanged.connect(self.toggle_yolo)
        self.select_yolo_model_btn = QPushButton("选择 YOLOv8 模型")
        self.select_yolo_model_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 12px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 120px;
                max-width: none;
                height: 25px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.4);
            }
        """)
        self.select_yolo_model_btn.clicked.connect(self.select_yolo_model)
        
        model_control_layout.addWidget(self.openpose_checkbox)
        model_control_layout.addWidget(self.select_openpose_model_btn)
        model_control_layout.addWidget(self.yolo_checkbox)
        model_control_layout.addWidget(self.select_yolo_model_btn)
        
        left_layout.addWidget(model_control_group)
        
        # 3. 参数设置组
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(5)  # 减小布局间距
        params_layout.setContentsMargins(8, 8, 8, 8)  # 减小边距
        
        # 检测间隔控制
        detection_layout = QVBoxLayout()
        detection_layout.setSpacing(2)  # 减小布局间距
        detection_label = QLabel("检测间隔(秒):")
        detection_label.setStyleSheet("padding: 0px;")  # 移除内边距
        self.detection_slider = QSlider(Qt.Horizontal)
        self.detection_slider.setFixedHeight(15)  # 设置滑块固定高度
        self.detection_slider.setRange(1, 10)
        self.detection_slider.setValue(5)
        self.detection_slider.valueChanged.connect(self.update_detection_interval)
        detection_layout.addWidget(detection_label)
        detection_layout.addWidget(self.detection_slider)
        
        # 帧率控制
        fps_layout = QVBoxLayout()
        fps_layout.setSpacing(2)  # 减小布局间距
        fps_label = QLabel("播放速度:")
        fps_label.setStyleSheet("padding: 0px;")  # 移除内边距
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setFixedHeight(15)  # 设置滑块固定高度
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(30)
        self.fps_slider.valueChanged.connect(self.update_fps)
        self.fps_value_label = QLabel("30 FPS")
        self.fps_value_label.setStyleSheet("padding: 0px;")  # 移除内边距
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_slider)
        fps_layout.addWidget(self.fps_value_label)
        
        # 置信度阈值
        confidence_layout = QVBoxLayout()
        confidence_layout.setSpacing(2)  # 减小布局间距
        confidence_label = QLabel("置信度阈值:")
        confidence_label.setStyleSheet("padding: 0px;")  # 移除内边距
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setFixedHeight(15)  # 设置滑块固定高度
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        
        # IOU阈值
        iou_layout = QVBoxLayout()
        iou_layout.setSpacing(2)  # 减小布局间距
        iou_label = QLabel("IOU阈值:")
        iou_label.setStyleSheet("padding: 0px;")  # 移除内边距
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setFixedHeight(15)  # 设置滑块固定高度
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        
        params_layout.addLayout(detection_layout)
        params_layout.addLayout(fps_layout)
        params_layout.addLayout(confidence_layout)
        params_layout.addLayout(iou_layout)
        
        left_layout.addWidget(params_group)
        left_layout.addStretch()
        
        # 右侧显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        
        # 视频显示区域
        video_container = QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.15);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
            }
        """)
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        right_layout.addWidget(video_container)
        
        # 按钮区域
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(10)
        
        # 创建按钮
        self.camera_button = QPushButton("打开摄像头")
        self.file_button = QPushButton("选择视频")
        self.start_button = QPushButton("开始识别")
        self.stop_button = QPushButton("停止识别")
        
        # 在按钮区域添加打开翻译窗口的按钮
        self.open_translation_btn = QPushButton("手语翻译")
        self.open_translation_btn.clicked.connect(self.open_translation_window)
        
        # 设置按钮图标
        camera_icon = QIcon("qt/img/摄像头.png")
        video_icon = QIcon("qt/img/视频.png")
        image_icon = QIcon("qt/img/图片.jpg")
        
        # 设置图标大小
        icon_size = QSize(20, 20)
        
        # 设置摄像头按钮图标
        self.camera_button.setIcon(camera_icon)
        self.camera_button.setIconSize(icon_size)
        
        # 添加按钮到布局
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.file_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.open_translation_btn)
        button_layout.addStretch()
        
        right_layout.addWidget(button_container)
        
        # 文本显示区域和语音播报按钮的容器
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setSpacing(10)
        
        # 检测结果显示
        detection_container = QWidget()
        detection_layout = QHBoxLayout(detection_container)
        detection_layout.setSpacing(10)
        
        self.text_display = QLabel("检测到：")
        self.text_display.setStyleSheet("""
            QLabel {
                color: black;
                font-size: 16px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 5px;
                min-height: 40px;
            }
        """)
        
        # 语音播报按钮和声音选择下拉框的容器
        voice_control_container = QWidget()
        voice_control_layout = QVBoxLayout(voice_control_container)
        voice_control_layout.setSpacing(5)
        
        # 声音选择下拉框
        self.voice_combo = QComboBox()
        self.voice_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 200px;
                max-width: 200px;
                height: 25px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QComboBox:hover {
                background-color: rgba(70, 130, 180, 1);
            }
        """)
        
        # 添加可用的声音到下拉框
        voices_added = False
        for voice in self.voices:
            # 获取声音名称（转换为小写以便比较）
            voice_name = voice.name.lower()
            # 获取友好的显示名称
            voice_info = self.voice_descriptions.get(voice_name, {'name': voice.name, 'lang': 'zh'})
            self.voice_combo.addItem(voice_info['name'])
            voices_added = True
        
        # 如果没有找到任何声音，添加默认选项
        if not voices_added:
            print("未找到任何可用的声音，添加默认选项")
            self.voice_combo.addItem("默认声音")
        
        # 连接声音选择变化事件
        self.voice_combo.currentIndexChanged.connect(self.change_voice)
        
        # 语音播报按钮
        self.voice_button = QPushButton("语音播报")
        
        voice_control_layout.addWidget(self.voice_combo)
        voice_control_layout.addWidget(self.voice_button)
        
        detection_layout.addWidget(self.text_display, stretch=1)
        detection_layout.addWidget(voice_control_container)
        
        # 完整句子显示区域
        sentence_container = QWidget()
        sentence_layout = QHBoxLayout(sentence_container)
        sentence_layout.setSpacing(10)
        
        self.sentence_display = QLabel("完整句子：")
        self.sentence_display.setStyleSheet("""
            QLabel {
                color: black;
                font-size: 16px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 5px;
                min-height: 40px;
            }
        """)
        
        # 播放句子和清空句子按钮
        self.play_sentence_button = QPushButton("播放句子")
        self.clear_sentence_button = QPushButton("清空句子")
        
        sentence_layout.addWidget(self.sentence_display, stretch=1)
        sentence_layout.addWidget(self.play_sentence_button)
        sentence_layout.addWidget(self.clear_sentence_button)
        
        # 将两个容器添加到主文本布局
        text_layout.addWidget(detection_container)
        text_layout.addWidget(sentence_container)
        
        right_layout.addWidget(text_container)
        
        # 统一按钮样式
        button_style = """
            QPushButton {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 14px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 80px;
                max-width: 80px;
                height: 28px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.4);
            }
        """
        
        # 模型选择按钮样式
        model_button_style = """
            QPushButton {
                background-color: rgba(70, 130, 180, 0.8);
                color: white;
                border-radius: 14px;
                padding: 4px 8px;
                font-size: 12px;
                min-width: 120px;
                max-width: none;
                height: 25px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton:hover {
                background-color: rgba(70, 130, 180, 1);
            }
            QPushButton:disabled {
                background-color: rgba(70, 130, 180, 0.4);
            }
        """
        
        # 应用模型选择按钮样式
        self.select_openpose_model_btn.setStyleSheet(model_button_style)
        self.select_yolo_model_btn.setStyleSheet(model_button_style)
        
        # 应用普通按钮样式
        for btn in [self.camera_button, self.file_button, self.start_button, self.stop_button, 
                   self.voice_button, self.play_sentence_button, self.clear_sentence_button]:
            btn.setStyleSheet(button_style)
        
        # 连接按钮信号
        self.camera_button.clicked.connect(self.toggle_camera)  # 添加摄像头按钮信号连接
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.file_button.clicked.connect(self.upload_file)
        self.voice_button.clicked.connect(self.speak_text)
        self.play_sentence_button.clicked.connect(self.speak_sentence)
        self.clear_sentence_button.clicked.connect(self.clear_sentence)
        
        # 初始状态设置
        self.stop_button.setEnabled(False)
        self.camera_button.setEnabled(True)  # 设置摄像头按钮初始状态
        
        # 更新mode_changed方法中的文件按钮图标设置
        self.mode_icons = {
            "video": video_icon,
            "image": image_icon
        }
        
        # 添加左右两个部分到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, stretch=1)

    def mode_changed(self, mode):
        self.current_mode = mode
        self.stop_detection()  # 停止当前检测
        
        if mode == "camera":
            self.file_button.hide()           # 隐藏文件选择按钮
            self.camera_button.show()         # 显示摄像头按钮
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.start_button.setText("开始识别")
        elif mode == "video":
            self.file_button.show()           # 显示文件选择按钮
            self.camera_button.hide()         # 隐藏摄像头按钮
            self.start_button.setEnabled(False)
            self.file_button.setText("选择视频")
            self.file_button.setIcon(self.mode_icons["video"])  # 设置视频图标
        else:  # image mode
            self.file_button.show()           # 显示文件选择按钮
            self.camera_button.hide()         # 隐藏摄像头按钮
            self.start_button.setEnabled(False)
            self.file_button.setText("选择图片")
            self.file_button.setIcon(self.mode_icons["image"])  # 设置图片图标

    def upload_file(self):
        if self.current_mode == "video":
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                "",
                "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
            )
            if file_name:
                self.stop_detection()  # 停止当前检测
                # 规范化路径并处理编码
                file_name = os.path.normpath(file_name)
                self.video_capture = cv2.VideoCapture(file_name)
                if self.video_capture.isOpened():
                    self.start_button.setEnabled(True)
                    self.text_display.setText("视频已加载，点击开始识别")
                else:
                    self.text_display.setText("无法加载视频文件")
        else:  # image mode
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片",
                "",
                "图片文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
            )
            if file_name:
                self.stop_detection()  # 停止当前检测
                # 规范化路径并处理编码
                file_name = os.path.normpath(file_name)
                frame = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self.process_and_display_frame(frame)
                else:
                    self.text_display.setText("无法加载图片")

    def start_detection(self):
        """开始检测"""
        if self.current_mode == "camera":
            if self.is_camera_running:
                # 设置更快的定时器间隔，让视频流畅显示
                self.timer.start(33)  # 约30fps的视频显示速度
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
        elif self.current_mode == "video" and self.video_capture is not None:
            self.timer.start(33)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.file_button.setEnabled(False)

    def stop_detection(self):
        """停止检测"""
        if self.current_mode == "camera":
            self.start_button.setEnabled(self.is_camera_running)
            self.stop_button.setEnabled(False)
        elif self.current_mode == "video":
            self.timer.stop()
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.file_button.setEnabled(True)  # 重新启用文件选择按钮
            self.video_label.clear()
            self.text_display.setText("未检测到手语")

    def toggle_camera(self):
        """控制摄像头的打开和关闭"""
        if not self.is_camera_running:
            # 尝试打开摄像头
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("无法打开摄像头")
                return
                
            self.video_thread = VideoThread(self.camera)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.start()
            
            # 更新按钮状态
            self.camera_button.setText("关闭摄像头")
            self.camera_button.setIcon(QIcon("qt/img/摄像头.png"))  # 保持图标
            self.camera_button.setIconSize(QSize(20, 20))  # 保持图标大小
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.is_camera_running = True
        else:
            # 关闭摄像头
            self.stop_camera()
            self.camera_button.setText("打开摄像头")
            self.camera_button.setIcon(QIcon("qt/img/摄像头.png"))  # 保持图标
            self.camera_button.setIconSize(QSize(20, 20))  # 保持图标大小
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.is_camera_running = False

    def stop_camera(self):
        """停止摄像头采集"""
        if self.is_camera_running:
            if self.video_thread is not None:
                self.video_thread.stop()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.camera_button.setText("打开摄像头")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.is_camera_running = False
            self.video_label.clear()
            self.text_display.setText("未检测到手语")

    def update_image(self, frame):
        self.process_and_display_frame(frame)

    def update_fps(self, value):
        """更新视频播放帧率"""
        if self.current_mode == "video" and self.video_capture is not None:
            # 计算新的定时器间隔（毫秒）
            interval = int(1000 / value)  # 转换为毫秒
            self.timer.setInterval(interval)
            # 更新帧率控制的时间间隔
            self.frame_interval = 1.0 / value
            # 更新帧率显示标签
            self.fps_value_label.setText(f"{value} FPS")

    def process_frame(self):
        if self.current_mode == "camera" and self.is_camera_running:
            ret, frame = self.camera.read()
            if not ret:
                return
                
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            elapsed_time = current_time - self.last_detection_time
            
            # 存储当前帧
            self.frame_buffer = frame
            
            # 只在达到检测间隔时进行检测
            if elapsed_time >= self.detection_interval:
                self.last_detection_time = current_time
                try:
                    self.process_and_display_frame(frame, do_detection=True)
                except Exception as e:
                    print(f"处理帧时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # 不进行检测，只显示帧
                self.process_and_display_frame(frame, do_detection=False)
                
        elif self.current_mode == "video" and self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                self.stop_detection()
                self.text_display.setText("视频播放完成")
                return
                
            try:
                self.process_and_display_frame(frame, do_detection=True)
            except Exception as e:
                print(f"处理帧时出错: {str(e)}")
                import traceback
                traceback.print_exc()

    def draw_chinese_text(self, img, text, position, text_color=(0, 255, 0)):
        """使用PIL绘制中文文本"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 加载中文字体（跨平台查找，随仓库附带字体优先级最高）
        fontpath = find_chinese_font()
        if fontpath:
            font = ImageFont.truetype(fontpath, 32)
        else:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=text_color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
    def process_and_display_frame(self, frame, do_detection=True):
        if frame is None:
            return
            
        # 创建帧的副本以避免修改原始帧
        display_frame = frame.copy()
            
        # 只在需要检测时执行YOLO和OpenPose处理
        if do_detection:
            detected_text = ""
            try:
                if self.yolo is not None and self.yolo_checkbox.isChecked():
                    # 获取当前置信度阈值
                    conf_threshold = self.confidence_slider.value() / 100.0
                    iou_threshold = self.iou_slider.value() / 100.0
                    
                    # 使用阈值进行预测
                    results = self.yolo.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    # 在图像上绘制检测结果
                    detected_classes = []
                    max_conf = 0
                    
                    # 获取第一个结果（如果有的话）
                    if len(results) > 0:
                        result = results[0]  # 获取第一个结果
                        boxes = result.boxes
                        
                        for box in boxes:
                            # 转换坐标为整数
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if conf > max_conf:
                                max_conf = conf
                            
                            # 获取类别名称
                            class_name = self.config['names'][cls] if cls < len(self.config['names']) else "未知"
                            detected_classes.append(f"{class_name} ({conf:.2f})")
                            
                            # 绘制边界框
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 使用自定义函数绘制中文标签
                            label = f"{class_name}: {conf:.2f}"
                            display_frame = self.draw_chinese_text(display_frame, label, (x1, max(y1 - 35, 0)))
                    
                    # 更新文本显示
                    if detected_classes:
                        detected_text = "检测到: " + ", ".join(detected_classes)
                        self.text_display.setText(detected_text)
                        
                        # 提取检测到的手语文字（不包括置信度）
                        for class_text in detected_classes:
                            word = class_text[:class_text.find(" (")]
                            if word not in self.detected_words:  # 避免重复添加相同的词
                                self.detected_words.append(word)
                                # 更新完整句子显示
                                self.update_sentence_display()
                    else:
                        self.text_display.setText("未检测到手语")
                        
            except Exception as e:
                print(f"YOLO检测失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # 使用OpenPose进行姿态估计
            try:
                if self.openpose is not None and self.openpose_checkbox.isChecked():
                    body_keypoints, hand_keypoints = self.openpose.estimate(frame)
                    
                    # 绘制身体关键点和连接线
                    if body_keypoints is not None and hasattr(self, 'body_pairs'):
                        # 绘制关键点（红色）
                        for point in body_keypoints:
                            if point is not None and len(point) >= 2:
                                cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                        
                        # 绘制身体连接线
                        for part, pairs in self.body_pairs.items():
                            color = self.body_colors.get(part, (0, 255, 0))  # 使用默认颜色（绿色）
                            for pair in pairs:
                                if (pair[0] < len(body_keypoints) and 
                                    pair[1] < len(body_keypoints) and
                                    body_keypoints[pair[0]] is not None and 
                                    body_keypoints[pair[1]] is not None and 
                                    len(body_keypoints[pair[0]]) >= 2 and 
                                    len(body_keypoints[pair[1]]) >= 2):
                                    pt1 = (int(body_keypoints[pair[0]][0]), int(body_keypoints[pair[0]][1]))
                                    pt2 = (int(body_keypoints[pair[1]][0]), int(body_keypoints[pair[1]][1]))
                                    cv2.line(display_frame, pt1, pt2, color, 2)
                    
                    # 绘制手部关键点和连接线
                    if hand_keypoints is not None and hasattr(self, 'hand_pairs'):
                        # 绘制手部关键点（红色）
                        for point in hand_keypoints:
                            if point is not None and len(point) >= 2:
                                cv2.circle(display_frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
                        
                        # 绘制手部连接线
                        for finger, pairs in self.hand_pairs.items():
                            color = self.hand_colors.get(finger, (0, 255, 0))  # 使用默认颜色（绿色）
                            for pair in pairs:
                                if (pair[0] < len(hand_keypoints) and 
                                    pair[1] < len(hand_keypoints) and 
                                    hand_keypoints[pair[0]] is not None and 
                                    hand_keypoints[pair[1]] is not None and 
                                    len(hand_keypoints[pair[0]]) >= 2 and 
                                    len(hand_keypoints[pair[1]]) >= 2):
                                    pt1 = (int(hand_keypoints[pair[0]][0]), int(hand_keypoints[pair[0]][1]))
                                    pt2 = (int(hand_keypoints[pair[1]][0]), int(hand_keypoints[pair[1]][1]))
                                    cv2.line(display_frame, pt1, pt2, color, 2)
                        
            except Exception as e:
                print(f"OpenPose检测失败: {str(e)}")
                import traceback
                traceback.print_exc()

        # 将OpenCV图像转换为Qt图像并显示
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_models(self):
        try:
            # 加载YOLO模型
            print("正在加载YOLO模型...")
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"YOLO模型文件不存在: {self.yolo_model_path}")
            self.yolo = YOLO(self.yolo_model_path)
            print("YOLO模型加载成功")
            
            # 加载OpenPose模型
            print("正在加载OpenPose模型...")
            if not os.path.exists(self.openpose_model_path):
                print(f"警告: OpenPose身体模型文件不存在: {self.openpose_model_path}")
                self.openpose = None
                return
                
            if not os.path.exists(self.openpose_hand_model_path):
                print(f"警告: OpenPose手部模型文件不存在: {self.openpose_hand_model_path}")
                self.openpose = None
                return
                
            try:
                # 检查所有必需的文件
                required_files = [
                    (self.openpose_model_path, "身体模型"),
                    (self.openpose_proto_path, "身体模型配置"),
                    (self.openpose_hand_model_path, "手部模型"),
                    (self.openpose_hand_proto_path, "手部模型配置")
                ]
                
                for file_path, file_desc in required_files:
                    if not os.path.exists(file_path):
                        print(f"警告: {file_desc}文件不存在: {file_path}")
                        self.openpose = None
                        return
                    else:
                        print(f"{file_desc}文件已找到: {file_path}")
                
                self.openpose = OpenPose(
                    self.openpose_model_path,
                    self.openpose_proto_path,
                    self.openpose_hand_model_path,
                    self.openpose_hand_proto_path
                )
                print("OpenPose模型加载成功")
            except Exception as e:
                print(f"OpenPose初始化失败: {str(e)}")
                print("详细错误信息:")
                import traceback
                traceback.print_exc()
                self.openpose = None
                
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.openpose = None
            self.yolo = None

    def toggle_openpose(self, state):
        """切换OpenPose模型的启用状态"""
        if state == Qt.Checked:
            self.load_openpose_model()
        else:
            self.openpose = None
            print("OpenPose已禁用")

    def toggle_yolo(self, state):
        """切换YOLOv8模型的启用状态"""
        if state == Qt.Checked:
            self.load_yolo_model()
        else:
            self.yolo = None
            print("YOLOv8已禁用")

    def select_openpose_model(self):
        """选择OpenPose模型文件"""
        model_dir = QFileDialog.getExistingDirectory(self, "选择OpenPose模型目录")
        if model_dir:
            # 更新模型路径
            self.openpose_model_path = os.path.join(model_dir, "pose_iter_440000.caffemodel")
            self.openpose_proto_path = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
            self.openpose_hand_model_path = os.path.join(model_dir, "hand_pose_model.caffemodel")
            self.openpose_hand_proto_path = os.path.join(model_dir, "hand_pose_deploy.prototxt")
            
            # 重新加载模型
            if self.openpose_checkbox.isChecked():
                self.load_openpose_model()

    def select_yolo_model(self):
        """选择YOLOv8模型文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLOv8模型文件",
            "",
            "YOLO模型 (*.pt);;所有文件 (*.*)"
        )
        if file_name:
            self.yolo_model_path = file_name
            # 重新加载模型
            if self.yolo_checkbox.isChecked():
                self.load_yolo_model()

    def load_openpose_model(self):
        """加载OpenPose模型"""
        try:
            if not all(os.path.exists(path) for path in [
                self.openpose_model_path,
                self.openpose_proto_path,
                self.openpose_hand_model_path,
                self.openpose_hand_proto_path
            ]):
                raise FileNotFoundError("OpenPose模型文件不完整")
                
            self.openpose = OpenPose(
                self.openpose_model_path,
                self.openpose_proto_path,
                self.openpose_hand_model_path,
                self.openpose_hand_proto_path
            )
            print("OpenPose模型加载成功")
        except Exception as e:
            print(f"OpenPose模型加载失败: {str(e)}")
            self.openpose = None
            self.openpose_checkbox.setChecked(False)

    def load_yolo_model(self):
        """加载YOLOv8模型"""
        try:
            if not os.path.exists(self.yolo_model_path):
                raise FileNotFoundError(f"YOLO模型文件不存在: {self.yolo_model_path}")
            self.yolo = YOLO(self.yolo_model_path)
            print("YOLOv8模型加载成功")
        except Exception as e:
            print(f"YOLOv8模型加载失败: {str(e)}")
            self.yolo = None
            self.yolo_checkbox.setChecked(False)

    def load_models(self):
        """初始化加载模型"""
        if self.yolo_checkbox.isChecked():
            self.load_yolo_model()
        if self.openpose_checkbox.isChecked():
            self.load_openpose_model()

    def update_detection_interval(self, value):
        """更新检测间隔"""
        # 将滑块值转换为实际的检测间隔（0.1-1.0秒）
        self.detection_interval = value / 10.0

    def speak_text(self):
        """播报当前显示的文本"""
        if self.is_speaking:
            print("正在播放其他内容，请稍后再试")
            return
            
        text = self.text_display.text()
        if text and text != "未检测到手语":
            try:
                # 提取"检测到："和"("之间的文本
                start_index = text.find("检测到: ") + len("检测到: ")
                end_index = text.find(" (")
                if start_index != -1 and end_index != -1:
                    speak_content = text[start_index:end_index]
                    self.speak_content_with_translation(speak_content)
            except Exception as e:
                print(f"语音播报失败: {str(e)}")
                self.is_speaking = False

    def speak_sentence(self):
        """播放完整的句子"""
        if self.is_speaking:
            print("正在播放其他内容，请稍后再试")
            return
            
        if self.detected_words:
            try:
                sentence = "".join(self.detected_words)
                self.speak_content_with_translation(sentence)
            except Exception as e:
                print(f"语音播报失败: {str(e)}")
                self.is_speaking = False
        else:
            print("没有检测到任何手语文字")

    def speak_content_with_translation(self, content):
        """使用翻译并播放内容"""
        try:
            if self.tts_engine is None:
                self.init_tts_engine()
                if self.tts_engine is None:
                    return
                    
            self.is_speaking = True
            
            # 获取当前选择的声音
            current_voice = self.voices[self.current_voice_id]
            current_voice_name = current_voice.name.lower()
            voice_info = self.voice_descriptions.get(current_voice_name, {'lang': 'zh'})
            
            # 如果是英文声音，翻译成英文
            if voice_info['lang'] == 'en':
                try:
                    if self.translator is None:
                        print("重新初始化翻译器...")
                        self.translator = Translator(service_urls=['translate.google.com'])
                    
                    # 添加重试机制
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            print(f"尝试翻译文本: {content}")
                            translation = self.translator.translate(text=content, dest='en')
                            if hasattr(translation, 'text'):
                                content = translation.text
                                print(f"翻译成功: {content}")
                                break
                            else:
                                raise Exception("翻译结果无效")
                        except Exception as e:
                            print(f"翻译尝试 {attempt + 1} 失败: {str(e)}")
                            if attempt == max_retries - 1:  # 如果是最后一次尝试
                                print("所有翻译尝试都失败，切换回中文声音")
                                for i, voice in enumerate(self.voices):
                                    if 'chinese' in voice.name.lower():
                                        self.current_voice_id = i
                                        self.tts_engine.setProperty('voice', voice.id)
                                        break
                            else:
                                print("重新初始化翻译器并重试...")
                                self.translator = Translator(service_urls=['translate.google.com'])
                                continue
                except Exception as e:
                    print(f"翻译过程出错: {str(e)}")
                    # 如果翻译失败，切换回中文声音
                    print("翻译失败，切换回中文声音")
                    for i, voice in enumerate(self.voices):
                        if 'chinese' in voice.name.lower():
                            self.current_voice_id = i
                            self.tts_engine.setProperty('voice', voice.id)
                            break
            
            print(f"开始播放内容: {content}")
            self.tts_engine.say(content)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            print(f"语音播报失败: {str(e)}")
        finally:
            self.is_speaking = False

    def update_sentence_display(self):
        """更新完整句子的显示"""
        if self.detected_words:
            sentence = "".join(self.detected_words)
            self.sentence_display.setText(f"完整句子：{sentence}")
        else:
            self.sentence_display.setText("完整句子：")

    def clear_sentence(self):
        """清空已检测的句子"""
        self.detected_words = []
        self.update_sentence_display()
        print("已清空检测到的手语文字")

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        if self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except:
                pass
        event.accept()

    def change_voice(self, index):
        """更改语音引擎的声音"""
        if 0 <= index < len(self.voices):
            try:
                if self.is_speaking:
                    print("正在播放其他内容，请稍后再试")
                    return
                    
                self.tts_engine.setProperty('voice', self.voices[index].id)
                self.current_voice_id = index
                
                # 获取声音信息
                voice = self.voices[index]
                voice_name = voice.name.lower()
                voice_info = self.voice_descriptions.get(voice_name, {'name': voice.name, 'lang': 'zh'})
                print(f"已切换到声音: {voice_info['name']}")
                
            except Exception as e:
                print(f"切换声音失败: {str(e)}")
                self.init_tts_engine()  # 重新初始化TTS引擎

    def open_translation_window(self):
        """打开手语翻译窗口"""
        if self.translation_window is None:
            self.translation_window = TranslationWindow()
        self.translation_window.show()
        self.translation_window.activateWindow()

if __name__ == '__main__':
    try:
        # 设置应用程序属性
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        app = QApplication(sys.argv)
        
        # 检查是否成功创建了QApplication实例
        if not QApplication.instance():
            raise RuntimeError("无法创建QApplication实例")
            
        # 创建并显示主窗口
        window = MainWindow()
        window.show()
        
        # 运行应用程序
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 