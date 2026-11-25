# 手语识别系统

这是一个基于YOLOv8和OpenPose的手语识别系统，使用PyQt5构建图形界面。

## 功能特点

- 实时摄像头手语检测
- 姿态估计和关键点检测
- 友好的图形用户界面
- 支持35种手语词汇识别

## 安装要求

- Python 3.8+
- CUDA支持的GPU（推荐）
- 摄像头

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository_url]
cd sign_language/qt
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 确保模型文件存在：
- YOLO模型：`models/yolo/exp/weights/best.pt`
- OpenPose模型：`models/openpose/`目录下的所有模型文件

## 使用方法

1. 运行程序：
```bash
python main.py
```

2. 点击"开始"按钮启动摄像头
3. 系统会自动检测手语并显示识别结果
4. 点击"停止"按钮关闭摄像头

## 配置说明

配置文件 `config.yaml` 包含以下主要设置：

- 模型路径配置
- 摄像头设置
- 识别参数
- 类别名称映射

## 注意事项

- 确保摄像头正确连接
- 保持适当的光照条件
- 手语动作要清晰可见
- 建议在GPU环境下运行以获得更好的性能 