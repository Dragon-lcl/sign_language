# 手语识别系统

基于 **YOLOv8** 与 **OpenPose** 的中文手语识别系统，支持 **Web 网页** 与 **Qt 桌面应用** 双端使用。可对图片、视频和摄像头实时画面进行手语识别，标注检测框与人体关键点，并提供语音播报和手语数字人翻译（需自行配置 SDK）。

本项目可识别 35 类常用手语词汇，包括「时间、你、早上、快乐、生日、谢谢、我爱你、叫什么名字」等日常用语。

---

## 目录

- [功能特性](#功能特性)
- [效果演示](#效果演示)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [模型下载与训练](#模型下载与训练)
- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [API 接口](#api-接口)
- [常见问题](#常见问题)
- [许可与说明](#许可与说明)

---

## 功能特性

- **三种识别方式**：上传图片识别、上传视频逐帧识别、调用摄像头实时识别
- **双端应用**：Flask Web 界面 + PyQt5 桌面客户端，功能一致
- **可视化结果**：在手语动作上绘制检测框、中文标签与置信度，叠加 OpenPose 人体骨架关键点
- **语音播报**：识别结果可通过本地 TTS 朗读，支持中英文（英文经翻译）
- **词汇积累**：识别出的词汇自动汇总，可查看和清空
- **手语数字人**：文本转手语数字人演示（可选，需自行申请第三方 SDK token）
- **模块化算法库**：`src/` 中提供检测 → 预处理 → 特征提取 → 分类的完整流水线，支持 YOLOv8 / OpenPose 两种检测器与 SVM / RF / MLP / LSTM 分类器

---

## 效果演示

Web 端提供三个识别页面：图片识别、视频识别、实时识别。识别结果示例（手部检测框 + 中文标签）：

```text
检测框标注：快乐/高兴  0.93
```

页面截图与演示素材见 `static/image/` 与 `templates/` 目录。

---

## 快速开始

### 1. 环境要求

- Python 3.8+
- 推荐 NVIDIA GPU（无 GPU 可用 CPU，但实时识别会较慢）
- 摄像头（仅实时识别需要）

### 2. 安装依赖

```bash
git clone <你的仓库地址>
cd sign_language

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install flask flask-cors pyyaml opencv-python numpy ultralytics torch \
            pillow pyttsx3 googletrans==4.0.0rc1 tqdm scikit-learn tensorflow
pip install -r qt/requirements.txt   # 仅桌面端需要
```

### 3. 准备模型

仓库不包含模型权重（体积过大），需要自行下载或训练。详见[模型下载与训练](#模型下载与训练)。

```bash
python download_models.py --check   # 检查模型是否齐全
python download_models.py           # 下载缺失的 OpenPose / YOLOv8n 模型
```

### 4. 启动应用

```bash
python run.py                       # Web 应用，推荐
python run.py --host 127.0.0.1 --port 5000 --debug
```

浏览器访问 `http://127.0.0.1:5000` 即可使用。

桌面应用：

```bash
cd qt
python main.py
```

---

## 使用说明

### Web 应用

| 页面 | 地址 | 说明 |
| --- | --- | --- |
| 首页 | `/` | 项目介绍与手语数字人入口 |
| 图片识别 | `/image_recognition` | 上传图片，返回识别类别与标注后的图片 |
| 视频识别 | `/video_recognition` | 上传视频，逐帧识别并汇总结果 |
| 实时识别 | `/realtime_recognition` | 调用摄像头实时识别 |
| 关于 | `/about` | 技术介绍 |

上传接口支持 `png / jpg / jpeg / gif / mp4 / avi / mov`，单文件上限 16 MB。可通过表单参数 `conf`（置信度阈值）、`iou`（IoU 阈值）、`speak`（是否语音播报）调整行为。

### 桌面应用

`qt/main.py` 提供图形界面：打开摄像头实时检测、加载图片/视频识别、调节模型参数、识别结果语音播报，以及手语数字人翻译窗口。

> 桌面端需从 `qt/` 目录运行，因为 `from openpose import OpenPose` 引用的是同目录的 `qt/openpose.py`。

### 手语数字人（可选）

数字人功能依赖第三方 SDK（yiyu），仓库不包含任何凭据。如需启用：

1. 自行到服务商申请 token；
2. 填入 `templates/index.html` 中的 `YIYU_SDK_TOKEN`（Web 端）和 `qt/main.py` HTML 模板中的 `YIYU_SDK_TOKEN`（桌面端）。

未配置时页面会显示提示，识别功能不受影响。

---

## 模型下载与训练

应用运行需要以下模型，均**不包含在仓库中**：

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `models/yolo/exp/weights/best.pt` | 应用实际加载的手势识别模型 | 自行训练 / 作者提供 |
| `models/yolo/yolov8n.pt` | 训练初始化权重 | ultralytics 自动下载 |
| `models/openpose/pose_iter_440000.caffemodel` | OpenPose 身体姿态模型 | CMU 官方公开下载 |
| `models/openpose/pose_deploy_linevec.prototxt` | 对应的网络结构 | CMU OpenPose GitHub |
| `models/openpose/hand_pose_deploy.prototxt` | 手部网络结构 | CMU OpenPose GitHub |
| `models/caffe_models/hand/pose_iter_102000.caffemodel` | OpenPose 手部关键点模型 | CMU 官方公开下载 |

### 下载 OpenPose 模型

```bash
python download_models.py            # 自动下载缺失文件并放到正确目录
python download_models.py --check    # 只检查不下载
```

脚本会从 CMU 官方地址下载约 350 MB 的模型文件。若国外地址不可达，会打印候选下载地址，可手动下载后按上表路径放置。

### 训练 YOLO 手势识别模型

`best.pt` 没有公开下载地址，需要自行训练。

**数据集格式**（YOLO 检测格式）：

```text
data/gesture_image/
├── images/train/    训练图片
├── images/val/      验证图片
├── labels/train/    与图片一一对应的 .txt 标注
└── labels/val/
```

每条标注格式为 `类别ID x_center y_center width height`（坐标归一化到 0~1）。共 35 类，类别 ID 与 `data.yaml` 的 `names` 及 `data/vocabulary.json` 对应。

**训练命令**：

```bash
yolo detect train \
  model=models/yolo/yolov8n.pt \
  data=data.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0 \
  project=runs/train \
  name=exp
```

无 GPU 时将 `device` 改为 `cpu`；显存不足时将 `batch` 降到 8。

**部署**：训练完成后把 `runs/train/exp/weights/best.pt` 复制到 `models/yolo/exp/weights/best.pt`，再运行 `python download_models.py --check` 确认齐全。

> 另有一套基于关键点特征 + 传统分类器/LSTM 的识别流水线（`src/recognition/`），输入为按类别分目录的视频，可通过 `SignRecognizer.train(data_dir)` 训练，与 YOLO 方案相互独立。

### 大文件外置（可选）

模型和数据集体积较大，可用环境变量 `SIGN_LANGUAGE_ASSETS_DIR` 指到项目外目录，代码会优先从那里查找资源：

```bash
export SIGN_LANGUAGE_ASSETS_DIR=/path/to/assets   # Windows PowerShell: $env:SIGN_LANGUAGE_ASSETS_DIR = "..."
```

不设置时默认在项目根目录内查找 `models/` 与 `data/`。

---

## 项目结构

```text
sign_language/
├── app.py                  Flask Web 主应用（路由 + 识别 API + 实时视频流）
├── run.py                  启动脚本
├── paths.py                统一路径解析（支持外部资源目录）
├── config.yaml             全局配置（模型路径、类别、服务地址）
├── data.yaml               YOLO 数据集配置
├── download_models.py      模型下载与检查脚本
│
├── src/                    核心算法包
│   ├── detection/          YOLOv8 / OpenPose 检测器
│   ├── data_processing/    预处理与特征提取
│   ├── recognition/        识别器与分类器
│   └── utils/              词汇表、可视化工具
│
├── data/                   数据（仓库内仅保留词汇表 vocabulary.json）
├── models/                 模型目录说明（权重需自行下载/训练）
├── templates/              Web 页面模板
├── static/                 Web 静态资源与上传目录
├── qt/                     PyQt5 桌面应用
└── comn/                   设计与论文材料
```

---

## 技术架构

```text
用户界面层      Web 页面（Bootstrap） / Qt 桌面界面
       ↓
应用服务层      Flask API / PyQt5 业务服务
       ↓
核心功能层      YOLOv8 目标检测 / OpenPose 姿态估计 / TTS / 数字人
       ↓
数据存储层      模型文件 / 词汇表 / 上传与结果文件
```

识别流程：

```text
输入图像/视频帧
   → YOLOv8 检测手势目标与置信度
   → OpenPose 提取身体与双手关键点
   → 绘制检测框、中文标签与骨架
   → 语音播报 / 结果保存
```

Web 端直接使用训练好的 YOLO 检测模型输出类别与置信度，OpenPose 用于骨架可视化；视频识别按帧推理后取出现频率最高的类别作为最终结果。`src/` 中另提供「关键点 → 特征 → 分类器」的完整方案。

---

## API 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 首页 |
| GET | `/about` | 关于页 |
| GET | `/image_recognition` | 图片识别页 |
| GET | `/video_recognition` | 视频识别页 |
| GET | `/realtime_recognition` | 实时识别页 |
| POST | `/api/recognize_image` | 图片识别，返回类别、置信度、结果图 URL |
| POST | `/api/recognize_video` | 视频识别，返回最终预测与结果视频 URL |
| GET | `/video_feed` | MJPEG 实时摄像头视频流 |
| POST | `/api/speak_text` | 文本转语音（`language: zh/en`） |
| POST | `/api/clear_words` | 清空已识别词汇 |
| GET | `/api/get_detected_words` | 获取已识别词汇 |

---

## 常见问题

**Q：启动时报模型文件不存在？**
A：运行 `python download_models.py --check` 查看缺失文件。OpenPose 模型可自动下载；`best.pt` 需自行训练或从作者处获取。

**Q：中文字体显示不正常？**
A：程序会按优先级查找系统字体（Windows 黑体/微软雅黑、macOS 苹方、Linux 文泉驿/Noto）。也可通过环境变量 `SIGN_LANGUAGE_FONT` 指定字体文件路径，或在项目根目录放 `fonts/simhei.ttf`。

**Q：数字人功能没反应？**
A：需在 `templates/index.html` 或 `qt/main.py` 中填入自己的 yiyu SDK token。未配置时会有提示，不影响识别。

**Q：没有 GPU 能用吗？**
A：可以。训练时把 `device` 设为 `cpu`，但速度较慢；识别时自动使用 CPU 推理。

**Q：想识别自己的手语动作？**
A：按 YOLO 格式准备数据集，修改 `data.yaml` 和 `config.yaml` 中的类别定义，重新训练 `best.pt`。

---

## 许可与说明

本项目仅供学习与研究使用。识别模型在特定数据集上训练，跨场景（光照、背景、手语风格差异）效果可能下降，实际使用时建议用自己的数据重新训练或微调。

第三方依赖（YOLOv8 / OpenPose / Flask / PyQt5 等）的许可请遵循各自项目的规定。手语数字人 SDK 为第三方服务，使用前请自行申请账号并遵守其服务条款。
