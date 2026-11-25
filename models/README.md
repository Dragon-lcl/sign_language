# 模型目录说明

本目录用于存放运行所需的模型权重。由于模型文件体积很大（约 1.2 GB），
不包含在 Git 仓库中，已移到项目外的 `E:/sign_language_assets/models/`。

## 两种恢复方式

### 方式一：直接链接外部目录（不可用）

设置环境变量后，代码会优先从外部目录查找模型：

```bash
# Git Bash / Linux / macOS
export SIGN_LANGUAGE_ASSETS_DIR=E:/sign_language_assets

# Windows PowerShell
$env:SIGN_LANGUAGE_ASSETS_DIR = "E:/sign_language_assets"
```

### 方式二：重新下载 / 训练到本目录

```bash
python download_models.py
```

注意：`models/yolo/exp/weights/best.pt` 是本项目在自己数据集上训练得到的成果，
没有公开下载地址，需要从数据集重新训练（见根目录 README「YOLO 模型如何训练」），
或从项目作者处单独获取后放到该路径。

## 需要的模型文件

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `yolo/exp/weights/best.pt` | Web/桌面应用实际加载的手势识别模型 | 自行训练 |
| `yolo/yolov8n.pt` | 训练用的 YOLOv8n 预训练权重 | ultralytics 自动下载 |
| `openpose/pose_iter_440000.caffemodel` | OpenPose COCO 身体姿态模型 | CMU 官方 |
| `openpose/pose_deploy_linevec.prototxt` | 对应的网络结构 | CMU 官方 |
| `openpose/hand_pose_deploy.prototxt` | 手部网络结构 | CMU 官方 |
| `caffe_models/hand/pose_iter_102000.caffemodel` | 手部关键点模型 | CMU 官方 |
