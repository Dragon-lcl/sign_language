#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载手语识别系统所需的预训练模型。

注意：
1. OpenPose 模型来自 CMU 官方，可从公开链接下载；
2. YOLOv8n 预训练权重由 ultralytics 自动下载，仅用于训练；
3. Web/桌面应用实际加载的 models/yolo/exp/weights/best.pt 是本项目在
   手势数据集上训练得到的成果，没有公开下载地址，需自行训练或单独获取
   （训练方法见 README「YOLO 模型如何训练」一节）。

用法：
    python download_models.py            # 下载缺失的模型
    python download_models.py --check    # 只检查模型是否齐全
"""

import argparse
import os
import shutil
import urllib.request

from paths import resolve_config_path

CMU_BASE = "http://posefs1.perception.cs.cmu.edu/OpenPose/models"
GH_BASE = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models"

# (说明, 相对路径, 候选下载地址列表)
DOWNLOADS = [
    (
        "OpenPose COCO 身体姿态模型",
        "models/openpose/pose_iter_440000.caffemodel",
        [f"{CMU_BASE}/pose/coco/pose_iter_440000.caffemodel"],
    ),
    (
        "OpenPose COCO 网络结构定义",
        "models/openpose/pose_deploy_linevec.prototxt",
        [
            f"{GH_BASE}/pose/coco/pose_deploy_linevec.prototxt",
            f"{GH_BASE}/pose/coco/pose_deploy.prototxt",
        ],
    ),
    (
        "OpenPose 手部关键点模型",
        "models/caffe_models/hand/pose_iter_102000.caffemodel",
        [f"{CMU_BASE}/hand/pose_iter_102000.caffemodel"],
    ),
    (
        "OpenPose 手部网络结构定义",
        "models/openpose/hand_pose_deploy.prototxt",
        [f"{GH_BASE}/hand/pose_deploy.prototxt"],
    ),
]

# 应用运行必需的模型（含需要自行训练的 best.pt）
RUNTIME_FILES = [
    "models/yolo/exp/weights/best.pt",
    "models/openpose/pose_iter_440000.caffemodel",
    "models/openpose/pose_deploy_linevec.prototxt",
    "models/caffe_models/hand/pose_iter_102000.caffemodel",
    "models/openpose/hand_pose_deploy.prototxt",
]


def download_url(url, output_path):
    """带进度条下载单个文件。"""
    from tqdm import tqdm  # 延迟导入：--check 模式不需要 tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".part"
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
                             desc=os.path.basename(output_path)) as bar:
        urllib.request.urlretrieve(url, filename=tmp_path, reporthook=bar.update_to)
    shutil.move(tmp_path, output_path)


def check_models(verbose=True):
    """检查运行所需模型是否齐全，返回缺失文件列表。"""
    missing = []
    for rel in RUNTIME_FILES:
        path = resolve_config_path(rel)
        if os.path.exists(path):
            if verbose:
                print(f"  [OK]   {rel}  ->  {path}")
        else:
            missing.append(rel)
            if verbose:
                print(f"  [缺失] {rel}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="下载手语识别系统所需模型")
    parser.add_argument("--check", action="store_true", help="只检查模型是否齐全，不下载")
    args = parser.parse_args()

    print("检查模型文件：")
    check_models()

    if args.check:
        missing = check_models(verbose=False)
        if missing:
            print(f"\n缺少 {len(missing)} 个模型文件，请运行 python download_models.py 下载。")
        else:
            print("\n所有运行所需模型均已就绪。")
        return

    print("\n开始下载 OpenPose 模型...")
    for desc, rel_path, urls in DOWNLOADS:
        target = resolve_config_path(rel_path)
        if os.path.exists(target):
            print(f"\n{desc} 已存在：{target}，跳过。")
            continue

        print(f"\n下载 {desc} -> {target}")
        success = False
        for url in urls:
            try:
                download_url(url, target)
                print(f"  下载完成：{url}")
                success = True
                break
            except Exception as exc:
                print(f"  下载失败（{url}）：{exc}")
        if not success:
            print(f"  请手动下载并放到：{target}")
            print(f"  候选地址：{', '.join(urls)}")

    print("\n下载 YOLOv8n 预训练权重（仅训练需要）...")
    try:
        from ultralytics import YOLO
        target = resolve_config_path("models/yolo/yolov8n.pt")
        if os.path.exists(target):
            print(f"  已存在：{target}，跳过。")
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            model = YOLO("yolov8n.pt")  # ultralytics 会自动下载到当前目录
            shutil.copy(model.ckpt_path, target)
            print(f"  已保存到：{target}")
    except Exception as exc:
        print(f"  下载 YOLOv8n 失败：{exc}")

    print("\n最终检查结果：")
    missing = check_models(verbose=False)
    if missing:
        print("\n注意：models/yolo/exp/weights/best.pt 是训练产物，不会自动下载。")
        print("如需自己训练，请按 README「YOLO 模型如何训练」一节操作，")
        print("或把训练好的 best.pt 放到该路径。")


if __name__ == "__main__":
    main()
