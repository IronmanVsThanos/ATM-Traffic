# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import cv2

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

from mmengine.config import Config
from mmseg.utils import get_classes, get_palette
from mmengine.runner.checkpoint import _load_checkpoint
from atm.utils import init_model
from mmseg.apis import inference_model
import atm
import tqdm
import mmengine
import torch
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("--config", default="./configs/dinov2/atm_dinov2_mask2former_1024x1024_bs4x2.py",
                        help="Path to the training configuration file.")
    parser.add_argument("--checkpoint",
                        default="/data/DL/code/atm/work_dirs/atm_dinov2_mask2former_1024x1024_bs4x2/20241204_164808_train_val/iter_80000.pth",
                        help="Path to the checkpoint file for both the atm and head models.")
    parser.add_argument("--input", default="/data/DL/code/atm/data", help="Path to input video directory.")
    parser.add_argument("--video-suffix", default=(".mp4", ".avi", ".mov"),
                        help="Tuple of video file suffixes to process.")
    parser.add_argument(
        "--backbone",
        default="/data/DL/code/atm/checkpoints/dinov2_converted_1024x1024.pth",
        help="Path to the backbone model checkpoint."
    )
    parser.add_argument("--save_dir", default="work_dirs/show_video_1024_1024",
                        help="Directory to save the output.")
    parser.add_argument("--device", default="cuda:0", help="Device to use for computation.")
    args = parser.parse_args()
    return args


def load_backbone(checkpoint: dict, backbone_path: str) -> None:
    converted_backbone_weight = _load_checkpoint(backbone_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"].update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )
    else:
        checkpoint.update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )


CLASSES = ('road', 'sidewalk', 'building', 'wall', 'railing', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car', 'truck',
           'bus', 'motorcycle', 'bicycle', 'indication line', 'lane line',
           'crosswalk', 'pole', 'traffic light', 'traffic sign')

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [80, 90, 40],
           [180, 165, 180], [107, 142, 35], [152, 251, 152], [70, 130, 180],
           [255, 0, 0], [255, 100, 0], [0, 0, 142],
           [0, 0, 70], [0, 60, 100], [0, 0, 230], [119, 11, 32],
           [250, 170, 160], [250, 200, 160], [250, 240, 180],
           [153, 153, 153], [250, 170, 30], [220, 220, 0]]
classes = CLASSES


def draw_sem_seg(sem_seg: torch.Tensor):
    num_classes = len(classes)
    sem_seg = sem_seg.data.squeeze(0)
    H, W = sem_seg.shape
    ids = torch.unique(sem_seg).cpu().numpy()
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    colors = [torch.tensor(color, dtype=torch.uint8).view(1, 1, 3) for color in colors]
    result = torch.zeros([H, W, 3], dtype=torch.uint8)
    for label, color in zip(labels, colors):
        result[sem_seg == label, :] = color
    return result.cpu().numpy()


def process_video(model, video_path, save_dir):
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        output_path = osp.join(save_dir, 'output_' + osp.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

        temp_path = osp.join(save_dir, f'temp_frame_{os.getpid()}.jpg')

        try:
            for _ in tqdm.tqdm(range(total_frames), desc=f"Processing {osp.basename(video_path)}"):
                ret, frame = cap.read()
                if not ret:
                    break

                # 将OpenCV的BGR格式转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 保存为临时文件
                cv2.imwrite(temp_path, frame)

                # 进行推理
                result = inference_model(model, temp_path)
                pred = draw_sem_seg(result.pred_sem_seg)

                # 调整预测结果大小以匹配原始帧
                pred = cv2.resize(pred, (frame_width, frame_height))

                # 水平拼接原始帧和预测结果
                combined = np.hstack((frame, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)))

                # 写入输出视频
                out.write(combined)

        finally:
            # 清理资源
            if osp.exists(temp_path):
                os.remove(temp_path)
            cap.release()
            out.release()

        print(f"Video saved to {output_path}")

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")


def get_video_files(directory, video_suffix):
    """获取目录中所有指定后缀的视频文件"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_suffix):
                video_files.append(osp.join(root, file))
    return video_files


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if "test_pipeline" not in cfg:
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=False,
                scale=(
                    2048,
                    1024,
                ),
                type="Resize",
            ),
            dict(type="PackSegInputs"),
        ]

    # 初始化模型
    model = init_model(cfg, args.checkpoint, device=args.device)
    model = model.cuda(args.device)
    state_dict = model.state_dict()
    load_backbone(state_dict, args.backbone)
    model.load_state_dict(state_dict)

    # 创建保存目录
    mmengine.mkdir_or_exist(args.save_dir)

    # 获取所有视频文件
    video_files = get_video_files(args.input, args.video_suffix)
    print(f"Found {len(video_files)} videos to process")

    # 处理每个视频
    for video_file in video_files:
        process_video(model, video_file, args.save_dir)


if __name__ == "__main__":
    main()