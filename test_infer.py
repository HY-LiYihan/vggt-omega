r"""VGGT-Omega 真实推理测试: 从示例视频均匀采样多帧 → 相机位姿 + 深度预测。

用法:
    python test_infer.py --checkpoint D:\MoGe3\checkpoints\vggt-omega\vggt_omega_1b_512.pt
    python test_infer.py --checkpoint ... --video examples/forest_road.mp4 --num-frames 16
"""
import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def sample_video_frames(video_path: str, num_frames: int):
    """从视频均匀采样 num 帧, 存为临时 jpg 返回路径列表。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f'无法读取视频: {video_path}')
    pick = set(int(i * (total - 1) / max(1, num_frames - 1)) for i in range(num_frames))
    tmpdir = Path(tempfile.mkdtemp(prefix='vggt_frames_'))
    names = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in pick:
            p = tmpdir / f'frame_{i:05d}.jpg'
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(str(p))
        i += 1
    cap.release()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='vggt_omega_1b_512.pt 路径')
    ap.add_argument('--video', default=str(Path(__file__).parent / 'examples' / 'desert_road.mp4'))
    ap.add_argument('--num-frames', type=int, default=8)
    ap.add_argument('--resolution', type=int, default=512)
    args = ap.parse_args()

    names = sample_video_frames(args.video, args.num_frames)
    print(f'视频采样: {len(names)} 帧 @ {args.video}')

    print('加载模型 ...')
    t0 = time.perf_counter()
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    print(f'模型加载: {time.perf_counter() - t0:.1f}s | VRAM {torch.cuda.memory_reserved() / 1024**3:.2f} GB')

    images = load_and_preprocess_images(names, image_resolution=args.resolution).to('cuda')
    print(f'输入张量: {tuple(images.shape)}')

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictions = model(images)
        torch.cuda.synchronize()
    print(f'前向耗时: {(time.perf_counter() - t0) * 1000:.0f}ms')

    extrinsics, intrinsics = encoding_to_camera(predictions['pose_enc'], predictions['images'].shape[-2:])
    depth = predictions['depth'].squeeze(-1)
    conf = predictions['depth_conf']

    print(f'\n=== {len(names)} 帧重建结果 ===')
    print(f'extrinsics: {tuple(extrinsics.shape)} | intrinsics: {tuple(intrinsics.shape)}')
    for i in range(len(names)):
        d = depth[0, i].cpu().numpy()
        c = conf[0, i].cpu().numpy()
        e = extrinsics[0, i]
        pos = -e[:3, :3].T @ e[:3, 3]
        print(f'  帧{i}: 深度 [{d.min():.2f}, {d.max():.2f}]m 均值 {d.mean():.2f}m | '
              f'置信度 {c.mean():.3f} | 相机位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})')

    print(f'\nVRAM 峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')


if __name__ == '__main__':
    main()
