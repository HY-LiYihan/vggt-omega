"""VGGT-Omega 环境验证: 构建模型 + 随机输入前向（无权重, 仅验证依赖兼容性）。"""
import time

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

print('构建 VGGTOmega 模型 (1B 参数) ...')
t0 = time.perf_counter()
model = VGGTOmega().to('cuda').eval()
print(f'模型构建+上卡: {time.perf_counter() - t0:.1f}s')

n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'参数量: {n_params:.2f}B')

# dummy 输入: 3 帧多视图, 512 分辨率 (balanced 模式 ~624x416 → 用 624x416 测试)
images = torch.randn(1, 3, 3, 416, 624, device='cuda')
with torch.inference_mode():
    t0 = time.perf_counter()
    predictions = model(images)
torch.cuda.synchronize()
print(f'前向耗时: {(time.perf_counter() - t0) * 1000:.0f}ms')

print('输出键:', sorted(predictions.keys()))
for k in ['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'camera_and_register_tokens']:
    if k in predictions:
        v = predictions[k]
        print(f'  {k}: {tuple(v.shape)} {v.dtype}')

extrinsics, intrinsics = encoding_to_camera(predictions['pose_enc'], predictions['images'].shape[-2:])
print(f'extrinsics: {tuple(extrinsics.shape)}, intrinsics: {tuple(intrinsics.shape)}')
print(f'深度范围: [{predictions["depth"].min().item():.3f}, {predictions["depth"].max().item():.3f}]')
print(f'VRAM 峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
print('\n环境验证通过 — 只差权重文件即可真实推理')
