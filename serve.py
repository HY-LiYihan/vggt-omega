r"""VGGT-Omega TCP 推理服务

用法:
    python serve.py [--checkpoint <pt路径>] [--host 0.0.0.0] [--port 8317]

协议 (与 MoGe 服务一致的帧格式: 4 字节小端 uint32 长度前缀 + 载荷):

  请求 (单帧消息 = JSON 头 + 二进制负载):
    {
      "type": "infer",
      "mode": "images" | "video",
      "images_bytes": [n1, n2, ...],   # images 模式: 各张图片(jpg/png)字节数
      "video_bytes": M,                 # video 模式: 整段视频(mp4等)字节数
      "params": {
        "resolution": 512,              # image_resolution, 须为 16 的倍数
        "preprocess": "balanced",       # balanced | max_size
        "num_frames": 8                # video 模式均匀采样帧数
      }
    }
    JSON 头之后紧跟二进制负载 (图片字节顺序拼接 / 完整视频字节)。

  响应 (同一连接按序):
    {"type":"status","stage":"received",...}
    {"type":"status","stage":"inferring"}
    {"type":"array","name":"depth","shape":[N,H,W],"dtype":"float32"} + 原始字节
    {"type":"array","name":"depth_conf","shape":[N,H,W],...} + 字节
    {"type":"array","name":"extrinsics","shape":[N,3,4],...} + 字节
    {"type":"array","name":"intrinsics","shape":[N,3,3],...} + 字节
    {"type":"done","frames":N,"height":H,"width":W,"fov_x_deg":...,...}

  连接为长连接, 可连续多次请求; 出错返回 {"type":"error","stage":...}。
"""
import argparse
import asyncio
import io
import json
import logging
import struct
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
    _max_size_target_shape,
    _pad_images_to_common_size,
)
from vggt_omega.utils.pose_enc import encoding_to_camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vggt-serve")

DEFAULT_CHECKPOINT = r"D:\MoGe3\checkpoints\vggt-omega\vggt_omega_1b_512.pt"
MAX_PAYLOAD = 512 * 1024 * 1024  # 单请求二进制负载上限
MAX_IMAGES = 64                  # images 模式最多图片数
MAX_FRAMES = 64                  # video 模式最多采样帧数
PATCH_SIZE = 16

# ---------------- 帧协议 ----------------


async def recv_msg(reader: asyncio.StreamReader) -> dict:
    (length,) = struct.unpack("<I", await reader.readexactly(4))
    if length > MAX_PAYLOAD:
        raise ValueError(f"消息头过长: {length}")
    return json.loads((await reader.readexactly(length)).decode("utf-8"))


async def send_json(writer: asyncio.StreamWriter, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    writer.write(struct.pack("<I", len(data)) + data)
    await writer.drain()


async def send_array(writer: asyncio.StreamWriter, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    await send_json(writer, {
        "type": "array", "name": name,
        "shape": list(arr.shape), "dtype": "float32",
    })
    writer.write(arr.tobytes())
    await writer.drain()


# ---------------- 预处理 (与 load_fn.py 逻辑一致, 内存版) ----------------


def preprocess_pils(pils: list, preprocess: str = "balanced", image_resolution: int = 512) -> torch.Tensor:
    """PIL 图像列表 → (N,3,H,W) 张量, 复用官方 load_fn 的形状策略。"""
    to_tensor = TF.ToTensor()
    images, shapes = [], set()
    for image in pils:
        image = _crop_to_supported_aspect_ratio(image)
        w, h = image.size
        aspect = h / max(w, 1)
        if preprocess == "balanced":
            th, tw = _balanced_target_shape(aspect, image_resolution, PATCH_SIZE)
        else:
            th, tw = _max_size_target_shape(aspect, image_resolution, PATCH_SIZE)
        image = image.resize((tw, th), Image.Resampling.BICUBIC)
        t = to_tensor(image)
        shapes.add((t.shape[1], t.shape[2]))
        images.append(t)
    if len(shapes) > 1:
        images = _pad_images_to_common_size(images, shapes)
    return torch.stack(images)


def decode_images(payload: bytes, sizes: list) -> list:
    """二进制负载按 sizes 切分并解码为 RGB PIL 图像列表。"""
    if sum(sizes) != len(payload):
        raise ValueError(f"images_bytes 总和 {sum(sizes)} 与负载长度 {len(payload)} 不符")
    pils, off = [], 0
    for n in sizes:
        img = Image.open(io.BytesIO(payload[off:off + n]))
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        pils.append(img.convert("RGB"))
        off += n
    return pils


def decode_video_sample(payload: bytes, num_frames: int) -> list:
    """视频字节 → 均匀采样 num_frames 帧的 RGB PIL 图像列表。"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(payload)
        path = f.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError("无法读取视频帧 (CAP_PROP_FRAME_COUNT=0)")
        k = max(1, min(num_frames, total))
        pick = sorted(set(int(i * (total - 1) / max(1, k - 1)) for i in range(k)))
        pils, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in pick:
                pils.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            i += 1
        cap.release()
        if not pils:
            raise RuntimeError("视频解码得到 0 帧")
        return pils
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------- 服务 ----------------


class VGGTService:
    def __init__(self, checkpoint: str):
        log.info("加载 VGGT-Omega 模型: %s", checkpoint)
        t0 = time.perf_counter()
        self.model = VGGTOmega().to("cuda").eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        log.info("模型就绪: %.1fs, 显存 %.2f GB", time.perf_counter() - t0,
                 torch.cuda.memory_reserved() / 1024 ** 3)
        self.infer_lock = asyncio.Lock()

    async def infer(self, tensor: torch.Tensor) -> tuple:
        """GPU 前向 (全局串行), 返回 (predictions, infer_ms)。"""
        async with self.infer_lock:
            loop = asyncio.get_running_loop()

            def _forward():
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    preds = self.model(tensor)
                    torch.cuda.synchronize()
                    return preds, (time.perf_counter() - t0) * 1000

            return await loop.run_in_executor(None, _forward)

    async def handle_request(self, reader, writer, req: dict, payload: bytes):
        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        params = req.get("params") or {}
        image_resolution = int(params.get("resolution", 512))
        preprocess = str(params.get("preprocess", "balanced"))
        if image_resolution % PATCH_SIZE != 0:
            raise ValueError(f"resolution 须为 {PATCH_SIZE} 的倍数: {image_resolution}")
        if preprocess not in ("balanced", "max_size"):
            raise ValueError(f"preprocess 须为 balanced|max_size: {preprocess}")

        mode = req.get("mode", "images")
        if mode == "images":
            sizes = [int(n) for n in req["images_bytes"]]
            if not (1 <= len(sizes) <= MAX_IMAGES):
                raise ValueError(f"图片数须在 1~{MAX_IMAGES}: {len(sizes)}")
            num_frames = len(sizes)
            try:
                pils = await loop.run_in_executor(None, decode_images, payload, sizes)
            except Exception as e:
                raise ValueError(f"图片解码失败: {e}") from e
        elif mode == "video":
            num_frames = int(params.get("num_frames", 8))
            if not (1 <= num_frames <= MAX_FRAMES):
                raise ValueError(f"num_frames 须在 1~{MAX_FRAMES}: {num_frames}")
            try:
                pils = await loop.run_in_executor(None, decode_video_sample, payload, num_frames)
            except Exception as e:
                raise ValueError(f"视频解码失败: {e}") from e
            num_frames = len(pils)
        else:
            raise ValueError(f"mode 须为 images|video: {mode}")

        t_dec = time.perf_counter()
        try:
            tensor = await loop.run_in_executor(
                None, preprocess_pils, pils, preprocess, image_resolution)
        except Exception as e:
            raise ValueError(f"预处理失败: {e}") from e
        tensor = tensor.unsqueeze(0).to("cuda")  # (1, N, 3, H, W)
        decode_ms = (time.perf_counter() - t_start) * 1000

        await send_json(writer, {"type": "status", "stage": "inferring", "frames": num_frames})
        preds, infer_ms = await self.infer(tensor)

        h, w = tensor.shape[-2:]
        extr, intr = encoding_to_camera(preds["pose_enc"], (h, w))

        def _to_numpy(t):  # (S,N,H,W[,1]) → (N,H,W)
            if t.dim() == 5:
                t = t[..., 0]
            return t[0].float().cpu().numpy()

        arrays = {
            "depth": _to_numpy(preds["depth"]),        # (N,H,W)
            "depth_conf": _to_numpy(preds["depth_conf"]),  # (N,H,W), head 已 squeeze 为 4 维
            "extrinsics": extr[0].float().cpu().numpy(),                      # (N,3,4)
            "intrinsics": intr[0].float().cpu().numpy(),                     # (N,3,3)
        }
        for name, arr in arrays.items():
            await send_array(writer, name, arr)

        pose_enc = preds["pose_enc"][0].float().cpu().numpy()  # (N,9): fov_h=7, fov_w=8
        await send_json(writer, {
            "type": "done",
            "mode": mode, "frames": num_frames,
            "height": int(h), "width": int(w),
            "fov_x_deg": float(np.degrees(pose_enc[:, 8].mean())),
            "fov_y_deg": float(np.degrees(pose_enc[:, 7].mean())),
            "decode_ms": round(decode_ms, 1),
            "infer_ms": round(infer_ms, 1),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
        })

    async def handle_conn(self, reader, writer):
        peer = writer.get_extra_info("peername")
        log.info("连接: %s", peer)
        recv_task = None  # 复用读取任务, 避免 wait_for 取消时丢失已读字节
        try:
            while True:
                if recv_task is None:
                    recv_task = asyncio.create_task(recv_msg(reader))
                done, _ = await asyncio.wait({recv_task}, timeout=0.5)
                if not done:
                    continue  # 空闲轮询 (保持 Ctrl+C 响应), 读取任务保持挂起
                try:
                    req = recv_task.result()
                except asyncio.IncompleteReadError:
                    break  # 客户端已断开
                except ValueError as e:  # 含 json.JSONDecodeError
                    recv_task = None
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": f"非法 JSON: {e}"})
                    continue
                recv_task = None
                if req.get("type") == "quit":
                    await send_json(writer, {"type": "bye"})
                    break
                if req.get("type") != "infer":
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": f"未知 type: {req.get('type')}"})
                    continue
                try:
                    if req.get("mode", "images") == "images":
                        sizes = [int(n) for n in req["images_bytes"]]
                        payload_len = sum(sizes)
                    else:
                        payload_len = int(req["video_bytes"])
                    if payload_len > MAX_PAYLOAD:
                        raise ValueError(f"负载超限 {payload_len} > {MAX_PAYLOAD}")
                    (length,) = struct.unpack("<I", await reader.readexactly(4))
                    if length != payload_len:
                        raise ValueError("二进制负载帧长度与 JSON 声明不符")
                    payload = await reader.readexactly(payload_len) if payload_len else b""
                    await send_json(writer, {"type": "status", "stage": "received",
                                             "payload_bytes": payload_len})
                    await self.handle_request(reader, writer, req, payload)
                except (KeyError, ValueError, RuntimeError) as e:
                    log.warning("请求失败(%s): %s", peer, e)
                    await send_json(writer, {"type": "error", "stage": "request",
                                             "message": str(e)})
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            if recv_task and not recv_task.done():
                recv_task.cancel()
            writer.close()
            log.info("断开: %s", peer)

    async def serve(self, host: str, port: int):
        server = await asyncio.start_server(self.handle_conn, host, port)
        log.info("VGGT-Omega 服务监听 %s:%d (images/video 两种模式)", host, port)
        async with server:
            await server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="VGGT-Omega TCP 推理服务")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8317)
    args = ap.parse_args()

    svc = VGGTService(args.checkpoint)
    try:
        asyncio.run(svc.serve(args.host, args.port))
    except KeyboardInterrupt:
        log.info("服务已停止")


if __name__ == "__main__":
    main()
