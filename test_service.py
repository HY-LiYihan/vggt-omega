r"""VGGT-Omega TCP 服务测试客户端: 验证 images / video 两种模式 + 并发。

用法:
    python test_service.py [--host 127.0.0.1] [--port 8317]
"""
import argparse
import json
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: F401 (确认环境)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接中断")
        buf += chunk
    return buf


def send_frame(sock, data: bytes):
    sock.sendall(struct.pack("<I", len(data)) + data)


def call_infer(host, port, mode, payload: bytes, sizes=None, params=None):
    """发起一次推理请求, 返回 (arrays, done)。"""
    params = params or {}
    header = {"type": "infer", "mode": mode, "params": params}
    if mode == "images":
        header["images_bytes"] = sizes
    else:
        header["video_bytes"] = len(payload)
    t0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=300) as sock:
        sock.settimeout(300)
        send_frame(sock, json.dumps(header).encode())
        send_frame(sock, payload)
        arrays, done = {}, None
        while True:
            msg = json.loads(recv_exact(sock, struct.unpack("<I", recv_exact(sock, 4))[0]))
            t = msg.get("type")
            if t == "status":
                continue
            if t == "array":
                n = int(np.prod(msg["shape"]))
                arrays[msg["name"]] = np.frombuffer(
                    recv_exact(sock, n * 4), dtype="<f4").reshape(msg["shape"])
            elif t == "done":
                done = msg
                done["client_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                break
            elif t == "error":
                raise RuntimeError(f"服务端错误: {msg}")
        return arrays, done


def video_to_jpgs(video_path, num):
    """从视频均匀采样 num 帧, jpg 编码返回 (字节列表, 帧索引)。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pick = set(int(i * (total - 1) / max(1, num - 1)) for i in range(num))
    jpgs, idxs, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in pick:
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            jpgs.append(buf.tobytes())
            idxs.append(i)
        i += 1
    cap.release()
    return jpgs, idxs


def summarize(tag, arrays, done):
    print(f"\n[{tag}] {done['frames']} 帧 {done['width']}x{done['height']} | "
          f"解码 {done['decode_ms']}ms 前向 {done['infer_ms']}ms 客户端全程 {done['client_ms']}ms | "
          f"fov_x {done['fov_x_deg']:.1f}°")
    d = arrays["depth"]
    print(f"  depth {d.shape} 范围 [{d.min():.2f}, {d.max():.2f}]m 均值 {d.mean():.2f}m | "
          f"置信度均值 {arrays['depth_conf'].mean():.2f} | "
          f"extrinsics {arrays['extrinsics'].shape} intrinsics {arrays['intrinsics'].shape}")
    for i in range(min(3, d.shape[0])):
        e = arrays["extrinsics"][i]
        pos = -e[:3, :3].T @ e[:3, 3]
        print(f"  帧{i}: 深度均值 {d[i].mean():.2f}m 相机位置 ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    assert np.isfinite(d).all(), "depth 存在非有限值"
    assert arrays["extrinsics"].shape[1:] == (3, 4) and arrays["intrinsics"].shape[1:] == (3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8317)
    ap.add_argument("--video", default=r"d:\MoGe3\vggt-omega\examples\desert_road.mp4")
    args = ap.parse_args()

    # ---- 模式 1: images (4 张独立 jpg) ----
    jpgs, idxs = video_to_jpgs(args.video, 4)
    arrays, done = call_infer(args.host, args.port, "images", b"".join(jpgs),
                              sizes=[len(j) for j in jpgs],
                              params={"resolution": 512, "preprocess": "balanced"})
    summarize("images 4图", arrays, done)

    # ---- 模式 2: video (整段 mp4, 服务端采样 8 帧) ----
    payload = open(args.video, "rb").read()
    arrays, done = call_infer(args.host, args.port, "video", payload,
                              params={"resolution": 512, "num_frames": 8})
    summarize("video 8帧", arrays, done)

    # ---- 并发: 3 个连接同时请求 ----
    print("\n[并发] 3 连接同时请求 ...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(call_infer, args.host, args.port, "video", payload,
                          params={"resolution": 512, "num_frames": 4}) for _ in range(3)]
        results = [f.result() for f in futs]
    print(f"并发墙钟: {(time.perf_counter() - t0) * 1000:.0f}ms (推理由全局锁串行)")
    for i, (a, d) in enumerate(results):
        print(f"  连接{i}: {d['frames']}帧 前向 {d['infer_ms']}ms 全程 {d['client_ms']}ms "
              f"深度范围 [{a['depth'].min():.2f}, {a['depth'].max():.2f}]m")
        assert np.isfinite(a["depth"]).all()
    print(f"并发墙钟: {(time.perf_counter() - t0) * 1000:.0f}ms (推理由全局锁串行)")
    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    main()
