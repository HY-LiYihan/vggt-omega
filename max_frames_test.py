"""测试 1080p 视频在服务上的最大可推理帧数。

用法:
  python max_frames_test.py --resolution 512 --frames 8,16,32,48,64
  python max_frames_test.py --resolution 1088 --frames 2,4,8,12,16

video 模式调用本地服务, 每档记录: 前向耗时 / 显存峰值 / 成败。
超时(默认180s)视为该档不可用并停止。
"""
import argparse
import json
import socket
import struct
import subprocess
import threading
import time

VIDEO = r"c:\Users\HYL\.trae-cn\attachments\6aac2ac3a02ade2173965fad\4ca063b2-86f5-4b6d-b94c-df76d6fa566a_9c1cb889-2aaf-4991-92a8-9874961f2ddb_微信视频2026-09-19_024627_602.mp4"

vram_peak = 0
stop_flag = False


def sample_vram():
    global vram_peak
    while not stop_flag:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines()[0]
        vram_peak = max(vram_peak, int(out))
        time.sleep(0.2)


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return buf


def call_video(num_frames, resolution, timeout):
    global vram_peak
    vb = open(VIDEO, "rb").read()
    params = {"num_frames": num_frames, "resolution": resolution}
    header = {"type": "infer", "mode": "video", "video_bytes": len(vb), "params": params}
    vram_peak = 0
    t0 = time.perf_counter()
    try:
        with socket.create_connection(("127.0.0.1", 8317), timeout=timeout) as s:
            s.settimeout(timeout)
            for d in (json.dumps(header).encode(), vb):
                s.sendall(struct.pack("<I", len(d)) + d)
            while True:
                m = json.loads(recv_exact(s, struct.unpack("<I", recv_exact(s, 4))[0]))
                if m["type"] == "array":
                    recv_exact(s, int(__import__("numpy").prod(m["shape"])) * 4)
                elif m["type"] == "done":
                    return {"ok": True, "shape": (m["height"], m["width"]),
                            "decode_ms": m["decode_ms"], "infer_ms": m["infer_ms"],
                            "wall_s": time.perf_counter() - t0}
                elif m["type"] == "error":
                    return {"ok": False, "err": m["message"][:120]}
    except (socket.timeout, TimeoutError):
        return {"ok": False, "err": f"超时(>{timeout}s), 服务端显存换页卡死"}


def main():
    global stop_flag
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--frames", default="8,16,32,48,64")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    threading.Thread(target=sample_vram, daemon=True).start()
    time.sleep(1)
    print(f"===== resolution={args.resolution} =====")
    for nf in [int(x) for x in args.frames.split(",")]:
        r = call_video(nf, args.resolution, args.timeout)
        if r["ok"]:
            print(f"  {nf:3d} 帧 ✓  {r['shape'][1]}x{r['shape'][0]} | 解码 {r['decode_ms']/1000:.1f}s "
                  f"前向 {r['infer_ms']/1000:.1f}s 全程 {r['wall_s']:.1f}s | VRAM峰值 {vram_peak}MB")
        else:
            print(f"  {nf:3d} 帧 ✗  {r['err']}")
            break
    stop_flag = True


if __name__ == "__main__":
    main()
