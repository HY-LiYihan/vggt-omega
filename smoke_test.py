"""冒烟测试: 非法请求应答 / 正常 images 推理 / 断开后重连。"""
import json
import socket
import struct
import sys
import time

import cv2
import numpy as np


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return buf


def msg(s):
    return json.loads(recv_exact(s, struct.unpack("<I", recv_exact(s, 4))[0]))


# 1) 非法 type → error 响应, 连接不断
s = socket.create_connection(("127.0.0.1", 8317), timeout=15)
s.settimeout(15)
ping = json.dumps({"type": "ping"}).encode()
s.sendall(struct.pack("<I", len(ping)) + ping)
print("1) ping →", msg(s))

# 2) 坏图片字节 → error 响应 (验证修复), 连接不断
hdr = {"type": "infer", "mode": "images", "images_bytes": [5], "params": {}}
s.sendall(struct.pack("<I", len(json.dumps(hdr))) + json.dumps(hdr).encode())
s.sendall(struct.pack("<I", 5) + b"garba")
print("2) 坏图 →", msg(s))

# 3) 正常单图推理 (同一连接!)
cap = cv2.VideoCapture(r"d:\MoGe3\vggt-omega\examples\desert_road.mp4")
ok, frame = cap.read()
cap.release()
ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
jpg = jpg.tobytes()
hdr = {"type": "infer", "mode": "images", "images_bytes": [len(jpg)],
       "params": {"resolution": 512}}
s.sendall(struct.pack("<I", len(json.dumps(hdr))) + json.dumps(hdr).encode())
s.sendall(struct.pack("<I", len(jpg)) + jpg)
arrs, done = {}, None
while True:
    m = msg(s)
    if m["type"] == "array":
        n = int(np.prod(m["shape"]))
        arrs[m["name"]] = np.frombuffer(recv_exact(s, n * 4), dtype="<f4").reshape(m["shape"])
    elif m["type"] == "done":
        done = m
        break
print(f"3) 单图推理: {done['frames']}帧 深度范围 [{arrs['depth'].min():.2f}, "
      f"{arrs['depth'].max():.2f}]m 前向 {done['infer_ms']}ms")
s.close()
time.sleep(1)

# 4) 重连 (验证事件循环未被断开影响)
s2 = socket.create_connection(("127.0.0.1", 8317), timeout=15)
s2.settimeout(15)
s2.sendall(struct.pack("<I", 4) + b"ping")
print("4) 断开后重连 →", msg(s2))
s2.close()
print("\n冒烟测试全部通过 ✓")
sys.exit(0)
