"""协议 shape 验证: depth/depth_conf/extrinsics/intrinsics 必须与 done 声明的 frames/H/W 一致。"""
import json
import socket
import struct
import sys

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


def call(host, port, images, params):
    header = {"type": "infer", "mode": "images",
              "images_bytes": [len(b) for b in images], "params": params}
    with socket.create_connection((host, port), timeout=180) as s:
        s.settimeout(180)
        for data in (json.dumps(header).encode(), b"".join(images)):
            s.sendall(struct.pack("<I", len(data)) + data)
        arrays, done = {}, None
        while True:
            m = json.loads(recv_exact(s, struct.unpack("<I", recv_exact(s, 4))[0]))
            if m["type"] == "array":
                n = int(np.prod(m["shape"]))
                arrays[m["name"]] = np.frombuffer(
                    recv_exact(s, n * 4), dtype="<f4").reshape(m["shape"])
            elif m["type"] == "done":
                return arrays, m
            elif m["type"] == "error":
                raise RuntimeError(m)


def check(tag, arrays, done):
    N, H, W = done["frames"], done["height"], done["width"]
    assert arrays["depth"].shape == (N, H, W), f"depth {arrays['depth'].shape} != {(N, H, W)}"
    assert arrays["depth_conf"].shape == (N, H, W), \
        f"depth_conf {arrays['depth_conf'].shape} != {(N, H, W)}"
    assert arrays["extrinsics"].shape == (N, 3, 4)
    assert arrays["intrinsics"].shape == (N, 3, 3)
    assert np.isfinite(arrays["depth"]).all() and np.isfinite(arrays["depth_conf"]).all()
    print(f"[{tag}] {N}帧 {W}x{H} | depth {arrays['depth'].shape} conf {arrays['depth_conf'].shape} "
          f"conf范围 [{arrays['depth_conf'].min():.2f}, {arrays['depth_conf'].max():.2f}] ✓")


cap = cv2.VideoCapture(r"d:\MoGe3\vggt-omega\examples\desert_road.mp4")
frames, idxs, i = [], {5, 40, 75}, 0
while len(frames) < 3:
    ok, f = cap.read()
    if i in idxs:
        _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(jpg.tobytes())
    i += 1
cap.release()

a, d = call("127.0.0.1", 8317, frames[:1], {"resolution": 512})
check("单图", a, d)
a, d = call("127.0.0.1", 8317, frames, {"resolution": 512})
check("3图", a, d)
a, d = call("127.0.0.1", 8317, frames, {"resolution": 256, "preprocess": "max_size"})
check("3图@256", a, d)
print("\nshape 验证全部通过 ✓")
sys.exit(0)
