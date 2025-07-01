# rb2_warning_display.py
import socket
import struct
import json
import cv2
import numpy as np

SERVER_IP = '192.168.3.28'  # 내 노트북 서버 주소
PORT = 9999

def receive_object(sock):
    meta_len_buf = sock.recv(4)
    if not meta_len_buf:
        return None
    meta_len = struct.unpack(">I", meta_len_buf)[0]

    meta_data = b''
    while len(meta_data) < meta_len:
        meta_data += sock.recv(meta_len - len(meta_data))
    meta = json.loads(meta_data.decode())

    img_bytes = b''
    while len(img_bytes) < meta['img_size']:
        img_bytes += sock.recv(meta['img_size'] - len(img_bytes))

    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img, meta

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("[✅] Connected to server for warning object display")

while True:
    result = receive_object(sock)
    if result is None:
        break

    img, meta = result
    label = meta['class']
    distance = meta['distance']

    cv2.putText(img, f"{label} {distance}m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("⚠️ WARNING OBJECT", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
