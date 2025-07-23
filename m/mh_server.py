#mh_server.py
import socket
import struct
import cv2
import numpy as np
import time
import json
from ultralytics import YOLO

# --- 설정 ---
FOCAL_LENGTH = 600

REAL_HEIGHTS = {
    "person": 1.6, "car": 1.5, "bus": 3.2, "truck": 3.4,
    "motorbike": 1.4, "bicycle": 1.2, "vehicle": 1.5,
    "big vehicle": 3.5, "bike": 1.2, "human": 1.7,
    "animal": 0.5, "obstacle": 1.0
}

REAL_WIDTHS = {
    "person": 0.5, "car": 1.8, "bus": 2.5, "truck": 2.5,
    "motorbike": 0.8, "bicycle": 0.7, "vehicle": 1.8,
    "big vehicle": 2.5, "bike": 0.5, "human": 0.5,
    "animal": 0.6, "obstacle": 1.0
}

def estimate_distance(h, w, label):
    try:
        dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h
        dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w
        return (dist_h + dist_w) / 2
    except:
        return -1

# YOLO 모델 로드
model = YOLO("/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt")
model.to('cuda')

# 폴리곤 영역
red_polygon = np.array([[198, 479], [241, 403], [458, 403], [504, 479]], np.int32)
yellow_polygon = np.array([[94, 475], [164, 388], [520, 388], [585, 475]], np.int32)

# 소켓 설정
HOST = '0.0.0.0'
PORT = 9888

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("클라이언트와 연결이 끊어졌습니다.")
        data += more
    return data

# 소켓 수신 대기
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"INFO: 서버가 {HOST}:{PORT}에서 대기 중입니다...")
conn, addr = server_socket.accept()
print(f"INFO: 클라이언트 연결됨: {addr}")

prev_time = time.time()

while True:
    try:
        # 프레임 수신
        length_buf = recvall(conn, 4)
        if not length_buf:
            break
        frame_len = struct.unpack('>I', length_buf)[0]
        frame_data = recvall(conn, frame_len)
        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame, conf=0.3)[0]

        danger_objects = []  # 클라이언트에 전송할 정보

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_id = int(box.cls)
            label = model.names[label_id]

            if label not in REAL_HEIGHTS:
                continue

            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue

            cx, cy = (x1 + x2) // 2, y2
            in_red = cv2.pointPolygonTest(red_polygon, (cx, cy), False) >= 0
            in_yellow = cv2.pointPolygonTest(yellow_polygon, (cx, cy), False) >= 0
            if not (in_red or in_yellow):
                continue

            distance = estimate_distance(h, w, label)

            # 전송용 데이터 구성
            danger_objects.append({
                "label": label,
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "distance": round(distance, 2),
                "zone": "red" if in_red else "yellow"
            })

            # 디버깅용 시각화
            color = (0, 0, 255) if in_red else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 폴리곤 시각화
        cv2.polylines(frame, [red_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.polylines(frame, [yellow_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        # FPS 측정
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 클라이언트에게 전송 (JSON 직렬화)
        response_json = json.dumps(danger_objects)
        conn.sendall(response_json.encode('utf-8'))

        # 디버깅 화면 출력
        cv2.imshow("YOLO + Polygon Danger Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print(f"[에러] {e}")
        break

conn.close()
server_socket.close()
cv2.destroyAllWindows()
