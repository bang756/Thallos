# server_receiver.py
import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO #000
import time #000

HOST = '0.0.0.0'
PORT = 8888

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((HOST, PORT))
server_sock.listen(1)
conn, addr = server_sock.accept()
print(f"Connected by {addr}")

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

############### #000
# 🔧 하이퍼파라미터
FOCAL_LENGTH = 700 # focal length in pixels (조정 가능)
resize_width, resize_height = 1200, 800  # 사이즈 변환 
# SCALE = 0.6 # 프레임 축소 비율 (0.5 = 50%)

# 🧍 실제 객체 크기 (meter 단위)
REAL_HEIGHTS = {
    "person": 1.6,
    "car": 1.5,
    "bus": 3.2,
    "truck": 3.4,
    "motorbike": 1.4,
    "bicycle": 1.2,
    "vehicle": 1.5, #우리가 학습시킨 모델의 클라스 추가 #004
    "big vehicle": 3.5,
    "bike": 1.2,
    "human": 1.7,
    "animal": 0.5,
    "obstacle":1.0
}

REAL_WIDTHS = {
    "person": 0.5,
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.5,
    "motorbike": 0.8,
    "bicycle": 0.7,
    "vehicle": 1.8, #우리가 학습시킨 모델의 클라스 추가 #004
    "big vehicle": 2.5,
    "bike": 0.5,
    "human": 0.5,
    "animal": 0.6,
    "obstacle":1.0
}

# 🔍 (하이브리드) 거리 추정 함수
def estimate_distance(h, w, label):
    try:
        dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h
        dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w
        return (dist_h + dist_w) / 2
    except:
        return -1

# ▶️ YOLO 모델 로드
#model = YOLO("yolov8n.pt")
#직접 훈련시킨 최종 모델
model = YOLO("/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt")

# GPU 사용을 위한 device 설정 #006
# ultralytics 라이브러리에서 YOLO 모델을 GPU로 실행하려면 device 설정을 model() 호출 시 to() 메서드를 사용하여 GPU로 전환해야 합니다.
model.to('cuda')  # 'cuda'를 지정하여 모델을 GPU로 전송

# 🚧 위험 폴리곤 설정 (해상도에 맞게 조정 가능) #왼쪽 아래,왼쪽 위, 오른쪽 위, 오른쪽 아래 [x, y]
red_polygon = np.array([[237, 673], [363, 543], [846, 545], [996, 678]], np.int32)
yellow_polygon = np.array([[100, 870], [390, 740], [1100, 740], [1350, 870]], np.int32)

# 메인 루프 전에 초기 시간 변수 설정
prev_time = time.time() 
####################### #000


while True:
    # 보드에서 수신
    length_buf = recvall(conn, 4)
    if not length_buf:
        break
    frame_len = struct.unpack('>I', length_buf)[0]
    frame_data = recvall(conn, frame_len)

    np_arr = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 여기서 YOLO / 차선 인식 처리! #000

    # [수정] 프레임 크기 축소 #100
    #frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    frame = cv2.resize(frame, (resize_width, resize_height))
    # [수정] YOLO 적용 #100
    results = model(frame, conf=0.3)[0] #005

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label_id = int(box.cls)
        label = model.names[label_id]

        if label not in REAL_HEIGHTS:
            continue

        pixel_height = y2 - y1
        pixel_width = x2 - x1
        if pixel_height <= 0 or pixel_width <= 0:
            continue

        # 중심점 (bbox 하단 중앙)
        cx, cy = (x1 + x2) // 2, y2

        # 어느 폴리곤에 포함되어 있는가? 
        in_red = cv2.pointPolygonTest(red_polygon, (cx, cy), False) >= 0
        in_yellow = cv2.pointPolygonTest(yellow_polygon, (cx, cy), False) >= 0 #001

        # # 객체가 차선 폴리곤 내부에 있을 때만 처리 #red하나만 있었을때.
        # if cv2.pointPolygonTest(red_polygon, (cx, cy), False) < 0:
        #     continue

        # # 폴리곤에 포함되지 않으면 무시 #red,yellow둘다 있을 때 #001 #002
        if not (in_red or in_yellow):
            continue

        distance = estimate_distance(pixel_height, pixel_width, label)

        # 색상 설정 #001
        if in_red:
            color = (0, 0, 255)  # 빨간색
        elif in_yellow:
            color = (0, 255, 255)  # 노란색

        # 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 차선 폴리곤 시각화
    cv2.polylines(frame, [red_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(frame, [yellow_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    # ▶️ FPS 측정 #003
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # FPS 표시 #003
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Received Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
