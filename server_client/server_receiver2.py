# server_receiver.py
import socket
import struct
import cv2
import numpy as np
import json #200
from ultralytics import YOLO #000
import time #000

#------설정-------
HOST = '0.0.0.0'
VIDEO_PORT = 8888
OBJECT_PORT = 9999

# 🔧 하이퍼파라미터 #000
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

# ---거리 추정--- #000
# 🔍 (하이브리드) 거리 추정 함수
def estimate_distance(h, w, label):
    try:
        dist_h = (REAL_HEIGHTS[label] * FOCAL_LENGTH) / h
        dist_w = (REAL_WIDTHS[label] * FOCAL_LENGTH) / w
        return (dist_h + dist_w) / 2
    except:
        return -1
    
#------수신 유틸-------
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# --- 객체 전송 --- #200
def send_object(conn, frame, label, distance, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    _, img_encoded = cv2.imencode('.jpg', roi)
    img_bytes = img_encoded.tobytes()

    meta = {
        'class': label,
        'distance': round(distance, 2),
        'bbox': [x1, y1, x2, y2],
        'img_size': len(img_bytes)
    }
    meta_json = json.dumps(meta).encode()

    conn.sendall(struct.pack(">I", len(meta_json)))
    conn.sendall(meta_json)
    conn.sendall(img_bytes)

# ---소켓 준비--- #200 (원래 server_sock -> video_sock,object_sock 2개로 만듦.)
video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
video_sock.bind((HOST, VIDEO_PORT))
video_sock.listen(1)
video_conn, addr = video_sock.accept()
print(f"[✅] Video stream connected.//Connected by {addr}")

object_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
object_sock.bind(('0.0.0.0', OBJECT_PORT))
object_sock.listen(1)
object_conn, _ = object_sock.accept()
print("[✅] Object display client connected.")

#######################  #000
# ▶️ YOLO 모델 로드 
#model = YOLO("yolov8n.pt")
#직접 훈련시킨 최종 모델
model = YOLO("/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt")

# GPU 사용을 위한 device 설정 #006
# ultralytics 라이브러리에서 YOLO 모델을 GPU로 실행하려면 device 설정을 model() 호출 시 to() 메서드를 사용하여 GPU로 전환해야 합니다.
model.to('cuda')  # 'cuda'를 지정하여 모델을 GPU로 전송

# 🚧 위험 폴리곤 설정 (해상도에 맞게 조정 가능) #왼쪽 아래,왼쪽 위, 오른쪽 위, 오른쪽 아래 [x, y]
red_polygon = np.array([[237, 623], [363, 493], [846, 495], [996, 628]], np.int32)
yellow_polygon = np.array([[100, 870], [390, 740], [1100, 740], [1350, 870]], np.int32)

# 메인 루프 전에 초기 시간 변수 설정
prev_time = time.time() 
#######################  #000

try:
    while True:
        # 보드에서 수신(프레임 수신)
        length_buf = recvall(video_conn, 4)
        if not length_buf:
            print("[⚠️] Video stream ended.")
            break
        frame_len = struct.unpack('>I', length_buf)[0]
        print(f"[📥] Received frame_len: {frame_len}") #300
        frame_data = recvall(video_conn, frame_len)
        if frame_data is None:
            print("[⚠️] Frame data is None.")
            continue
        
        print(f"[📦] Received frame of size: {len(frame_data)} bytes") #300

        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[⚠️] Frame decode failed.")
            continue
        
        ### 여기서 YOLO / 차선 인식 처리! #000
        # 프레임 크기 변경
        frame = cv2.resize(frame, (resize_width, resize_height))
        #frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        # YOLO 적용
        results = model(frame, conf=0.3)[0] #005
        print(f"[🎯] YOLO detected {len(results.boxes)} objects") #300

        if len(results.boxes) == 0:
            # 감지된 객체가 없을 경우 메시지 출력
            cv2.putText(frame, "No objects detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        else:
            # 감지된 객체가 있을 때
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_id = int(box.cls)
                label = model.names[label_id]
                print(f"  ↳ {label} detected") #300

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
                send_object(object_conn, frame, label, distance, (x1, y1, x2, y2))
            
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

except Exception as e:
    print("[❌ SERVER ERROR]:", e)
finally:
    video_conn.close()
    object_conn.close()
