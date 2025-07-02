# final_server.py
import socket
import struct
import cv2
import numpy as np
import time
import json
from ultralytics import YOLO
import traceback #888

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
#model.to('cuda')

# ROI의 위쪽/아래쪽 y좌표 및 마스크 포함 임계값 설정 #777
danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260
danger_threshold = 0.1
warning_threshold = 0.1
# ROI 및 FPS 관련 전역 변수 초기화 #777
prev_frame_time = 0
prev_edges = None
frame_count = 0
roi_update_interval = 5  # ROI 업데이트 간격 (프레임 단위)
prev_danger_roi = None
prev_warning_roi = None
# 바운딩 박스가 ROI 마스크 내 일정 비율 이상 포함되었는지 확인 #777
def inside_roi(box, mask, threshold):
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    return inside / roi_box.size >= threshold
# 그림자 제거용 HSV 채널 필터링 함수 #777
def remove_shadows_color_based(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([179, 19, 68])
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(shadow_mask))
    return result
# 차선 기반 trapezoid 형태의 ROI 생성 함수 #777
def create_trapezoid_roi(frame, y_bottom, y_top):
    global prev_edges
    height, width = frame.shape[:2]
    # 그레이스케일 변환 및 에지 검출 #777
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 140)
    # 이전 에지 맵과 가중 평균하여 안정화 #777
    if prev_edges is not None:
        edges = cv2.addWeighted(edges.astype(np.float32), 0.7,
                                prev_edges.astype(np.float32), 0.3, 0).astype(np.uint8)
    prev_edges = edges.copy()
    # 관심영역 설정 #777
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    # ROI 내 차선 검출 #777
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=70)
    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5: left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5: right_lines.append((x1, y1, x2, y2))
    # 직선 근사로 평균화 #777
    def average_line(lines):
        if not lines: return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]; y += [y1, y2]
        return np.polyfit(y, x, 1)
    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)
    if left_fit is None or right_fit is None: return None
    lx1, lx2 = int(np.polyval(left_fit, y_bottom)), int(np.polyval(left_fit, y_top))
    rx1, rx2 = int(np.polyval(right_fit, y_bottom)), int(np.polyval(right_fit, y_top))
    return np.array([[(lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)]], dtype=np.int32)

# 소켓 설정
HOST = '0.0.0.0'
PORT = 7777

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
        #frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # ② 디코딩
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # [ min 수정 ] 초기화 부분을 위로 올림.
        danger_objects = []

        height, width = frame.shape[:2] # 888 프레임 크기를 기준으로 정확하게 맞춰주기 위함.

        # ROI 갱신 (간격마다) #777
        if frame_count % roi_update_interval == 0:
            danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
            warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
            if danger_roi is not None:
                prev_danger_roi = danger_roi
            if warning_roi is not None:
                prev_warning_roi = warning_roi
        else:
            danger_roi = prev_danger_roi
            warning_roi = prev_warning_roi
        # ROI 마스크 생성 #777
        mask_danger = np.zeros((height, width), dtype=np.uint8)
        mask_warning = np.zeros((height, width), dtype=np.uint8)
        if danger_roi is not None:
            cv2.fillPoly(mask_danger, [danger_roi], 255)
        if warning_roi is not None:
            cv2.fillPoly(mask_warning, [warning_roi], 255)
            mask_warning = cv2.subtract(mask_warning, mask_danger)  # 위험 ROI 제외
        # 시각화용 프레임 복사 #777
        roi_overlay = frame.copy()
        frame_output = frame.copy()
        # ROI 시각화 (빨강/노랑 폴리라인) #777
        if danger_roi is not None:
            cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)
        if warning_roi is not None:
            warning_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(warning_mask, [warning_roi], 255)
            danger_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(danger_mask, [danger_roi], 255)
            warning_mask = cv2.subtract(warning_mask, danger_mask)
            contours, _ = cv2.findContours(warning_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)
        
        # 객체 감지 수행 
        results = model(frame, conf=0.3)[0]
        boxes = results.boxes #888
        
        # [ min 수정 ] 기존 코드에서 수정함. -------        
        if boxes is not None and boxes.xyxy is not None:
            # for 루프가 모든 객체를 순회하도록 아래 로직 전체를 포함시켜야 합니다.
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                label_id = int(boxes.cls[i].item())
                label = model.names[label_id]

                # [수정] 아래 로직 전체를 for 루프 안으로 들여쓰기
                if label not in REAL_HEIGHTS:
                    continue
                
                in_danger = inside_roi((x1, y1, x2, y2), mask_danger, danger_threshold)
                in_warning = inside_roi((x1, y1, x2, y2), mask_warning, warning_threshold)
                
                # ROI 바깥 객체는 무시 (수정된 로직에서는 continue가 루프에 영향을 줌)
                if not (in_danger or in_warning):
                    continue

                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                cx, cy = (x1 + x2) // 2, y2
                distance = estimate_distance(h, w, label)

                # 전송용 데이터 구성
                danger_objects.append({
                    "label": str(label),
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(w),
                    "h": int(h),
                    "distance": float(round(distance, 2)),
                    "zone": "red" if in_danger else "yellow"
                })

                # 디버깅용 시각화
                color = (0, 0, 255) if in_danger else (0, 255, 255)
                cv2.rectangle(frame_output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_output, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # [ min 수정 ] 기존 코드 주석 처리함.
        # if boxes is None or boxes.xyxy is None: #888
        #     danger_objects = []
        #     pass
        # else:
        #     for i in range(len(boxes)):
        #         x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
        #         label_id = int(boxes.cls[i].item())
        #         label = model.names[label_id]

        #     if label not in REAL_HEIGHTS:
        #         continue
            
        #     in_danger = inside_roi((x1, y1, x2, y2), mask_danger, danger_threshold) #777
        #     in_warning = inside_roi((x1, y1, x2, y2), mask_warning, warning_threshold)
        #     if in_danger or in_warning:
        #         color = (0, 0, 255) if in_danger else (0, 255, 255)
        #         cv2.rectangle(frame_output, (x1, y1), (x2, y2), color, 2)
        #         cv2.putText(frame_output, label, (x1, y1 - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #     else:
        #         continue  # ROI 바깥 객체는 그리지 않음 #777


        #     w, h = x2 - x1, y2 - y1
        #     if w <= 0 or h <= 0:
        #         continue

        #     cx, cy = (x1 + x2) // 2, y2

        #     distance = estimate_distance(h, w, label)

        #     # 전송용 데이터 구성
        #     danger_objects.append({
        #         "label": str(label),
        #         "x": int(x1),
        #         "y": int(y1),
        #         "w": int(w),
        #         "h": int(h),
        #         "distance": float(round(distance, 2)),
        #         "zone": "red" if in_danger else "yellow"
        #     })

        #     # 디버깅용 시각화
        #     color = (0, 0, 255) if in_danger else (0, 255, 255)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     cv2.circle(frame, (cx, cy), 5, color, -1)
        #     cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # FPS 측정
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 클라이언트에게 전송 (JSON 직렬화)
        try:
            response_json = json.dumps(danger_objects)
        except Exception as e:
            print(f"[JSON 직렬화 에러] {e}")
            print("문제가 된 데이터:", danger_objects)
            continue
        #conn.sendall(response_json.encode('utf-8'))
        payload = response_json.encode('utf-8')
        conn.sendall(struct.pack('>I', len(payload)))  # 4바이트 길이
        conn.sendall(payload)

        # 디버깅 화면 출력
        # 오버레이 합성 후 디스플레이 #777
        final_display = cv2.addWeighted(frame_output, 1.0, roi_overlay, 0.3, 0) #777
        cv2.imshow("YOLO + Polygon Danger Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print(f"[에러] {e}")
        traceback.print_exc() #888
        break

conn.close()
server_socket.close()
cv2.destroyAllWindows()
