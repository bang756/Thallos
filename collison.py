# import cv2
# import numpy as np
# from ultralytics import YOLO

# # 카메라 초점 거리 (픽셀 단위) — 추정값, 필요 시 조정
# FOCAL_LENGTH = 700

# REAL_HEIGHTS = {
#     "person": 1.6,
#     "car": 1.5,
#     "bus": 3.2,
#     "truck": 3.4,
#     "motorbike": 1.4,
#     "bicycle": 1.2
# }

# REAL_WIDTHS = {
#     "person": 0.5,
#     "car": 1.8,
#     "bus": 2.5,
#     "truck": 2.5,
#     "motorbike": 0.8,
#     "bicycle": 0.7
# }


# # 실제 객체 높이 정보 (단위: meter)
# REAL_HEIGHTS = {
#     "person": 1.6,
#     "car": 1.5,
#     "bus": 3.2,
#     "truck": 3.4,
#     "motorbike": 1.4,
#     "bicycle": 1.2
# }

# # YOLOv8 모델 불러오기
# model = YOLO("yolov8n.pt")  # 또는 yolov8n.yaml로 학습한 custom 모델 사용 가능

# # 테스트 영상 경로
# video_path = "/home/hkit/Downloads/road_video1.mp4"
# cap = cv2.VideoCapture(video_path)

# # 프레임 리사이즈 비율 설정 (예: 0.5 = 50% 크기로 축소)
# SCALE = 0.4

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 크기 축소
#     frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

#     results = model(frame)[0]

#     #1번째(높이 기반 버전)
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         label_id = int(box.cls)
#         label = model.names[label_id]

#         if label not in REAL_HEIGHTS:
#             continue

#         # 픽셀 상 높이
#         pixel_height = y2 - y1
#         if pixel_height <= 0:
#             continue

#         real_height = REAL_HEIGHTS[label]
#         # 거리 추정 (높이 기반)
#         distance = (real_height * FOCAL_LENGTH) / pixel_height  # 단위: meter

#         # 바운딩 박스 + 텍스트 출력
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         text = f"{label} {distance:.2f} m"
#         cv2.putText(frame, text, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         # 시각화
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#         cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


#     cv2.imshow("YOLOv8 Distance Estimation", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
import time #003

# 🔧 하이퍼파라미터
FOCAL_LENGTH = 600 # focal length in pixels (조정 가능)
VIDEO_PATH = "rural_cut.webm" #test_movie_009.mp4
SCALE = 0.3 # 프레임 축소 비율 (0.5 = 50%)

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
# 🎞️ 비디오 열기
cap = cv2.VideoCapture(VIDEO_PATH)

# 🚧 위험 폴리곤 설정 (해상도에 맞게 조정 가능) #왼쪽 아래,왼쪽 위, 오른쪽 위, 오른쪽 아래 [x, y]
red_polygon = np.array([[237, 643], [363, 513], [846, 515], [996, 638]], np.int32)
yellow_polygon = np.array([[100, 870], [390, 740], [1100, 740], [1350, 870]], np.int32)

# 메인 루프 전에 초기 시간 변수 설정
prev_time = time.time()  #003

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 축소
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

    results = model(frame, conf=0.5)[0] #005

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

    cv2.imshow("YOLOv8 Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# FPS 측정 방식이 두 가지 (GPT "충돌 방지 코드") #003
# 1. start_time
# - 매 프레임의 실제 처리 시간 기준으로 FPS를 구함
# - 더 직관적이고 단순함
# - 주로 모델 추론 성능, 알고리즘 속도 측정에 사용됨 (가장 많이 사용됨 ✔️)
# 
# 2. curr_time 방식
# - 현재 프레임과 이전 프레임 사이의 시간차로 FPS 계산
# - 프레임 간 간격이 기준이라 GUI 렌더링/디코딩 시간 등에도 영향받음
# - 사용 시 반드시 prev_time = curr_time 마지막에 갱신 필요
# - 재생 속도 반영에 유리 (영상 스트리밍, 실시간 화면 표시)

# 🔧 참고 팁
# start_time 기준이 느린 경우 → 알고리즘 병목
# prev_time 기준이 느린 경우 → 영상 디코딩/렌더링 병목