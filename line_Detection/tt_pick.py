# ROI 재계산 최소화 : 현재 5프레임마다 갱신중

import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 모델 로드 및 초기화
model = YOLO("/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt")
model.eval()
model.fuse()
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 모델 warm-up (빈 프레임으로 1회 호출)

# 동영상 파일 로드 및 파라미터 설정
cap = cv2.VideoCapture('test_movie_009.mp4') # 'test_movie_009.mp4'(그나마 안전) rural_cut.webm
resize_width, resize_height = 640, 360
fps = cap.get(cv2.CAP_PROP_FPS)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
delay = int(1000 // (fps * 15))  # 15배속으로 waitKey 딜레이 설정

# 사용자 정의 클래스 이름 매핑
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# 카메라 및 ROI 관련 파라미터
focal_length = 630
CAMERA_TO_BUMPER_OFFSET = 1.0  # 실제 거리 계산 시 차량 전면까지 거리 보정

danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260
danger_threshold = 0.2
warning_threshold = 0.3

# ROI 및 FPS 관련 전역 변수 초기화
prev_frame_time = 0
prev_edges = None
frame_count = 0
roi_update_interval = 5  # ROI 업데이트 간격 (프레임 단위)
prev_danger_roi = None
prev_warning_roi = None

# 객체 높이를 통한 거리 계산 함수
def calculate_distance_from_height(vehicle_screen_height, class_id):
    if vehicle_screen_height == 0:
        return 0
    if class_id == 0:
        real_height = 2.0
    elif class_id == 1:
        real_height = (2.2 if vehicle_screen_height < 50 else 3.0 
                           if vehicle_screen_height < 90 else 4.0)
    else:
        return 0
    camera_based_distance = (focal_length * real_height) / vehicle_screen_height
    return max(camera_based_distance - CAMERA_TO_BUMPER_OFFSET, 0)

# 바운딩 박스가 ROI 마스크 내 일정 비율 이상 포함되었는지 확인하는 함수
def inside_roi(box, mask, threshold):
    x1, y1, x2, y2 = map(int, box)
    
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    
    inside = np.count_nonzero(roi_box == 255)
    return inside / roi_box.size >= threshold

# 그림자 제거 함수 (HSV 채널 기반)
def remove_shadows_color_based(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([179, 19, 68])   # [색상,채도,명도] -> 채도와 명도 조절
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)
    
    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(shadow_mask))
    return result

# 차선 기반 trapezoid ROI 생성 함수
def create_trapezoid_roi(frame, y_bottom, y_top):
    
    global prev_edges
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 140)  # Canny 에지 검출

    # 이전 프레임과 혼합하여 ROI 안정성 향상, 보간 및 누적
    if prev_edges is not None:
        edges = cv2.addWeighted(edges.astype(np.float32), 0.7, 
                                prev_edges.astype(np.float32), 0.3, 0).astype(np.uint8)
    prev_edges = edges.copy()

    # ROI 내부 영역 설정
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)

    # ROI 내에서 차선 탐지
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=70)
    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # 기울기 계산
            if slope < -0.5: left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5: right_lines.append((x1, y1, x2, y2))

    # 좌/우 차선 평균화 후 ROI 영역 설정
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

# 트랙바 설정 (영상 탐색용)
cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, 
                   video_length - 1, lambda val: cap.set(cv2.CAP_PROP_POS_FRAMES, val))

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (resize_width, resize_height))

    # ROI 생성 또는 유지
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

    # ROI 마스크 생성
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)
    if danger_roi is not None:
        cv2.fillPoly(mask_danger, [danger_roi], 255)
    if warning_roi is not None:
        cv2.fillPoly(mask_warning, [warning_roi], 255)
        mask_warning = cv2.subtract(mask_warning, mask_danger)  # 중첩 제거

    # ROI 영역 시각화용 오버레이 복사본
    roi_overlay = frame.copy()
    if danger_roi is not None:
        cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)
    if warning_roi is not None:
        warning_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
        cv2.fillPoly(warning_mask, [warning_roi], 255)
        danger_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
        cv2.fillPoly(danger_mask, [danger_roi], 255)
        warning_mask = cv2.subtract(warning_mask, danger_mask)
        contours, _ = cv2.findContours(warning_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)

    # YOLO 객체 감지
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_names:
            continue
        x1, y1, x2, y2 = [int(c) for c in box]
        class_name = class_names[class_id]

        # ROI 기준으로 색상 지정
        color = (
            (0, 0, 255) if inside_roi(box, mask_danger, danger_threshold)
            else (0, 255, 255) if inside_roi(box, mask_warning, warning_threshold)
            else (0, 255, 0)
        )

        # 거리 계산 및 시각화
        px_height = y2 - y1
        if class_id in [0, 1]:
            distance = calculate_distance_from_height(px_height, class_id)
            cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS 표시
    new_frame_time = time.time()
    fps_value = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 오버레이 합성 및 디스플레이
    overlay = cv2.addWeighted(frame, 1.0, roi_overlay, 0.3, 0)
    cv2.imshow("YOLOv8 ROI Detection", overlay)

    # 키 입력 처리
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == 81:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == 83:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()