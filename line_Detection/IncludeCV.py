import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLOv8 모델 로드 및 초기화
model = YOLO('/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt')
model.eval()    # 모델을 테스트 모드로 추론 (학습용 x)
model.fuse()    # 모델 최적화 ->속도 향상 -> CPU일 때 향상률 ↑
_ = model(np.zeros((360, 640, 3), dtype=np.uint8))  # 워밍업

# GPU 사용 설정 #희진
model.to('cuda')  # 'cuda'를 사용하여 GPU로 모델 이동

# 동영상 로드
cap = cv2.VideoCapture("rural_cut.webm")  # 동영상 열기  #test_movie_009.mp4
resize_width, resize_height = 640, 360  # 사이즈 변환 
fps = cap.get(cv2.CAP_PROP_FPS) # 동영상 fps값 추출 (fps = 24)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상 전체 프레임 수
delay = int(1000 // (fps * 8))  # (fps*8)로 동영상 속도 개선

# 클래스 이름 정의
class_names = {
    0: "vehicle", 1: "big vehicle", 4: "bike",
    5: "human", 6: "animals", 7: "obstacles"
}

# 카메라 초점 거리 (픽셀 단위) 기본값
focal_length = 700  # 초점거리 (적절하게 설정)

# 차량의 실제 크기 (미터 단위) - vehicle과 big vehicle 구분
vehicle_real_length = 4.0  # 일반 차량 크기 (최대 4미터)
big_vehicle_real_length = 5.0  # 대형 차량 크기 (최소 5미터 이상)

# 박스가 ROI 안에 일정 비율 이상 들어갔는지 확인하는 함수
def inside_roi(box, mask, threshold):   
    x1, y1, x2, y2 = map(int, box)
    roi_box = mask[y1:y2, x1:x2]
    if roi_box.size == 0:
        return False
    inside = np.count_nonzero(roi_box == 255)
    ratio = inside / roi_box.size
    return ratio >= threshold

# 차선 기반으로 사다리꼴 ROI 생성
def create_trapezoid_roi(frame, y_bottom, y_top):
    height, width = frame.shape[:2]

    # 영상 전처리 : grayscale -> blur -> edge(canny)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI 마스크 생성
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height), (width // 2 - 60, height // 2),
        (width // 2 + 60, height // 2), (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 허프 변환  - 왼쪽 기울기 <0, 오른쪽 기울기 >0로 직선을 분류
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50,
                            minLineLength=40, maxLineGap=100)

    left_lines, right_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))

    def average_line(lines):
        if not lines:
            return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        return np.polyfit(y, x, deg=1)  # x = ay + b

    left_fit = average_line(left_lines)
    right_fit = average_line(right_lines)

    if left_fit is None or right_fit is None:
        return None

    lx1, lx2 = int(np.polyval(left_fit, y_bottom)), int(np.polyval(left_fit, y_top))
    rx1, rx2 = int(np.polyval(right_fit, y_bottom)), int(np.polyval(right_fit, y_top))

    return np.array([[
        (lx1, y_bottom), (lx2, y_top), (rx2, y_top), (rx1, y_bottom)
    ]], dtype=np.int32)

# ROI 높이 설정 - 360은 하단값, 뒤의 값은 상단값으로 작을수록 영역이 길어짐.
danger_bottom, danger_top = 360, 300
warning_bottom, warning_top = 360, 260

# ROI 별 threshold 설정 - 픽셀 비율로 ROI 영역 침범 시 감지
danger_threshold = 0.2    # 0.1은 좌우 다 잡아버림.
warning_threshold = 0.3

# 거리 계산 함수
def calculate_distance(vehicle_screen_width, vehicle_real_length):
    """차량의 화면 너비와 실제 차량 크기를 비교하여 거리 계산"""
    return (focal_length * vehicle_real_length) / vehicle_screen_width

# 트랙바 콜백 함수
def update_video_position(val):
    """트랙바 값에 따라 동영상 재생 위치를 업데이트"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)

# 트랙바 생성
cv2.namedWindow('YOLOv8 ROI Detection')
cv2.createTrackbar('Video Position', 'YOLOv8 ROI Detection', 0, video_length - 1, update_video_position)

# FPS 측정을 위한 변수
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():

    # 프레임 읽고 사이즈 조정
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))

    # 빈 마스크 생성
    mask_danger = np.zeros((resize_height, resize_width), dtype=np.uint8)
    mask_warning = np.zeros((resize_height, resize_width), dtype=np.uint8)

    # danger ROI 생성
    danger_roi = create_trapezoid_roi(frame, danger_bottom, danger_top)
    if danger_roi is not None:
        cv2.fillPoly(mask_danger, [danger_roi], 255)

    # warning ROI 생성
    warning_roi = create_trapezoid_roi(frame, warning_bottom, warning_top)
    if warning_roi is not None:
        cv2.fillPoly(mask_warning, [warning_roi], 255)
        mask_warning = cv2.subtract(mask_warning, mask_danger)

    roi_overlay = frame.copy()

    # Danger ROI 테두리: 무조건 먼저 빨간색으로 그림
    if danger_roi is not None:
        cv2.polylines(roi_overlay, [danger_roi], isClosed=True, color=(0, 0, 255), thickness=3)

    # Warning ROI 테두리: Danger ROI와 겹치는 부분은 제외하고 노란색으로 그림
    if warning_roi is not None:
        if danger_roi is not None:
            # Warning ROI 마스크 만들기
            warning_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(warning_mask, [warning_roi], 255)

            # Danger ROI 마스크 빼기 (겹치는 부분 제거)
            danger_mask = np.zeros((resize_height, resize_width), dtype=np.uint8)
            cv2.fillPoly(danger_mask, [danger_roi], 255)

            # Warning 마스크에서 Danger 부분 제거
            warning_mask = cv2.subtract(warning_mask, danger_mask)

            # 남은 Warning 부분의 외곽선 찾아 테두리 그림
            contours, _ = cv2.findContours(warning_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.polylines(roi_overlay, [cnt], isClosed=True, color=(0, 255, 255), thickness=3)
        else:
            # Danger ROI가 없으면 Warning ROI 전체를 그림
            cv2.polylines(roi_overlay, [warning_roi], isClosed=True, color=(0, 255, 255), thickness=3)

    # 객체 탐지
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, class_id in zip(boxes, classes):
        class_id = int(class_id)
        if class_id not in class_names:
            continue

        x1, y1, x2, y2 = [int(c) for c in box]
        class_name = class_names[class_id]

        # 객체 탐지 시 색 지정 - danger, warning 외
        if inside_roi(box, mask_danger, danger_threshold):
            color = (0, 0, 255)
        elif inside_roi(box, mask_warning, warning_threshold):
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        # 차량의 화면 상 너비 계산
        vehicle_screen_width = x2 - x1

        # 거리 계산 (차량에 대해서만 계산)
        if class_id == 0 or class_id == 1:  # big vehicle과 vehicle에 대해서만 거리 계산
            vehicle_real_length_used = vehicle_real_length if class_id == 0 else big_vehicle_real_length  # 4m 또는 5m
            distance = calculate_distance(vehicle_screen_width, vehicle_real_length_used)

            # 거리 표시
            cv2.putText(frame, f"Dis: {distance:.2f}[m]", (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 바운딩 박스, 클래스별 라벨 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS 계산
    new_frame_time = time.time()  # 현재 시간을 프레임 처리 시간으로 저장
    fps_value = 1 / (new_frame_time - prev_frame_time)  # FPS 계산
    prev_frame_time = new_frame_time  # 이전 프레임 시간 업데이트

    # FPS 출력
    cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 표시
    overlay = cv2.addWeighted(frame, 1.0, roi_overlay, 0.3, 0)
    cv2.imshow("YOLOv8 ROI Detection", overlay)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == 81:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) - 5000)
    elif key == 83:
        cap.set(cv2.CAP_PROP_POS_MSEC, cap.get(cv2.CAP_PROP_POS_MSEC) + 5000)

cap.release()
cv2.destroyAllWindows()




