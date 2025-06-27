import cv2

video_path = "/home/hkit/Downloads/road_video1.mp4"
cap = cv2.VideoCapture(video_path)
SCALE = 0.4 # 프레임 축소 비율 (0.5 = 50%)

# 클릭된 좌표 저장 리스트
click_points = []

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"클릭한 좌표: ({x}, {y})")  # 터미널에 출력

# 창 이름 정의
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기 축소
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

    # 클릭된 점 표시 (빨간 점)
    for pt in click_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    # 안내 문구 표시 (왼쪽 상단)
    cv2.putText(frame, "r: point reset", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):  # r 키 → 점 초기화
        click_points = []
        print("좌표 초기화됨")

    elif key == 27:  # ESC 키 → 종료
        break

cap.release()
cv2.destroyAllWindows()
