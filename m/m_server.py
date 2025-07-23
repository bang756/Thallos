# m_server.py
import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드 (작은 모델로 테스트)
model = YOLO('yolov8n.pt')  
#직접 훈련시킨 최종 모델
model = YOLO("/home/heejin/Documents/Thallos/yolov8_custom14/weights/best.pt")

# 서버 설정
HOST = '0.0.0.0'
PORT = 9888

def recvall(sock, length):
    """정확한 길이만큼 수신하는 함수"""
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise ConnectionError("클라이언트와의 연결이 끊어졌습니다.")
        data += more
    return data

# TCP 서버 소켓 생성 및 대기
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"INFO: 서버가 {HOST}:{PORT}에서 대기 중입니다...")
# 클라이언트 연결 수락
conn, addr = server_socket.accept() #conn: 통신용 소켓
print(f"INFO: 클라이언트 연결됨: {addr}") #addr: 클라이언트 주소

while True:
    try:
        # 1. 프레임 크기 수신
        packed_size = recvall(conn, 4)
        frame_size = struct.unpack('>L', packed_size)[0]

        # 2. 프레임 데이터 수신
        frame_data = recvall(conn, frame_size)

        # 3. JPEG 디코딩
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            print("WARNING: 프레임 디코딩 실패")
            continue

        # 4. YOLOv8 객체 탐지
        results = model(frame, verbose=False)[0]  # 하나의 이미지 결과

        boxes = []
        for box in results.boxes.xywh.cpu().numpy():
            x_center, y_center, w, h = box
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            w, h = int(w), int(h)
            boxes.extend([x, y, w, h])

        # 5. 문자열로 인코딩하여 전송
        if boxes:
            result_str = str(boxes)  # 예: "[x, y, w, h, ...]"
        else:
            result_str = "[]"  # 객체 없을 경우

        conn.sendall(result_str.encode('utf-8'))

        # (선택) 수신된 프레임 디버깅 표시
        # cv2.imshow("YOLOv8 Server", frame)
        # if cv2.waitKey(1) == 27: break

    except (ConnectionError, struct.error) as e:
        print(f"ERROR: 수신 중 예외 발생: {e}")
        break

# 연결 종료
print("INFO: 연결 종료, 서버 종료")
conn.close()
server_socket.close()
cv2.destroyAllWindows()
