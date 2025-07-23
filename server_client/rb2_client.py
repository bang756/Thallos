# rb2_client.py
import cv2 
import socket 
import struct # 바이너리 데이터 전송을 위한 struct 모듈 (길이 정보 등을 packing/ unpacking)
import time 

SERVER_IP = '192.168.3.28'  # 내 노트북 서버 주소
PORT = 8888
#VIDEO_SOURCE = 0 # USB 카메라
VIDEO_SOURCE = "rural_cut.webm"

while True:
    try: #TCP 소켓 생성
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 서버에 연결 시도
        sock.connect((SERVER_IP, PORT))
        print("[✅] Connected to server for video stream")
        break
    except ConnectionRefusedError:
        print("[⏳] Server not ready, retrying...")
        time.sleep(1) # 1초 대기 후 재시도

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 카메라 해상도를 cv2.VideoCapture()에서 강제로 설정하는 방법
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try: 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # JPEG 압축
        ret, buffer = cv2.imencode('.jpg', frame) # 프레임 -> JPEG형식으로 압축
        data = buffer.tobytes() # 압축된 이미지 -> 바이트 배열로 변환
        print(f"[📤] Sent frame size: {len(data)} bytes") #300

        try: # [프레임 크기] 전송 후 [프레임 데이터] 전송
            sock.sendall(struct.pack(">I", len(data))) #데이터의 크기를 4바이트 big-endian 형식으로 packing 하여 전송.
            if not ret:
                print("[❌] Encoding failed.")
                continue

            sock.sendall(data)
        except BrokenPipeError: # 서버가 연결을 끊은 경우 -> 예외 발생 -> 메시지 출력 후 루프 종료
            print("[❌] Server disconnected. Exiting...")
            break

        time.sleep(0.03) # 약 30fps

except KeyboardInterrupt: # 사용자 키보드 인터럽트(Ctrl+C) 처리
    print("Interrupted by user.")
finally: # 종료 시 소켓 및 비디오 리소스 정리
    sock.close()  # 소켓 연결 종료
    cap.release() # 비디오 캡처 객체 해제