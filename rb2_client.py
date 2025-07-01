# rb2_client.py
import cv2
import socket
import struct
import time

SERVER_IP = '192.168.3.28'  # 내 노트북 서버 주소
PORT = 8888

while True:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, PORT))
        print("[✅] Connected to server for video stream")
        break
    except ConnectionRefusedError:
        print("[⏳] Server not ready, retrying...")
        time.sleep(1)

cap = cv2.VideoCapture(0)  # USB 카메라
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # JPEG 압축
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()

        print(f"[📤] Sent frame size: {len(data)} bytes") #300q

        try: # [프레임 크기] 전송 후 [프레임 데이터] 전송
            sock.sendall(struct.pack(">I", len(data)))
            sock.sendall(data)
        except BrokenPipeError:
            print("[❌] Server disconnected. Exiting...")
            break

        time.sleep(0.03) # 약 30fps

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    sock.close()
    cap.release()