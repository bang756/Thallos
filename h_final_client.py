import cv2
import socket
import struct
import numpy as np
import time
import json
import threading
from queue import Queue

# --- 설정 ---
SERVER_IP = '192.168.3.28'
SERVER_PORT = 7777
VIDEO_SOURCE = 'rural_cut.webm' #'test_movie_009.mp4'
resize_width, resize_height = 640, 480

# 스레드 간 공유 큐
frame_queue = Queue(maxsize=10)
response_queue = Queue()

# 소켓 전역 객체
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

def frame_sender():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: 비디오 소스를 열 수 없습니다: {VIDEO_SOURCE}")
        return

    print("INFO: 전송 스레드 시작됨.")
    frame_id = 0  # 프레임 번호 #h
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1 #h

        # 🔻 프레임 스킵 적용: 홀수 프레임은 전송하지 않음 #h
        if frame_id % 2 != 0:
            continue

        frame = cv2.resize(frame, (resize_width, resize_height))

        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            continue
        data = encoded_frame.tobytes()

        try:
            # 길이 + 프레임 전송
            client_socket.sendall(struct.pack('>I', len(data)))
            client_socket.sendall(data)

            # 응답 수신 (길이 먼저)
            len_buf = client_socket.recv(4, socket.MSG_WAITALL)
            if not len_buf:
                continue
            response_len = struct.unpack('>I', len_buf)[0]
            response = client_socket.recv(response_len, socket.MSG_WAITALL).decode('utf-8')

            # 공유 큐에 결과 전송
            frame_queue.put(frame)
            response_queue.put(response)

        except socket.error as e:
            print(f"[ERROR] 전송 스레드 소켓 에러: {e}")
            break

    cap.release()
    print("INFO: 전송 스레드 종료됨.")

def main():
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"INFO: 서버({SERVER_IP}:{SERVER_PORT})에 성공적으로 연결되었습니다.")
    except socket.error as e:
        print(f"ERROR: 서버 연결에 실패했습니다: {e}")
        return

    # 전송 스레드 시작
    sender_thread = threading.Thread(target=frame_sender, daemon=True)
    sender_thread.start()

    # fps = 0.0
    # frame_cnt = 0
    # fps_t0 = time.time()
    # 메인 루프 전에 초기 시간 변수 설정
    prev_time = time.time()  #c

    print("INFO: 렌더링 루프 시작됨. 'q' 키를 누르면 종료됩니다.")

    while True:
        if frame_queue.empty() or response_queue.empty():
            time.sleep(0.005)
            continue

        frame = frame_queue.get()
        response = response_queue.get()

        try:
            objects = json.loads(response)

            for obj in objects:
                label = obj.get("label", "unknown")
                x = obj.get("x", 0)
                y = obj.get("y", 0)
                w = obj.get("w", 0)
                h = obj.get("h", 0)
                dist = obj.get("distance", -1)
                zone = obj.get("zone", "red")

                color = (0, 0, 255) if zone == "red" else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {dist:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON 파싱 에러: {e}")

        # # FPS 계산
        # frame_cnt += 1
        # elapsed = time.time() - fps_t0
        # if elapsed >= 1.0:
        #     fps = frame_cnt / elapsed
        #     fps_t0, frame_cnt = time.time(), 0
        # cv2.putText(frame, f"FPS {fps:.1f}", (30, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # ▶️ FPS 측정 #c
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS {fps:.1f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 디스플레이
        cv2.imshow('Client View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()
    cv2.destroyAllWindows()
    print("INFO: 클라이언트 종료됨.")

if __name__ == '__main__':
    main()
