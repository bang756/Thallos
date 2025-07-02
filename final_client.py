# final_client.py
import cv2
import socket
import struct
import numpy as np
import time
import json  # JSON 파싱을 위한 모듈

# --- 설정 ---
#SERVER_IP = '192.168.3.28' #내 노트북 서버 주소
SERVER_IP = '127.0.0.1' #내 노트북 서버 주소
SERVER_PORT = 7777
#VIDEO_SOURCE = 'rural_cut.webm'
VIDEO_SOURCE = 'rural_cut.webm'

resize_width, resize_height = 640, 480 #300

def main():
    fps = 0.0
    frame_cnt = 0
    fps_t0 = time.time()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"INFO: 서버({SERVER_IP}:{SERVER_PORT})에 성공적으로 연결되었습니다.")
    except socket.error as e:
        print(f"ERROR: 서버 연결에 실패했습니다: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"ERROR: 비디오 소스를 열 수 없습니다: {VIDEO_SOURCE}")
        client_socket.close()
        return

    print("INFO: 클라이언트를 시작합니다. 'q' 키를 누르면 종료됩니다.")

    # # 출력 창 크기 지정 #300
    # cv2.namedWindow('Client View', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Client View', resize_width, resize_height)

    while cap.isOpened():
        ret, frame = cap.read()
        print(ret) # 진짜 True인지 디버깅용 #999
        if not ret:
            print("INFO: 비디오 스트림의 끝에 도달했거나 오류가 발생했습니다.")
            break

        # 🔻 전송 전에 영상 축소 (실제 네트워크 전송 및 서버 YOLO 성능 향상 목적) 
        frame = cv2.resize(frame, (resize_width, resize_height)) #640, 480

        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            print("WARNING: 프레임 인코딩에 실패했습니다.")
            continue

        data = encoded_frame.tobytes()

        try:
            #client_socket.sendall(struct.pack('>L', len(data))) # [ min 수정 ] 주석 처리
            client_socket.sendall(struct.pack('>I', len(data))) # [ min 수정 ]
            client_socket.sendall(data)

            # 서버 응답 수신 (4096 바이트 제한)
            #response = client_socket.recv(4096).decode('utf-8') # [ min 수정 ] 주석 처리

            # [ min 수정 ] 추가 ------------------
            len_buf = client_socket.recv(4, socket.MSG_WAITALL)
            if not len_buf:
                print("WARNING: 서버로부터 응답 길이를 받지 못했습니다.")
                continue
            response_len = struct.unpack('>I', len_buf)[0]
            
            # 2. 받은 길이만큼만 실제 응답 데이터를 받음
            response = client_socket.recv(response_len, socket.MSG_WAITALL).decode('utf-8')
            # [ min 수정 ] 추가 끝 -----------------

            # JSON 파싱
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

                    color = (0, 0, 255) if zone == "red" else (0, 255, 255)  # 빨간/노란 박스
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {dist:.2f}m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except json.JSONDecodeError as e:
                print(f"WARNING: JSON 디코딩 오류 발생: {e}, 수신 데이터: '{response}'")

            # FPS 측정
            frame_cnt += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                fps = frame_cnt / elapsed
                fps_t0, frame_cnt = time.time(), 0
            cv2.putText(frame, f"FPS {fps:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        except socket.error as e:
            print(f"ERROR: 소켓 통신 오류: {e}")
            break

        # 출력
        print("DEBUG: 프레임 받아서 그리기 직전") #999
        cv2.imshow('Client View', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("INFO: 자원을 해제하고 클라이언트를 종료합니다.")
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
