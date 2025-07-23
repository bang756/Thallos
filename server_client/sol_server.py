# server_receiver.py
import socket
import struct
import cv2
import numpy as np

HOST = '0.0.0.0' # 모든 네트워크 인터페이스에서 접속을 허용함 (서버가 외부에서 접속 가능)
PORT = 8888 # 서버가 수신 대기할 포트 번호 설정

# TCP 소켓을 생성 (IPv4, TCP 스트림 소켓)
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((HOST, PORT)) # 지정한 IP와 포트에 바인딩 (서버 소켓을 HOST:PORT에 연결)
server_sock.listen(1) # 클라이언트 연결 요청을 받을 준비 시작 (최대 1개의 대기 연결 허용)
conn, addr = server_sock.accept() # 클라이언트 연결 수락 (연결이 오면 conn: 통신용 소켓, addr: 클라이언트 주소)
print(f"Connected by {addr}")

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# 무한 루프를 통해 계속해서 영상 프레임 수신 및 처리
while True:
    # 보드에서 수신

    # [1] 먼저, 클라이언트가 보내는 4바이트 프레임 길이 정보를 수신
    # (recvall은 정확히 4바이트를 받을 때까지 반복 수신)
    length_buf = recvall(conn, 4)
    # [2] 수신이 중단되었거나 오류가 있으면 루프 종료 (클라이언트 종료 등)
    if not length_buf:
        break
    # [3] 받은 4바이트 데이터를 unsigned int (빅 엔디안)으로 변환 → 실제 프레임 크기 추출
    frame_len = struct.unpack('>I', length_buf)[0]
    # [4] 해당 길이만큼 실제 프레임(JPEG 인코딩된 이미지 바이트) 수신
    frame_data = recvall(conn, frame_len)
    # [5] 수신된 바이트 데이터를 NumPy 배열로 변환 (OpenCV가 처리 가능하도록)
    np_arr = np.frombuffer(frame_data, dtype=np.uint8)
    # [6] NumPy 배열을 OpenCV 이미지(BGR 형식)로 디코딩 (JPEG → 이미지 복원)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 이후에는 YOLOv8 추론, 시각화, 결과 전송 등의 작업이 뒤따를 수 있음
    
    cv2.imshow("Received Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

conn.close()
server_sock.close()
cv2.destroyAllWindows()
