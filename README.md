# Thallos
### 객체 인식 모델 만들기(YOLOv8 활용)
#### class 구분 및 객체 선정
- class로 나눌 객체 선정 기준도 나름 이슈가 있음(훈련시킬때 특징점이 유사해야 학습이 잘 될테니까.)
- 클래스 수를 늘리면 성능이 향상될 것이다? -> 모른다(미지수)
- 어떤 객체를 몇개의 클래스로 분류 할것인지.
#### class별 데이터셋 수집
vehicle, big vehicle, -, -, bike, human, animal, obstacle
#### labelImg로 라벨링
- 유튜브 영상에서 시계열로 객체들의 위치에 따른 데이터들을 수집. (낮, 밤)
- 가드레일, 반대편에서 오는 차들도 추가
- obstacle은 빨간 꼬깔콘, 공사현장 표시들만 포함 (비교적 자주 볼 수 있는)
- 이륜(자전거, 오토바이)은 사람을 제외하고 객체만 잡음. (테스트 해보니 사람이랑 너무 겹치게 판단해서)
#### YOLOv8(n)으로 훈련 -> 우리의 커스텀 모델(best.pt)
- 각 클래스별 데이터 개수 비율을 맞춰야한다.
- 데이터가 많을 수록 성능이 더 좋아질거라는 보장은 없다.
- 검은 옷을 입은 사람들은 human데이터에서 제외(그림자도 human으로 잡음)
- 애매한 거리의 자동차 잘 못잡음(가깝거나 아예 멀면 잘 잡음) -> 애매한 거리의 자동차 데이터셋 부족이라고 판단(보충함)
  
### 충돌 인식(감지)
1. 테스트 영상에 좌표로 고정시킨 폴리곤으로 충돌 감지 및 판단
2. OpenCV를 활용해 차선 인식 -> 거리 제한 -> 추돌 감지 및 판단
- 모든 객체를 표시하지 않은이유: 모든객체에 박스표시를 하게되면, 후처리 속도가 느려지기 때문에, FPS를 높여 실시간 처리 능력을 높이고자 제외함.

### 통신 (SBC <-> 서버)
- 카메라: 1200*800 (해상도: 가로, 세로의 픽셀(Pixel) 개수를 기준으로 표현하는 규격) ex)1920x1080(Full HD), 3840x2160(4K UHD)
- SoC: Qualcomm® QRB4210
- 소켓(TCP) 통신
1. 보드에서 USB 카메라로 영상을 얻는다고 가정.
2. 보드에서 영상 캡처 -> JPEG형식으로 압축 -> 인코딩 -> 서버로 전송 (멀티 스레드)
3. 서버 디코딩 -> YOLO 커스텀 모델로 객체 판단 -> OpenCV로 차선 인식 -> 추돌 감지 -> 좌표, 메타데이터 문자열 형식"[, , ,]"으로 보드에 전송
4. 보드 받은 데이터로 영상에 OpenCV처리 후 창 출력 (멀티 스레드)

소감)
- 모델을 만들때 **전처리의 중요성**(클래스는 몇개로 구분할 것인지, 수집한 이미지 데이터와 라벨링 범위에 따라 모델이 학습되는 정도가 달라지는것 등)에 대해 몸소 느끼며 이해할 수 있었고, 
- 차선 인식에는 OpenCV만 활용하였기 때문에 **허프 변환**으로는 곡선에는 대응하기 어려운점, **엣지 기반 방식**은 차선의 상태, 그림자와 밝기변화(광량)에 따라 엣지로 잘못 인식해 노이즈로 작용되는 것, **ROI 설정의 한계**로는 오르막길, 내리막길에서는 전방 하단으로 설정한 ROI화면 밖에 차선이 위치하게 되어 대응이 불가능해 지거나, 복잡한 형태의 차선은 반영이 어려운 점 등의 명확한 한계점이 있는 것도 알게 되어 추후 머신러닝이나 딥러닝을 활용하여 보완해 보고 싶은 생각이 들었습니다.
- YOLO 기반 차선 인식, CNN 기반 세그멘테이션 모델 사용 (ex. SCNN, ENet, LaneNet 등)
- 통신에 대한 지식은 전무해서 배운 **소켓 통신**을 활용하여 만들었는데 영상을 송수신할때 활용되는 방법이 다양하다는 것과 서버에서 추론한 결과를 다시 보드로 JOSN형태를 직렬화 해서 보낼때 들어가 있는 값들이 모두 기본 Python 타입으로 보장되어 있지 않으면( ex : numpy 타입의 값) 직렬화에 실패하게 되고, 직렬화에 실패하였기 때문에 보드로 응답이 전송되지 않아, 보드(클라이언트)쪽에서는 아무것도 recv받지 못해 계속 대기 상태가 되어 양쪽에서 창이 출력되지 않는것을 보고 코드 내부 동작원리를 이해하고 코딩하는 것의 중요성을 다시금 깨닫는 계기가 되었습니다.
- 또한 클라이언트 쪽에서 카메라 영상을 프레임 단위로 JPEG형식으로 압축 후 인코딩해서 서버로 보내는데 가장 많은 시간이 할애 되는것을 확인하고, 압축률 조정, 멀티 스레드로 구현하였지만 그래도 실시간성이 부족하여 아쉬움이 남았습니다. (프레임 스킵까지도 고려해 봤으나 충돌 감지는 안전과 직결이 되는 기능이기 때문에 적용하지 않았습니다.) - 실시간성은 ROS로 보완해 볼 예정입니다.
