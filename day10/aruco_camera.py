import cv2
from cv2 import aruco

# 마커 검출 준비
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# 웹캠 열기
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ArUco 마커 검출
    corner, ids, rejected = detector.detectMarkers(frame)
    
    # 검출된 마커 그리기
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corner, ids)

        #각 마커의 정보 출력 
        for (id_val, corner) in zip(ids, corner):
            # corner: (1, 4, 2)형테 -> 4개 모서리 좌표
            corner = corner.reshape(4,2).astype(int)

            # 중심성 계산
            center_x = int(sum(corner[:, 0]) / 4)
            center_y = int(sum(corner[:, 1]) / 4)

            # ID와 위치 표시
            cv2.putText(frame, f'ID: {id_val[0]}', (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"ID={id_val[0]}, Center=({center_x}, {center_y})")

    cv2.imshow('ArUco Detction', frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()