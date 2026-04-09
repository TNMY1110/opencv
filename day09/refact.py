import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: {path} 이미지를 찾을 수 없습니다.")
        return None
    
    return img

# 이미지 그레이 스케일
def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Top-Hat과 Black-Hat 연산을 사용하여 이미지의 대비를 강조
def enhance_contrast(gray_img):
    # 구조 요소 생성 (3x3 사각형)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Top-Hat: 밝은 부분 강조, Black-Hat: 어두운 부분 강조
    imgTopHat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, structuringElement)

    # 그레이스케일 + Top-Hat - Black-Hat
    img_combined = cv2.add(gray_img, imgTopHat)
    img_enhanced = cv2.subtract(img_combined, imgBlackHat)
    
    return img_enhanced

# 블러 처리 후 적응형 이진화를 적용
def get_binary_image(gray_img):
    # 노이즈 제거를 위한 가우시안 블러
    img_blurred = cv2.GaussianBlur(gray_img, ksize=(5, 5), sigmaX=0)

    # 적응형 이진화 (Adaptive Threshold)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    return img_thresh

# 이진화 이미지에서 윤곽선을 찾고 위치 정보를 리스트로 반환
def find_contours(img_binary):
    # 1. 윤곽선 찾기
    contours, _ = cv2.findContours(
        img_binary,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # 2. 찾은 윤곽선들을 순회하며 사각형 정보 추출
    contours_dict = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # 각 윤곽선의 정보를 딕셔너리에 담아 리스트에 추가
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
        
    return contours_dict

# 크기와 비율을 기준으로 번호판 글자 후보군을 추출
def select_candidate_contours(contours_dict):
    # 필터링 기준 상수
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    
    possible_contours = []
    cnt = 0
    
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        # 조건 검사: 면적, 최소 크기, 가로세로 비율
        if (area > MIN_AREA and 
            d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and 
            MIN_RATIO < ratio < MAX_RATIO):
            
            # 나중에 매칭을 위해 고유 인덱스 부여
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
            
    return possible_contours

# 후보 윤곽선들 사이의 거리, 각도, 크기 차이를 분석하여 글자 묶음을 찾습니다
def find_chars(contour_list):
    # 매칭 기준 상수
    MAX_DIAG_MULTIPLYER = 5      # 글자 대각선 길이의 5배 이내에 다음 글자가 있어야 함
    MAX_ANGLE_DIFF = 12.0        # 박스 중심점 사이의 각도가 최대 12도 이내
    MAX_AREA_DIFF = 0.5          # 면적 차이가 50% 이내
    MAX_WIDTH_DIFF = 0.8         # 너비 차이가 80% 이내
    MAX_HEIGHT_DIFF = 0.2        # 높이 차이가 20% 이내
    MIN_N_MATCHED = 3            # 위 조건을 만족하는 박스가 최소 3개 이상 모여야 함

    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            # 두 박스 사이의 거리 및 각도 계산
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))

            # 두 박스의 크기 차이 계산
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            # 모든 조건이 기준치 이내라면 '같은 줄의 글자'로 판단
            if (distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and
                angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and
                width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF):
                matched_contours_idx.append(d2['idx'])

        # 시작점(d1) 본인 인덱스 추가
        matched_contours_idx.append(d1['idx'])

        # 최소 개수 미달이면 패스
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        # 최종 후보군에 추가
        matched_result_idx.append(matched_contours_idx)

        # 이미 매칭된 박스들을 제외하고 나머지 박스들에 대해 재귀적으로 다시 찾기
        unmatched_contour_idx = [d['idx'] for d in contour_list if d['idx'] not in matched_contours_idx]
        unmatched_contour = [d for d in contour_list if d['idx'] in unmatched_contour_idx]
        
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break # 첫 번째 그룹을 찾으면 반복 중단 (재귀에서 처리됨)

    return matched_result_idx

# 글자 묶음을 기준으로 번호판 영역을 수평으로 정렬하여 잘라냅니다.
def crop_plate_images(img_ori, img_thresh, matched_result, width, height):
    # 번호판 여백 및 비율 기준 상수
    PLATE_WIDTH_PADDING = 1.4
    PLATE_HEIGHT_PADDING = 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        # 1. 글자들을 x축 기준(왼쪽->오른쪽)으로 정렬
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        # 2. 번호판의 중심점 계산
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        # 3. 번호판의 너비와 높이 계산 (여백 포함)
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        sum_height = sum(d['h'] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        # 4. 기울어진 각도 계산 (삼각함수 arcsin 사용)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) if triangle_hypotenus != 0 else 0
        
        # 5. 이미지 회전 변환 (Affine Transformation)
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        
        # 6. 회전된 이미지에서 번호판 영역만 크롭
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        # 7. 번호판 비율 검증 (너무 뚱뚱하거나 홀쭉하면 제외)
        actual_ratio = img_cropped.shape[1] / img_cropped.shape[0]
        if actual_ratio < MIN_PLATE_RATIO or actual_ratio > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    return plate_imgs, plate_infos

# 번호판 이미지에서 문자를 추출하고 신뢰도를 계산
def recognize_chars(plate_imgs):
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    longest_idx, longest_text_len = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        # 1. 확대 및 오츠 이진화
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)  # 2.0 → 1.6
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 2. 번호판 내 글자 재탐지 후 정밀 크롭
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h
            if (area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT
                    and MIN_RATIO < ratio < MAX_RATIO):
                plate_min_x = min(plate_min_x, x)
                plate_min_y = min(plate_min_y, y)
                plate_max_x = max(plate_max_x, x + w)
                plate_max_y = max(plate_max_y, y + h)

        # 유효한 글자 영역이 없으면 스킵
        if plate_max_x <= plate_min_x or plate_max_y <= plate_min_y:
            plate_chars.append(("", 0))
            continue

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        # 3. 블러 → 재이진화
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 4. 여백 추가 및 반전
        img_result = cv2.copyMakeBorder(img_result, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img_result_inverted = cv2.bitwise_not(img_result)

        # 5. OCR
        ocr_data = pytesseract.image_to_data(
            img_result_inverted, lang='kor',
            config='--psm 7 --oem 1',
            output_type=pytesseract.Output.DICT
        )

        full_text = ""
        conf_values = []
        has_digit = False

        for j in range(len(ocr_data['text'])):
            text = ocr_data['text'][j].strip()
            conf = int(ocr_data['conf'][j]) if str(ocr_data['conf'][j]).strip() != '' else -1
            if text != "" and conf != -1:
                clean_text = ""
                for c in text:
                    if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                        if c.isdigit():
                            has_digit = True
                        clean_text += c
                full_text += clean_text
                conf_values.append(conf)

        avg_conf = np.mean(conf_values) if conf_values else 0
        plate_chars.append((full_text, avg_conf))

        if has_digit and len(full_text) > longest_text_len:
            longest_text_len = len(full_text)
            longest_idx = i

    return plate_chars, longest_idx

# ================
# ===== 메인 ======
# ================
path = input("이미지 경로 입력: ")
img_ori = load_image(path)

if img_ori is None:
    print('이미지가 존재하지 않습니다.')
    exit()

height, width, channel = img_ori.shape

plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

gray = get_grayscale(img_ori)
img_enhanced = enhance_contrast(gray)
img_binary = get_binary_image(img_enhanced)
contours_dict = find_contours(img_binary)
possible_contours = select_candidate_contours(contours_dict)
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append([d for d in possible_contours if d['idx'] in idx_list])

plate_imgs, plate_infos = crop_plate_images(img_ori, img_binary, matched_result, width, height)

# 결과 확인 (검출된 번호판 후보들 출력)
if plate_imgs:
    plt.figure(figsize=(12, 5))
    for i, img in enumerate(plate_imgs):
        plt.subplot(1, len(plate_imgs), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Candidate {i}')
    plt.show()
else:
    print("번호판 후보를 찾지 못했습니다.")

plate_chars_list, longest_idx = recognize_chars(plate_imgs)

if longest_idx != -1:
    final_chars, final_conf = plate_chars_list[longest_idx]
    print(f"\n[최종 인식 결과] {final_chars}")
    print(f"[평균 신뢰도] {final_conf:.2f}%")

    # 번호판 위치를 원본 이미지에 표시 후 출력
    info = plate_infos[longest_idx]
    img_out = img_ori.copy()
    
    # 번호판 영역 사각형 표시
    cv2.rectangle(
        img_out,
        pt1=(info['x'], info['y']),
        pt2=(info['x'] + info['w'], info['y'] + info['h']),
        color=(255, 0, 0),
        thickness=2
    )
    
    # 번호판 문자 텍스트 표시
    cv2.putText(
        img_out,
        final_chars,
        org=(info['x'], info['y'] - 10),  # 사각형 위에 표시
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(255, 0, 0),
        thickness=2
    )
    
    # matplotlib으로 화면 출력
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
else:
    print("인식된 번호판 문자가 없습니다.")