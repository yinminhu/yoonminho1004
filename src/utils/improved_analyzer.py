import os
import cv2
import numpy as np
import re
import math

def improve_angle_detection(image_path):
    """
    개선된 각도 검출 알고리즘
    
    Parameters:
    -----------
    image_path : str
        분석할 이미지 경로
    
    Returns:
    --------
    dict
        검출된 각도 정보
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("이미지를 로드할 수 없습니다.")
    
    # 결과 이미지 (시각화용)
    result_img = img.copy()
    
    # 이미지 전처리
    preprocessed = preprocess_image_improved(img)
    
    # OCR로 텍스트 추출
    text_data = extract_text_improved(preprocessed)
    
    # 도형 요소 검출 (개선된 알고리즘)
    shapes = detect_shapes_improved(preprocessed)
    
    # 각도 정보 추출 (개선된 알고리즘)
    angles = extract_angles_improved(text_data, shapes, preprocessed)
    
    # 각도 정보 시각화
    visualized_img = visualize_angles(result_img, angles)
    
    # 결과 이미지 저장
    result_filename = f"improved_result_{os.path.basename(image_path)}"
    result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'results', result_filename)
    cv2.imwrite(result_path, visualized_img)
    
    # 상대 경로로 변환 (웹에서 접근 가능하도록)
    relative_original_path = os.path.join('uploads', os.path.basename(image_path))
    relative_result_path = os.path.join('results', result_filename)
    
    # 결과 데이터 구성
    result = {
        'original_image': relative_original_path,
        'analyzed_image': relative_result_path,
        'angles': angles,
        'total_angles': len(angles)
    }
    
    return result

def preprocess_image_improved(img):
    """
    개선된 이미지 전처리 함수
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거 (개선된 파라미터)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 적응형 이진화 (개선된 방식)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 엣지 검출 (Canny)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    return {
        'original': img,
        'gray': gray,
        'blur': blur,
        'binary': binary,
        'edges': edges
    }

def extract_text_improved(preprocessed):
    """
    개선된 OCR 텍스트 추출 함수
    """
    # OCR 설정 - 숫자와 기호에 최적화
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+/'
    
    import pytesseract
    
    # 텍스트 추출 (원본 그레이스케일 이미지 사용)
    text_data_gray = pytesseract.image_to_data(preprocessed['gray'], config=custom_config, output_type=pytesseract.Output.DICT)
    
    # 텍스트 추출 (이진화 이미지 사용)
    text_data_binary = pytesseract.image_to_data(preprocessed['binary'], config=custom_config, output_type=pytesseract.Output.DICT)
    
    # 두 결과 병합
    filtered_text = []
    
    # 그레이스케일 결과 처리
    for i in range(len(text_data_gray['text'])):
        if int(text_data_gray['conf'][i]) > 30 and text_data_gray['text'][i].strip() != '':
            filtered_text.append({
                'text': text_data_gray['text'][i],
                'x': text_data_gray['left'][i],
                'y': text_data_gray['top'][i],
                'width': text_data_gray['width'][i],
                'height': text_data_gray['height'][i],
                'conf': text_data_gray['conf'][i],
                'source': 'gray'
            })
    
    # 이진화 결과 처리 (중복 제거)
    for i in range(len(text_data_binary['text'])):
        if int(text_data_binary['conf'][i]) > 30 and text_data_binary['text'][i].strip() != '':
            # 중복 확인
            is_duplicate = False
            for existing in filtered_text:
                # 위치와 텍스트가 유사하면 중복으로 간주
                if (abs(existing['x'] - text_data_binary['left'][i]) < 20 and 
                    abs(existing['y'] - text_data_binary['top'][i]) < 20 and
                    existing['text'] == text_data_binary['text'][i]):
                    is_duplicate = True
                    # 더 높은 신뢰도의 결과 선택
                    if text_data_binary['conf'][i] > existing['conf']:
                        existing['conf'] = text_data_binary['conf'][i]
                        existing['source'] = 'binary'
                    break
            
            if not is_duplicate:
                filtered_text.append({
                    'text': text_data_binary['text'][i],
                    'x': text_data_binary['left'][i],
                    'y': text_data_binary['top'][i],
                    'width': text_data_binary['width'][i],
                    'height': text_data_binary['height'][i],
                    'conf': text_data_binary['conf'][i],
                    'source': 'binary'
                })
    
    return filtered_text

def detect_shapes_improved(preprocessed):
    """
    개선된 도형 요소 검출 함수
    """
    binary = preprocessed['binary']
    edges = preprocessed['edges']
    
    # 직선 검출 (개선된 파라미터)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=15)
    detected_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 너무 짧은 선 필터링
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 10:
                continue
                
            detected_lines.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'length': length,
                'angle': np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            })
    
    # 원 검출 (개선된 파라미터)
    circles = cv2.HoughCircles(
        preprocessed['blur'],
        cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=5, maxRadius=50
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            detected_circles.append({
                'center': (center_x, center_y),
                'radius': radius
            })
    
    # 윤곽선 검출 (개선된 방식)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 작은 노이즈 제거
            detected_contours.append(contour)
    
    # 각도 호 검출 (추가된 기능)
    detected_arcs = []
    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 호 형태 감지 (점이 5-20개 사이인 경우)
        if 5 <= len(approx) <= 20:
            # 면적과 둘레 비율로 호 여부 판단
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0 and area / perimeter < 5:  # 호는 면적/둘레 비율이 작음
                # 호의 중심점 추정
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_arcs.append({
                        'center': (cx, cy),
                        'contour': contour,
                        'approx': approx
                    })
    
    return {
        'lines': detected_lines,
        'circles': detected_circles,
        'contours': detected_contours,
        'arcs': detected_arcs
    }

def extract_angles_improved(text_data, shapes, preprocessed):
    """
    개선된 각도 정보 추출 함수
    """
    angles = []
    
    # 1. 텍스트에서 각도 표기 찾기 (개선된 패턴 인식)
    for text_item in text_data:
        text = text_item['text']
        
        # 직접 표기 패턴 (예: 45°)
        degree_match = re.search(r'(\d+\.?\d*)°', text)
        if degree_match:
            angle_value = float(degree_match.group(1))
            angles.append({
                'value': angle_value,
                'type': 'direct_notation',
                'position': (text_item['x'], text_item['y']),
                'text': text,
                'importance': classify_angle_importance(angle_value),
                'description': get_angle_description(angle_value, 'direct_notation')
            })
            continue
        
        # 텍스트 표기 패턴 (예: 45 deg)
        deg_match = re.search(r'(\d+\.?\d*)\s*(?:deg|degree|degrees)', text.lower())
        if deg_match:
            angle_value = float(deg_match.group(1))
            angles.append({
                'value': angle_value,
                'type': 'text_notation',
                'position': (text_item['x'], text_item['y']),
                'text': text,
                'importance': classify_angle_importance(angle_value),
                'description': get_angle_description(angle_value, 'text_notation')
            })
            continue
            
        # 숫자만 있는 경우 (각도 표시일 가능성)
        num_match = re.search(r'^(\d+\.?\d*)$', text)
        if num_match:
            angle_value = float(num_match.group(1))
            # 각도로 의미 있는 범위인지 확인 (0-180도)
            if 0 <= angle_value <= 180:
                # 주변에 호가 있는지 확인
                has_nearby_arc = False
                for arc in shapes['arcs']:
                    dist = np.sqrt((arc['center'][0] - text_item['x'])**2 + 
                                  (arc['center'][1] - text_item['y'])**2)
                    if dist < 100:  # 100픽셀 이내에 호가 있으면
                        has_nearby_arc = True
                        break
                
                if has_nearby_arc:
                    angles.append({
                        'value': angle_value,
                        'type': 'numeric_with_arc',
                        'position': (text_item['x'], text_item['y']),
                        'text': text,
                        'importance': classify_angle_importance(angle_value),
                        'description': get_angle_description(angle_value, 'numeric_with_arc')
                    })
    
    # 2. 선 사이의 각도 계산 (개선된 알고리즘)
    lines = shapes['lines']
    if len(lines) >= 2:
        # 선 그룹화 (교차점 기준)
        line_groups = group_lines_by_intersection(lines)
        
        for group in line_groups:
            if len(group) >= 2:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        line1 = group[i]
                        line2 = group[j]
                        
                        intersection = find_intersection(line1, line2)
                        if intersection:
                            angle = calculate_angle_between_lines(line1, line2)
                            
                            # 이미 텍스트로 발견된 각도와 중복 확인
                            is_duplicate = False
                            for existing_angle in angles:
                                if 'position' in existing_angle:
                                    dist = np.sqrt((existing_angle['position'][0] - intersection[0])**2 + 
                                                  (existing_angle['position'][1] - intersection[1])**2)
                                    angle_diff = abs(existing_angle['value'] - angle)
                                    if dist < 100 and angle_diff < 10:  # 위치와 각도가 유사하면 중복
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                # 큰 각도 처리 (보완 각도 계산)
                                if angle > 90:
                                    complementary_angle = 180 - angle
                                    angles.append({
                                        'value': complementary_angle,
                                        'type': 'calculated',
                                        'position': intersection,
                                        'lines': [line1, line2],
                                        'importance': classify_angle_importance(complementary_angle),
                                        'description': get_angle_description(complementary_angle, 'calculated')
                                    })
                                else:
                                    angles.append({
                                        'value': angle,
                                        'type': 'calculated',
                                        'position': intersection,
                                        'lines': [line1, line2],
                                        'importance': classify_angle_importance(angle),
                                        'description': get_angle_description(angle, 'calculated')
                                    })
    
    # 3. 호를 이용한 각도 검출 (추가된 기능)
    for arc in shapes['arcs']:
        center = arc['center']
        
        # 호 주변의 선 찾기
        nearby_lines = []
        for line in lines:
            # 선의 끝점과 호 중심 사이의 거리 계산
            dist_start = np.sqrt((line['start'][0] - center[0])**2 + (line['start'][1] - center[1])**2)
            dist_end = np.sqrt((line['end'][0] - center[0])**2 + (line['end'][1] - center[1])**2)
            
            # 선이 호 중심 근처를 지나는 경우
            if dist_start < 30 or dist_end < 30:
                nearby_lines.append(line)
        
        # 호 주변에 두 개 이상의 선이 있으면 각도 계산
        if len(nearby_lines) >= 2:
            for i in range(len(nearby_lines)):
                for j in range(i+1, len(nearby_lines)):
                    line1 = nearby_lines[i]
                    line2 = nearby_lines[j]
                    
                    angle = calculate_angle_between_lines(line1, line2)
                    
                    # 이미 발견된 각도와 중복 확인
                    is_duplicate = False
                    for existing_angle in angles:
                        if 'position' in existing_angle:
                            dist = np.sqrt((existing_angle['position'][0] - center[0])**2 + 
                                          (existing_angle['position'][1] - center[1])**2)
                            angle_diff = abs(existing_angle['value'] - angle)
                            if dist < 50 and angle_diff < 10:  # 위치와 각도가 유사하면 중복
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        angles.append({
                            'value': angle,
                            'type': 'arc_based',
                            'position': center,
                            'lines': [line1, line2],
                            'importance': classify_angle_importance(angle),
                            'description': get_angle_description(angle, 'arc_based')
                        })
    
    return angles

def group_lines_by_intersection(lines, max_distance=20):
    """
    교차점을 기준으로 선을 그룹화하는 함수
    """
    groups = []
    
    for i in range(len(lines)):
        line1 = lines[i]
        
        # 이미 그룹에 속한 선인지 확인
        already_grouped = False
        for group in groups:
            if line1 in group:
                already_grouped = True
                break
        
        if not already_grouped:
            # 새 그룹 시작
            new_group = [line1]
            
            # 다른 선과의 교차점 확인
            for j in range(len(lines)):
                if i != j:
                    line2 = lines[j]
                    
                    # 두 선이 교차하거나 가까운지 확인
                    if are_lines_connected(line1, line2, max_distance):
                        new_group.append(line2)
            
            if len(new_group) > 1:
                groups.append(new_group)
    
    return groups

def are_lines_connected(line1, line2, max_distance=20):
    """
    두 선이 연결되어 있는지 확인
    """
    start1, end1 = line1['start'], line1['end']
    start2, end2 = line2['start'], line2['end']
    
    # 각 끝점 사이의 거리 계산
    distances = [
        np.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2),
        np.sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2),
        np.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2),
        np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
    ]
    
    # 최소 거리가 임계값보다 작으면 연결된 것으로 간주
    return min(distances) < max_distance

def find_intersection(line1, line2):
    """
    두 선의 교차점 찾기
    """
    x1, y1 = line1['start']
    x2, y2 = line1['end']
    x3, y3 = line2['start']
    x4, y4 = line2['end']
    
    # 선 방정식: (x1, y1) - (x2, y2)와 (x3, y3) - (x4, y4)의 교차점
    denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    
    if denominator == 0:  # 평행한 경우
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
    
    # 선분 내에 교차점이 있는지 확인 (여유 있게 판단)
    if -0.1 <= ua <= 1.1 and -0.1 <= ub <= 1.1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    
    return None

def calculate_angle_between_lines(line1, line2):
    """
    두 선 사이의 각도 계산 (개선된 버전)
    """
    angle1 = line1['angle']
    angle2 = line2['angle']
    
    # 각도 정규화 (-180 ~ 180)
    while angle1 < -180: angle1 += 360
    while angle1 > 180: angle1 -= 360
    while angle2 < -180: angle2 += 360
    while angle2 > 180: angle2 -= 360
    
    # 각도 차이 계산
    angle_diff = abs(angle1 - angle2)
    
    # 예각 계산 (0-180도 사이)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    # 예각 계산 (0-90도 사이)
    if angle_diff > 90:
        angle_diff = 180 - angle_diff
    
    return round(angle_diff, 1)

def classify_angle_importance(angle):
    """
    각도의 중요도 분류
    """
    # 특정 중요 각도 (예: 45도, 90도)
    if abs(angle - 90) < 5:
        return 'critical'  # 직각 (매우 중요)
    elif abs(angle - 45) < 5 or abs(angle - 60) < 5:
        return 'high'  # 45도, 60도 (중요)
    elif angle < 30:
        return 'warning'  # 30도 미만 (주의 필요)
    else:
        return 'normal'  # 일반

def get_angle_description(angle, angle_type):
    """
    각도에 대한 설명 생성
    """
    description = ""
    
    # 각도 유형에 따른 기본 설명
    if angle_type == 'direct_notation':
        description = f"{angle}° 각도가 직접 표기되어 있습니다."
    elif angle_type == 'text_notation':
        description = f"{angle}도 각도가 텍스트로 표기되어 있습니다."
    elif angle_type == 'calculated':
        description = f"두 선 사이의 각도는 {angle}°입니다."
    elif angle_type == 'arc_based':
        description = f"호를 기준으로 측정된 각도는 {angle}°입니다."
    elif angle_type == 'numeric_with_arc':
        description = f"호 근처에 표시된 {angle}° 각도입니다."
    
    # 각도 값에 따른 추가 설명
    if abs(angle - 90) < 5:
        description += " 이것은 직각입니다."
    elif abs(angle - 45) < 5:
        description += " 이것은 45도 각도입니다."
    elif abs(angle - 60) < 5:
        description += " 이것은 60도 각도입니다."
    elif abs(angle - 30) < 5:
        description += " 이것은 30도 각도입니다."
    elif angle < 30:
        description += " 이 각도는 30도 미만으로, 주의가 필요합니다."
    elif angle > 150:
        description += " 이 각도는 150도 이상의 큰 각도입니다."
    
    return description

def visualize_angles(image, angles):
    """
    각도 정보 시각화 함수
    """
    visualization = image.copy()
    
    for angle_data in angles:
        # 각도 중요도에 따른 색상 설정
        if angle_data['importance'] == 'critical':
            color = (0, 0, 255)  # 빨간색 (BGR)
        elif angle_data['importance'] == 'high':
            color = (0, 165, 255)  # 주황색
        elif angle_data['importance'] == 'warning':
            color = (0, 255, 255)  # 노란색
        else:
            color = (0, 255, 0)  # 녹색
        
        # 각도 위치에 원 표시
        cv2.circle(visualization, angle_data['position'], 20, color, 2)
        
        # 각도를 형성하는 선 강조 표시 (계산된 각도인 경우)
        if 'lines' in angle_data:
            for line in angle_data['lines']:
                cv2.line(visualization, line['start'], line['end'], color, 2)
        
        # 각도 값 표시
        cv2.putText(
            visualization,
            f"{angle_data['value']}°",
            (angle_data['position'][0] + 25, angle_data['position'][1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
    return visualization

if __name__ == "__main__":
    # 테스트 이미지 디렉토리
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'test_images')
    
    # 테스트 이미지 목록
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        print(f"이미지 분석 중: {img_file}")
        
        try:
            result = improve_angle_detection(img_path)
            print(f"  검출된 각도: {len(result['angles'])}개")
            for angle in result['angles']:
                print(f"    - {angle['value']}° ({angle['type']})")
        except Exception as e:
            print(f"  오류 발생: {str(e)}")
