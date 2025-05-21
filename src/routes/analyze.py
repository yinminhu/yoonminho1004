from flask import Blueprint, request, jsonify, current_app
import os
import cv2
import numpy as np
import pytesseract
from werkzeug.utils import secure_filename
import uuid
import re
import math

analyze_bp = Blueprint('analyze', __name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tif', 'tiff'}

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_numpy_types(data, path="result"):
    if isinstance(data, dict):
        for k, v in data.items():
            log_numpy_types(v, path=f"{path}.{k}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            log_numpy_types(item, path=f"{path}[{i}]")
    elif isinstance(data, tuple):
        for i, item in enumerate(data):
            log_numpy_types(item, path=f"{path}({i})")
    elif isinstance(data, (np.generic, np.ndarray)): # Check for any numpy generic type or array
        current_app.logger.warning(f"NumPy type found at {path}: {type(data)} - Value: {data}")

@analyze_bp.route('/api/analyze', methods=['POST'])
def analyze_drawing():
    """
    도면 이미지를 분석하는 API 엔드포인트
    """
    # 파일 확인
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400
    
    try:
        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # 이미지 분석 수행
        analysis_result = process_drawing_image(file_path)
        
        current_app.logger.info("Performing pre-serialization check for NumPy types...")
        log_numpy_types(analysis_result)
        current_app.logger.info("Pre-serialization check complete.")
        
        # 결과 반환
        return jsonify(analysis_result), 200
    
    except Exception as e:
        current_app.logger.error(f'이미지 분석 중 오류가 발생했습니다: {str(e)}', exc_info=True)
        return jsonify({'error': f'이미지 분석 중 오류가 발생했습니다: {str(e)}'}), 500

def process_drawing_image(image_path):
    """
    도면 이미지를 처리하고 각도 정보를 추출하는 함수
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("이미지를 로드할 수 없습니다.")
    
    # 결과 이미지 (시각화용)
    result_img = img.copy()
    
    # 이미지 전처리
    preprocessed = preprocess_image(img)
    
    # OCR로 텍스트 추출
    text_data = extract_text(preprocessed)
    
    # 도형 요소 검출
    shapes = detect_shapes(preprocessed)
    
    # 각도 정보 추출
    angles = extract_angles(text_data, shapes)
    
    # 각도 정보 시각화
    visualized_img = visualize_angles(result_img, angles)
    
    # 결과 이미지 저장
    result_filename = f"result_{os.path.basename(image_path)}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
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

def preprocess_image(img):
    """
    이미지 전처리 함수
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 이진화
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return {
        'original': img,
        'gray': gray,
        'blur': blur,
        'binary': binary
    }

def extract_text(preprocessed):
    """
    OCR을 사용하여 텍스트 추출
    """
    # OCR 설정 - 숫자와 기호에 최적화
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.°ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+/'
    
    # 텍스트 추출
    text_data = pytesseract.image_to_data(preprocessed['gray'], config=custom_config, output_type=pytesseract.Output.DICT)
    
    # 유효한 텍스트만 필터링
    filtered_text = []
    for i in range(len(text_data['text'])):
        if int(text_data['conf'][i]) > 30 and text_data['text'][i].strip() != '':
            filtered_text.append({
                'text': text_data['text'][i],
                'x': text_data['left'][i],
                'y': text_data['top'][i],
                'width': text_data['width'][i],
                'height': text_data['height'][i],
                'conf': text_data['conf'][i]
            })
    
    return filtered_text

def detect_shapes(preprocessed):
    """
    도형 요소 검출 함수
    """
    binary = preprocessed['binary']
    
    # 직선 검출
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    detected_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'length': np.sqrt((x2 - x1)**2 + (y2 - y1)**2),
                'angle': np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            })
    
    # 원 검출
    circles = cv2.HoughCircles(
        preprocessed['blur'],
        cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50
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
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 작은 노이즈 제거
            detected_contours.append(contour)
    
    return {
        'lines': detected_lines,
        'circles': detected_circles,
        'contours': detected_contours
    }

def extract_angles(text_data, shapes):
    """
    각도 정보 추출 함수
    """
    angles = []
    
    # 1. 텍스트에서 각도 표기 찾기
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
    
    # 2. 선 사이의 각도 계산
    lines = shapes['lines']
    if len(lines) >= 2:
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                
                # 두 선이 충분히 가까운지 확인 (교차점 근처)
                if are_lines_connected(line1, line2):
                    intersection = find_intersection(line1, line2)
                    if intersection:
                        angle = calculate_angle_between_lines(line1, line2)
                        
                        # 이미 텍스트로 발견된 각도와 중복 확인
                        is_duplicate = False
                        for existing_angle in angles:
                            if 'position' in existing_angle:
                                dist = np.sqrt((existing_angle['position'][0] - intersection[0])**2 + 
                                              (existing_angle['position'][1] - intersection[1])**2)
                                if dist < 100:  # 100픽셀 이내에 텍스트 각도가 있으면 중복으로 간주
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            angles.append({
                                'value': angle,
                                'type': 'calculated',
                                'position': intersection,
                                'lines': [line1, line2],
                                'importance': classify_angle_importance(angle),
                                'description': get_angle_description(angle, 'calculated')
                            })
    
    return angles

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
    
    # 선분 내에 교차점이 있는지 확인
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (int(x), int(y))
    
    return None

def calculate_angle_between_lines(line1, line2):
    """
    두 선 사이의 각도 계산
    """
    angle1 = line1['angle']
    angle2 = line2['angle']
    
    # 각도 차이 계산
    angle_diff = abs(angle1 - angle2)
    
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
