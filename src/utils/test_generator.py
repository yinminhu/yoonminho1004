import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 테스트 이미지 저장 디렉토리
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'test_images')
TEST_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'test_results')

# 디렉토리가 없으면 생성
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def generate_test_drawing(filename, angles=None):
    """
    테스트용 도면 이미지 생성 함수
    
    Parameters:
    -----------
    filename : str
        저장할 파일 이름
    angles : list of tuples
        각도 정보 리스트 [(각도, x, y), ...]
    """
    # 기본 각도 설정
    if angles is None:
        angles = [(45, 200, 200), (90, 500, 200), (30, 350, 400)]
    
    # 빈 이미지 생성 (흰색 배경)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 각도별 선 그리기
    for angle_deg, x, y in angles:
        # 라디안으로 변환
        angle_rad = np.deg2rad(angle_deg)
        
        # 첫 번째 선 (수평선)
        x1, y1 = x - 80, y
        x2, y2 = x, y
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # 두 번째 선 (각도 적용)
        length = 80
        x3 = int(x + length * np.cos(angle_rad))
        y3 = int(y - length * np.sin(angle_rad))  # y축이 아래로 증가하므로 부호 반전
        cv2.line(img, (x2, y2), (x3, y3), (0, 0, 0), 2)
        
        # 각도 표시 (텍스트)
        text_x = x + 10
        text_y = y - 20
        cv2.putText(img, f"{angle_deg}°", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 각도 호 그리기
        radius = 20
        start_angle = 0  # 수평선 기준
        end_angle = -angle_rad  # y축이 아래로 증가하므로 부호 반전
        cv2.ellipse(img, (x, y), (radius, radius), 0, np.rad2deg(start_angle), np.rad2deg(end_angle), (0, 0, 0), 1)
    
    # 이미지 저장
    file_path = os.path.join(TEST_IMAGES_DIR, filename)
    cv2.imwrite(file_path, img)
    
    return file_path

def generate_test_drawings():
    """
    다양한 테스트 도면 이미지 생성
    """
    # 기본 각도 테스트 이미지
    generate_test_drawing('basic_angles.png')
    
    # 다양한 각도 테스트 이미지
    angles = [(30, 150, 150), (45, 400, 150), (60, 650, 150), 
              (90, 150, 350), (120, 400, 350), (135, 650, 350)]
    generate_test_drawing('various_angles.png', angles)
    
    # 작은 각도 테스트 이미지
    small_angles = [(10, 200, 200), (5, 500, 200), (15, 350, 400)]
    generate_test_drawing('small_angles.png', small_angles)
    
    # 큰 각도 테스트 이미지
    large_angles = [(170, 200, 200), (160, 500, 200), (150, 350, 400)]
    generate_test_drawing('large_angles.png', large_angles)
    
    print("테스트 도면 이미지 생성 완료")
    return [
        os.path.join(TEST_IMAGES_DIR, 'basic_angles.png'),
        os.path.join(TEST_IMAGES_DIR, 'various_angles.png'),
        os.path.join(TEST_IMAGES_DIR, 'small_angles.png'),
        os.path.join(TEST_IMAGES_DIR, 'large_angles.png')
    ]

if __name__ == "__main__":
    # 테스트 도면 이미지 생성
    test_images = generate_test_drawings()
    
    # 생성된 이미지 확인
    for img_path in test_images:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
        
        # 결과 저장
        result_path = os.path.join(TEST_RESULTS_DIR, f"preview_{os.path.basename(img_path)}")
        plt.savefig(result_path)
        plt.close()
    
    print("테스트 이미지 미리보기 생성 완료")
