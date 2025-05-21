import os
import cv2
import numpy as np
import json
from flask import current_app
from src.routes.analyze import process_drawing_image

def test_analyzer_accuracy():
    """
    테스트 이미지에 대한 분석 정확도 테스트 함수
    """
    # 테스트 이미지 디렉토리
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'test_images')
    accuracy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'accuracy_tests')
    
    # 결과 저장 디렉토리 생성
    os.makedirs(accuracy_dir, exist_ok=True)
    
    # 결과 요약
    summary = {
        'total_tests': 0,
        'successful_tests': 0,
        'accuracy_rate': 0,
        'test_details': []
    }
    
    # 테스트 이미지 목록
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        
        # 예상 각도 추출 (파일명에서 패턴 찾기)
        expected_angles = []
        if 'basic_angles' in img_file:
            expected_angles = [45, 90, 30]
        elif 'various_angles' in img_file:
            expected_angles = [30, 45, 60, 90, 120, 135]
        elif 'small_angles' in img_file:
            expected_angles = [10, 5, 15]
        elif 'large_angles' in img_file:
            expected_angles = [170, 160, 150]
        
        # 이미지 분석 실행
        try:
            analysis_result = process_drawing_image(img_path)
            
            # 검출된 각도 목록
            detected_angles = [angle['value'] for angle in analysis_result['angles']]
            
            # 정확도 계산
            correct_detections = 0
            angle_matches = []
            
            for expected in expected_angles:
                best_match = None
                min_diff = float('inf')
                
                for detected in detected_angles:
                    diff = abs(expected - detected)
                    if diff < min_diff and diff <= 5:  # 5도 이내 오차 허용
                        min_diff = diff
                        best_match = detected
                
                if best_match is not None:
                    correct_detections += 1
                    angle_matches.append({
                        'expected': expected,
                        'detected': best_match,
                        'difference': min_diff
                    })
            
            # 정확도 계산
            if len(expected_angles) > 0:
                accuracy = (correct_detections / len(expected_angles)) * 100
            else:
                accuracy = 0
            
            # 테스트 결과 저장
            test_result = {
                'image_file': img_file,
                'expected_angles': expected_angles,
                'detected_angles': detected_angles,
                'angle_matches': angle_matches,
                'correct_detections': correct_detections,
                'total_expected': len(expected_angles),
                'accuracy': accuracy,
                'result_image': analysis_result['analyzed_image']
            }
            
            summary['test_details'].append(test_result)
            summary['total_tests'] += 1
            
            if accuracy >= 80:  # 80% 이상 정확도를 성공으로 간주
                summary['successful_tests'] += 1
            
        except Exception as e:
            print(f"테스트 실패 ({img_file}): {str(e)}")
            summary['test_details'].append({
                'image_file': img_file,
                'error': str(e),
                'accuracy': 0
            })
            summary['total_tests'] += 1
    
    # 전체 정확도 계산
    if summary['total_tests'] > 0:
        summary['accuracy_rate'] = (summary['successful_tests'] / summary['total_tests']) * 100
    
    # 결과 저장
    with open(os.path.join(accuracy_dir, 'accuracy_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

def improve_analyzer_based_on_tests(summary):
    """
    테스트 결과를 바탕으로 분석기 개선 제안 생성
    """
    improvements = {
        'general_suggestions': [],
        'angle_specific_improvements': [],
        'algorithm_adjustments': []
    }
    
    # 전체 정확도 확인
    if summary['accuracy_rate'] < 70:
        improvements['general_suggestions'].append(
            "전체 정확도가 낮습니다. 알고리즘 전반적인 개선이 필요합니다."
        )
    
    # 테스트 세부 결과 분석
    for test in summary['test_details']:
        if 'error' in test:
            improvements['general_suggestions'].append(
                f"이미지 '{test['image_file']}'에서 오류 발생: {test['error']}"
            )
            continue
        
        if test['accuracy'] < 70:
            improvements['angle_specific_improvements'].append(
                f"이미지 '{test['image_file']}'의 각도 검출 정확도가 낮습니다 ({test['accuracy']:.1f}%)."
            )
            
            # 누락된 각도 확인
            detected = set(round(a) for a in test['detected_angles'])
            expected = set(test['expected_angles'])
            missed = expected - detected
            
            if missed:
                improvements['angle_specific_improvements'].append(
                    f"이미지 '{test['image_file']}'에서 다음 각도를 검출하지 못했습니다: {', '.join(map(str, missed))}"
                )
        
        # 각도별 오차 분석
        if 'angle_matches' in test:
            large_errors = [m for m in test['angle_matches'] if m['difference'] > 2]
            if large_errors:
                improvements['algorithm_adjustments'].append(
                    f"이미지 '{test['image_file']}'에서 일부 각도의 오차가 큽니다: " + 
                    ", ".join([f"예상 {e['expected']}° vs 검출 {e['detected']}° (오차 {e['difference']}°)" for e in large_errors])
                )
    
    # 알고리즘 개선 제안
    if any('small_angles' in test['image_file'] and test['accuracy'] < 80 for test in summary['test_details'] if 'accuracy' in test):
        improvements['algorithm_adjustments'].append(
            "작은 각도(15° 이하) 검출 정확도 개선 필요: 더 정밀한 선 검출 알고리즘 적용 고려"
        )
    
    if any('large_angles' in test['image_file'] and test['accuracy'] < 80 for test in summary['test_details'] if 'accuracy' in test):
        improvements['algorithm_adjustments'].append(
            "큰 각도(150° 이상) 검출 정확도 개선 필요: 보완 각도 계산 로직 추가 고려"
        )
    
    # 결과 저장
    accuracy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'accuracy_tests')
    with open(os.path.join(accuracy_dir, 'improvement_suggestions.json'), 'w', encoding='utf-8') as f:
        json.dump(improvements, f, ensure_ascii=False, indent=2)
    
    return improvements

if __name__ == "__main__":
    # 테스트 실행
    summary = test_analyzer_accuracy()
    
    # 개선 제안 생성
    improvements = improve_analyzer_based_on_tests(summary)
    
    # 결과 출력
    print(f"테스트 완료: 전체 정확도 {summary['accuracy_rate']:.1f}%")
    print(f"성공한 테스트: {summary['successful_tests']}/{summary['total_tests']}")
    
    print("\n개선 제안:")
    for category, suggestions in improvements.items():
        if suggestions:
            print(f"\n{category.replace('_', ' ').title()}:")
            for suggestion in suggestions:
                print(f"- {suggestion}")
