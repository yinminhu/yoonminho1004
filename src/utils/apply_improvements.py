from src.routes.analyze import analyze_bp
from src.utils.improved_analyzer import improve_angle_detection

# 기존 분석 함수를 개선된 버전으로 교체
analyze_bp.process_drawing_image = improve_angle_detection
