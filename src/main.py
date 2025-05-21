from flask import Flask, render_template, send_from_directory
import os
import sys

# 절대 경로 추가 (배포 환경에서 필요)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Flask 앱 생성
app = Flask(__name__)

# 정적 파일 경로 설정
app.static_folder = 'static'

# 라우트 등록
from src.routes.analyze import analyze_bp
app.register_blueprint(analyze_bp)

# 개선된 알고리즘 적용
from src.utils.improved_analyzer import improve_angle_detection
from src.routes.analyze import process_drawing_image
# 기존 함수를 개선된 버전으로 교체
analyze_bp.process_drawing_image = improve_angle_detection

# 업로드 및 결과 폴더 설정
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
RESULT_FOLDER = os.path.join(app.static_folder, 'results')

# 폴더가 없으면 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """
    메인 페이지 라우트
    """
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    업로드된 파일 제공
    """
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    """
    결과 파일 제공
    """
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    # 개발 서버 실행 (0.0.0.0으로 설정하여 외부 접속 허용)
    app.run(host='0.0.0.0', port=5002, debug=True)
