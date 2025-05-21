# 도면 이미지 자동 해독 프로그램 설치 가이드

이 문서는 도면 이미지 자동 해독 프로그램을 서버에 설치하고 실행하는 방법을 안내합니다.

## 시스템 요구사항

- Python 3.8 이상
- Ubuntu 20.04 LTS 이상 (다른 Linux 배포판도 가능하지만 명령어가 다를 수 있음)
- 최소 4GB RAM
- 최소 10GB 디스크 공간

## 1. 필수 패키지 설치

먼저 시스템에 필요한 패키지를 설치합니다:

```bash
# 시스템 패키지 업데이트
sudo apt-get update

# 필수 시스템 패키지 설치
sudo apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv

# Tesseract OCR 엔진 설치
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng

# OpenCV 의존성 설치
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

## 2. 프로젝트 설정

프로젝트 디렉토리를 생성하고 가상 환경을 설정합니다:

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/drawing-analyzer-app
cd ~/drawing-analyzer-app

# 가상 환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 필요한 Python 패키지 설치
pip install flask opencv-python pytesseract numpy pillow
```

## 3. 소스 코드 배포

제공된 소스 코드 패키지를 다운로드하고 압축을 풀어 프로젝트 디렉토리에 배치합니다:

```bash
# 소스 코드 압축 파일 다운로드 (URL은 실제 다운로드 링크로 변경해야 함)
wget https://example.com/drawing-analyzer-app.zip -O drawing-analyzer-app.zip

# 압축 해제
unzip drawing-analyzer-app.zip -d .

# 또는 제공된 ZIP 파일을 직접 복사하여 압축 해제
```

## 4. 애플리케이션 실행

애플리케이션을 실행하는 방법:

```bash
# 가상 환경이 활성화되어 있는지 확인
source venv/bin/activate

# 애플리케이션 실행
cd ~/drawing-analyzer-app
python -m src.main
```

이제 웹 브라우저에서 `http://localhost:5000`으로 접속하여 애플리케이션을 사용할 수 있습니다.

## 5. 외부 접속 설정

외부에서 접속할 수 있도록 설정하려면:

```bash
# 모든 IP에서 접속 가능하도록 실행
python -m src.main --host=0.0.0.0
```

이제 서버의 IP 주소와 포트 5000을 통해 외부에서 접속할 수 있습니다: `http://서버IP:5000`

## 6. 서비스로 등록 (선택 사항)

시스템 서비스로 등록하여 서버 재시작 시에도 자동으로 실행되도록 설정할 수 있습니다:

```bash
# 서비스 파일 생성
sudo nano /etc/systemd/system/drawing-analyzer.service
```

다음 내용을 입력합니다 (경로는 실제 설치 경로에 맞게 수정):

```
[Unit]
Description=Drawing Analyzer Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/drawing-analyzer-app
ExecStart=/home/ubuntu/drawing-analyzer-app/venv/bin/python -m src.main --host=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

서비스를 활성화하고 시작합니다:

```bash
sudo systemctl daemon-reload
sudo systemctl enable drawing-analyzer
sudo systemctl start drawing-analyzer
```

## 7. 문제 해결

### Tesseract OCR 오류

Tesseract OCR이 제대로 설치되지 않았거나 PATH에 없는 경우:

```bash
# Tesseract 설치 확인
tesseract --version

# 설치되어 있지 않다면 다시 설치
sudo apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
```

### OpenCV 관련 오류

OpenCV 의존성 문제가 발생하는 경우:

```bash
# 추가 의존성 설치
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# 가상 환경에서 OpenCV 재설치
pip uninstall opencv-python
pip install opencv-python-headless
```

### 메모리 부족 오류

이미지 처리 중 메모리 부족 오류가 발생하는 경우:

```bash
# 스왑 공간 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 8. 보안 고려사항

프로덕션 환경에서는 다음 보안 조치를 고려하세요:

1. HTTPS 설정을 위해 Nginx와 Let's Encrypt 사용
2. 방화벽 설정으로 필요한 포트만 개방
3. 정기적인 시스템 및 패키지 업데이트 수행

## 9. 백업 및 복원

정기적인 백업을 위한 스크립트 예시:

```bash
#!/bin/bash
BACKUP_DIR="/backup/drawing-analyzer"
DATE=$(date +%Y%m%d)
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/drawing-analyzer-$DATE.tar.gz /home/ubuntu/drawing-analyzer-app
```

이 스크립트를 cron에 등록하여 정기적으로 실행할 수 있습니다.

## 10. 추가 지원

추가 지원이 필요하시면 개발자에게 문의하세요.
