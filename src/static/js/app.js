// 프론트엔드와 백엔드 연동 스크립트
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const changeFileBtn = document.getElementById('change-file-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const dropMessage = document.getElementById('drop-message');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const fileName = document.getElementById('file-name');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const resultsContainer = document.getElementById('results-container');
    const loadingResults = document.getElementById('loading-results');
    const analysisResults = document.getElementById('analysis-results');
    const errorMessage = document.getElementById('error-message');
    const originalImage = document.getElementById('original-image');
    const analyzedImage = document.getElementById('analyzed-image');
    const angleInfo = document.getElementById('angle-info');
    const downloadResultsBtn = document.getElementById('download-results-btn');
    
    let selectedFile = null;
    
    // 드래그 앤 드롭 이벤트 처리
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('active');
    });
    
    dropZone.addEventListener('dragleave', function() {
        dropZone.classList.remove('active');
    });
    
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('active');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // 파일 선택 버튼 클릭 이벤트
    fileSelectBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });
    
    // 파일 변경 버튼 클릭 이벤트
    changeFileBtn.addEventListener('click', function() {
        resetUploadUI();
    });
    
    // 업로드 버튼 클릭 이벤트
    uploadBtn.addEventListener('click', function() {
        if (selectedFile) {
            uploadFile(selectedFile);
        }
    });
    
    // 파일 처리 함수
    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('이미지 파일만 업로드 가능합니다.');
            return;
        }
        
        selectedFile = file;
        
        // 파일 미리보기 표시
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            fileName.textContent = file.name;
            dropMessage.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
    
    // 파일 업로드 함수
    function uploadFile(file) {
        // 업로드 UI 표시
        uploadProgress.classList.remove('hidden');
        
        // FormData 생성
        const formData = new FormData();
        formData.append('image', file);
        
        // 실제 API 호출
        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // 업로드 완료 표시
            progressBarFill.style.width = '100%';
            
            if (!response.ok) {
                throw new Error('서버 오류');
            }
            return response.json();
        })
        .then(data => {
            // 업로드 UI 숨기기
            uploadProgress.classList.add('hidden');
            
            // 결과 표시
            resultsContainer.classList.remove('hidden');
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            uploadProgress.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            showError();
        });
        
        // 업로드 진행 상태 시뮬레이션
        let progress = 0;
        const progressInterval = setInterval(function() {
            progress += 5;
            if (progress > 90) {
                clearInterval(progressInterval);
            }
            progressBarFill.style.width = progress + '%';
        }, 100);
    }
    
    // 분석 결과 표시 함수
    function displayResults(data) {
        loadingResults.classList.add('hidden');
        analysisResults.classList.remove('hidden');
        
        // 원본 이미지 표시
        originalImage.src = '/' + data.original_image;
        
        // 분석된 이미지 표시
        analyzedImage.src = '/' + data.analyzed_image;
        
        // 각도 정보 표시
        const angles = data.angles;
        let angleHtml = `
            <div class="mb-2">
                <span class="font-medium">총 발견된 각도:</span> ${data.total_angles}개
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        `;
        
        angles.forEach((angle, index) => {
            let importanceClass = '';
            switch(angle.importance) {
                case 'critical':
                    importanceClass = 'text-red-600';
                    break;
                case 'high':
                    importanceClass = 'text-orange-500';
                    break;
                case 'warning':
                    importanceClass = 'text-yellow-500';
                    break;
                default:
                    importanceClass = 'text-blue-600';
            }
            
            angleHtml += `
                <div class="bg-white p-3 rounded shadow-sm">
                    <div class="${importanceClass} font-medium">각도 ${index + 1}: ${angle.value}°</div>
                    <div class="text-sm text-gray-600">유형: ${getAngleTypeText(angle.type)}</div>
                    <div class="text-sm text-gray-600">위치: (${angle.position[0]}, ${angle.position[1]})</div>
                    <div class="text-sm text-gray-600 mt-1">${angle.description || '설명 없음'}</div>
                </div>
            `;
        });
        
        angleHtml += '</div>';
        angleInfo.innerHTML = angleHtml;
    }
    
    // 각도 유형 텍스트 변환 함수
    function getAngleTypeText(type) {
        switch(type) {
            case 'direct_notation':
                return '직접 표기';
            case 'text_notation':
                return '텍스트 표기';
            case 'calculated':
                return '계산된 각도';
            default:
                return '알 수 없음';
        }
    }
    
    // 오류 표시 함수
    function showError() {
        loadingResults.classList.add('hidden');
        analysisResults.classList.add('hidden');
        errorMessage.classList.remove('hidden');
    }
    
    // UI 초기화 함수
    function resetUploadUI() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        fileName.textContent = '';
        dropMessage.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        uploadProgress.classList.add('hidden');
        resultsContainer.classList.add('hidden');
        progressBarFill.style.width = '0%';
        loadingResults.classList.remove('hidden');
        analysisResults.classList.add('hidden');
        errorMessage.classList.add('hidden');
    }
    
    // 결과 다운로드 버튼 클릭 이벤트
    downloadResultsBtn.addEventListener('click', function() {
        if (analyzedImage.src) {
            const link = document.createElement('a');
            link.href = analyzedImage.src;
            link.download = 'analyzed_drawing.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });
});
