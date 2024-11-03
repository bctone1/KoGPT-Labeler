document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const progressBar = document.querySelector('.progress');
    const result = document.getElementById('result');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const submitBtn = document.getElementById('submitBtn');

            // 버튼 비활성화
            submitBtn.disabled = true;
            submitBtn.innerHTML = '처리 중...';

            // 진행바 표시
            progressBar.style.display = 'block';
            const progressBarInner = progressBar.querySelector('.progress-bar');
            progressBarInner.style.width = '0%';

            // 파일 업로드 및 데이터셋 생성 요청
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    result.className = 'alert alert-success';
                    result.textContent = data.message;

                    // 통계 정보 표시
                    if (data.stats) {
                        const statsHtml = `
                            <div class="mt-3">
                                <h4>생성 결과</h4>
                                <ul>
                                    <li>생성된 예제: ${data.stats.generated}</li>
                                    <li>중복 제외: ${data.stats.duplicates}</li>
                                    <li>실패: ${data.stats.failed}</li>
                                    <li>저장 위치: ${data.stats.output_file}</li>
                                </ul>
                            </div>
                        `;
                        result.innerHTML += statsHtml;
                    }
                } else {
                    result.className = 'alert alert-danger';
                    result.textContent = data.error || '오류가 발생했습니다.';
                }
            })
            .catch(error => {
                result.className = 'alert alert-danger';
                result.textContent = '서버 오류가 발생했습니다.';
                console.error('Error:', error);
            })
            .finally(() => {
                // 버튼 활성화
                submitBtn.disabled = false;
                submitBtn.innerHTML = '데이터셋 생성';
                progressBar.style.display = 'none';
            });
        });
    }

    // 피드백 관련 기능
    const feedbackButtons = document.querySelectorAll('.feedback-btn');
    const feedbackModal = document.getElementById('feedbackModal');
    const submitFeedbackBtn = document.getElementById('submitFeedback');

    if (feedbackButtons.length > 0) {
        feedbackButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const dataset = this.dataset.dataset;
                const exampleId = this.dataset.exampleId;

                document.getElementById('feedbackDataset').value = dataset;
                document.getElementById('feedbackExampleId').value = exampleId;

                const modal = new bootstrap.Modal(feedbackModal);
                modal.show();
            });
        });
    }

    if (submitFeedbackBtn) {
        submitFeedbackBtn.addEventListener('click', function() {
            const formData = new FormData(document.getElementById('feedbackForm'));
            const feedbackData = Object.fromEntries(formData.entries());

            fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    bootstrap.Modal.getInstance(feedbackModal).hide();
                    alert('피드백이 제출되었습니다.');
                } else {
                    alert(data.error || '오류가 발생했습니다.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('서버 오류가 발생했습니다.');
            });
        });
    }
});