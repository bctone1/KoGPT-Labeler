import torch
from pathlib import Path

class Config:
    # 기본 설정
    BASE_DIR = Path(__file__).resolve().parent.parent
    SECRET_KEY = 'dev-key-change-in-production'

    # OpenAI API 설정
    OPENAI_API_KEY = "..."  # 여기에 실제 API 키를 입력하세요

    # 파일 업로드 설정
    UPLOAD_FOLDER = BASE_DIR / 'data' / 'uploads'
    GENERATED_FOLDER = BASE_DIR / 'data' / 'generated'
    FEEDBACK_FOLDER = BASE_DIR / 'data' / 'feedback'
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # GPU 설정
    USE_GPU = torch.cuda.is_available()
    CUDA_DEVICE = "cuda:0" if USE_GPU else "cpu"
    CUDA_VISIBLE_DEVICES = "0" if USE_GPU else ""

    @classmethod
    def get_device(cls):
        if cls.USE_GPU:
            print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 버전: {torch.version.cuda}")
            return torch.device(cls.CUDA_DEVICE)
        else:
            print("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            return torch.device("cpu")

    # 태스크 템플릿
    TASK_TEMPLATES = {
        'customer_service': {
            'name': '고객 상담',
            'system_prompt': '당신은 전문적이고 친절한 고객 상담사입니다',
            'example_user': '이 제품의 배터리 수명은 얼마나 되나요?',
            'example_assistant': '본 제품의 완충 시 배터리 사용 시간은 약 8시간입니다. 사용 환경에 따라 달라질 수 있습니다.'
        },
        'product_recommendation': {
            'name': '상품 추천',
            'system_prompt': '당신은 전문적인 쇼핑 어드바이저입니다.',
            'example_user': '20만원 대의 블루투스 이어폰 추천해주세요.',
            'example_assistant': '예산에 맞는 추천 제품들을 알려드리겠습니다.'
        },
        'technical_support': {
            'name': '기술 지원',
            'system_prompt': '당신은 IT 기술 지원 전문가입니다.',
            'example_user': '윈도우 업데이트 후 프린터가 작동하지 않아요.',
            'example_assistant': '문제 해결을 위해 단계별로 도와드리겠습니다.'
        }
    }

    # 데이터 생성 설정
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MAX_EXAMPLES_PER_TASK = 1000
    DEFAULT_EXAMPLES_PER_TASK = 100

    # 품질 검증 설정
    MIN_QUALITY_SCORE = 0.7
    SIMILARITY_THRESHOLD = 0.85

    # 파일 업로드 관련 설정
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 제한
    MAX_TOTAL_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB 전체 제한
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    UPLOAD_FOLDER = Path(__file__).parent.parent / 'data' / 'uploads'
    GENERATED_FOLDER = Path(__file__).parent.parent / 'data' / 'generated'


config = Config()