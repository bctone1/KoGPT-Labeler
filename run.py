from app import create_app
from app.config import config
import torch
from pathlib import Path


def check_gpu_status():
    """GPU 상태 체크 및 출력"""
    if torch.cuda.is_available():
        print("\n=== GPU 정보 ===")
        print(f"GPU 사용 가능: {torch.cuda.is_available()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        print("==============\n")
    else:
        print("\nGPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.\n")


def check_gpu():
    if torch.cuda.is_available() and config.USE_GPU:
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")


def init_directories():
    """필요한 디렉토리 생성"""
    directories = [
        config.UPLOAD_FOLDER,
        config.GENERATED_FOLDER,
        config.FEEDBACK_FOLDER
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # GPU 상태 체크
    check_gpu_status()

    # 디렉토리 초기화
    init_directories()

    # Flask 애플리케이션 실행
    app = create_app()
    app.run(debug=True, port=5000)