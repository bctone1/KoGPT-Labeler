import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from datetime import datetime


def setup_logging(app=None):
    """로깅 설정"""
    try:
        # 로그 디렉토리 생성
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일명 설정 (날짜별)
        current_date = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f'app_{current_date}.log'

        # 로거 설정
        logger = logging.getLogger('chatgpt_finetuning')
        logger.setLevel(logging.INFO)

        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 포맷터 설정
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        )

        # 파일 핸들러 설정
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8',
            delay=True
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Flask 앱이 제공된 경우 웹 로거 설정
        if app:
            # Flask 앱의 기본 핸들러 제거
            app.logger.handlers = []
            for handler in logger.handlers:
                app.logger.addHandler(handler)

        logger.info('Logging setup completed')
        return logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        # 기본 로거 반환
        return logging.getLogger()


def get_logger(name: str = None):
    """로거 인스턴스 반환"""
    if name:
        return logging.getLogger(f'chatgpt_finetuning.{name}')
    return logging.getLogger('chatgpt_finetuning')
