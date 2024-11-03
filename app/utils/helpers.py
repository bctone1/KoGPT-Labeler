import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from datetime import datetime
import psutil
import hashlib
from app.utils.logging_config import get_logger

logger = get_logger('helpers')


def clean_text(text: str) -> str:
    """텍스트 전처리 및 정제"""
    try:
        # 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()

        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)

        # 특수문자 처리
        text = re.sub(r'[^\w\s\.,!?가-힣]', '', text)

        # 중복 문장부호 정리
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        return text
    except Exception as e:
        logger.error(f"텍스트 정제 중 오류: {str(e)}")
        return text


def check_memory_usage() -> Dict[str, float]:
    """메모리 사용량 확인"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        memory_usage = {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }

        # GPU 메모리 확인 (torch 사용 시)
        if torch.cuda.is_available():
            memory_usage['gpu'] = {
                'allocated': torch.cuda.memory_allocated(0) / 1024 / 1024,  # MB
                'cached': torch.cuda.memory_reserved(0) / 1024 / 1024  # MB
            }

        return memory_usage

    except Exception as e:
        logger.error(f"메모리 사용량 확인 중 오류: {str(e)}")
        return {}


def calculate_file_hash(file_path: Path) -> str:
    """파일 해시값 계산"""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"파일 해시 계산 중 오류 ({file_path}): {str(e)}")
        return ""


def save_jsonl(data: List[Dict[str, Any]], output_path: Path, append: bool = False) -> bool:
    """JSONL 형식으로 데이터 저장"""
    try:
        mode = 'a' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        logger.error(f"JSONL 저장 중 오류 ({output_path}): {str(e)}")
        return False


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일 로드"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"JSONL 로드 중 오류 ({file_path}): {str(e)}")
        return []


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """타임스탬프 포맷팅"""
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """파일 정보 조회"""
    try:
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'created': format_timestamp(stat.st_ctime),
            'modified': format_timestamp(stat.st_mtime),
            'hash': calculate_file_hash(file_path)
        }
    except Exception as e:
        logger.error(f"파일 정보 조회 중 오류 ({file_path}): {str(e)}")
        return {}


def chunked_read(file_path: Path, chunk_size: int = 8192) -> str:
    """대용량 파일 청크 단위 읽기"""
    try:
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content.append(chunk)
        return ''.join(content)
    except Exception as e:
        logger.error(f"파일 읽기 중 오류 ({file_path}): {str(e)}")
        return ""


def clear_gpu_memory():
    """GPU 메모리 정리"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU 메모리 정리 완료")
    except Exception as e:
        logger.error(f"GPU 메모리 정리 중 오류: {str(e)}")