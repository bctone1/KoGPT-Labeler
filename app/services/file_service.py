from pathlib import Path
import fitz  # PyMuPDF
import hashlib
from typing import List, Tuple
import os
import logging
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from app.utils.helpers import (
    calculate_file_hash,
    get_file_info,
    chunked_read,
    clean_text
)


class FileService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def process_pdf(self, file_path: Path) -> str:
        """PDF 파일 텍스트 추출"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf:
                    text += page.get_text()
                pdf.close()
            # helpers의 clean_text 활용하여 추출된 텍스트 정제
            return clean_text(text)
        except Exception as e:
            self.logger.error(f"PDF 처리 중 오류 ({file_path}): {str(e)}")
            raise

    def process_txt(self, file_path: Path) -> str:
        """TXT 파일 텍스트 추출"""
        encodings = ['utf-8', 'cp949', 'euc-kr']
        for encoding in encodings:
            try:
                # 대용량 파일을 위한 청크 단위 읽기 활용
                text = chunked_read(file_path)
                return clean_text(text)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"지원되지 않는 파일 인코딩: {file_path}")

    def save_uploaded_file(self, file, upload_dir: Path) -> Tuple[Path, str, dict]:
        """
        업로드된 파일 저장 및 메타데이터 반환

        Args:
            file: 업로드된 파일 객체
            upload_dir (Path): 저장할 디렉토리 경로

        Returns:
            Tuple[Path, str, dict]: (저장된 파일 경로, 안전한 파일명, 파일 메타데이터)

        Raises:
            ValueError: 지원하지 않는 파일 형식
            Exception: 파일 저장 중 발생한 기타 오류
        """
        try:
            # 디렉토리 생성
            upload_dir.mkdir(parents=True, exist_ok=True)

            # 안전한 파일명 생성
            original_filename = file.filename
            safe_filename = secure_filename(original_filename)

            # 확장자 확인 및 검증
            ext = Path(safe_filename).suffix.lower()
            if ext not in ['.txt', '.pdf']:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

            # 파일명 중복 처리
            base_filename = safe_filename
            counter = 1
            while (upload_dir / safe_filename).exists():
                name_parts = os.path.splitext(base_filename)
                safe_filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
                counter += 1

            file_path = upload_dir / safe_filename

            # 파일 저장
            try:
                file.seek(0)  # 파일 포인터 초기화
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(file.stream, f)
            except Exception as e:
                self.logger.error(f"파일 저장 실패 ({original_filename}): {str(e)}")
                if file_path.exists():
                    file_path.unlink()  # 실패한 파일 삭제
                raise

            # 파일 메타데이터 수집
            file_info = get_file_info(file_path)

            # 무결성 검증을 위한 해시값 계산
            file_hash = calculate_file_hash(file_path)

            # 메타데이터 보강
            file_info.update({
                'original_filename': original_filename,
                'safe_filename': safe_filename,
                'hash': file_hash,
                'extension': ext,
                'uploaded_at': datetime.now().isoformat(),
                'status': 'uploaded'
            })

            self.logger.info(
                f"파일 저장 완료: {original_filename} -> {safe_filename}\n"
                f"위치: {file_path}\n"
                f"크기: {file_info['size']:,} bytes\n"
                f"해시: {file_hash}"
            )

            return file_path, safe_filename, file_info

        except ValueError as e:
            self.logger.error(f"파일 형식 오류 ({file.filename}): {str(e)}")
            raise

        except Exception as e:
            self.logger.error(f"파일 저장 중 오류 ({file.filename}): {str(e)}")
            raise

    def process_file(self, file_path: Path) -> str:
        """파일 형식에 따른 처리"""
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext == '.txt':
            return self.process_txt(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    def clean_old_files(self, directory: Path, max_age_days: int = 7) -> List[Path]:
        """오래된 파일 정리"""
        cleaned_files = []
        try:
            current_time = datetime.now()
            for filepath in directory.glob('*.*'):
                if filepath.is_file():
                    file_age = current_time - datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_age > timedelta(days=max_age_days):
                        filepath.unlink()
                        cleaned_files.append(filepath)
                        self.logger.info(f"파일 삭제: {filepath}")
            return cleaned_files
        except Exception as e:
            self.logger.error(f"파일 정리 중 오류: {str(e)}")
            raise