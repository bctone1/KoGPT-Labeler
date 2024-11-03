from flask import Blueprint, render_template, request, jsonify, current_app, send_file, url_for
from werkzeug.utils import secure_filename
import json
from pathlib import Path
from datetime import datetime
import logging
from app.config import config
from app.services.file_service import FileService
from app.services.data_generator import ChatGPTDataGenerator
from app.utils.helpers import clean_text
import threading
import shutil
import tempfile
import os
import io
from queue import Queue
from typing import List, Dict, Any

from app.utils.logging_config import get_logger

# Blueprint 및 서비스 초기화
bp = Blueprint('main', __name__)
file_service = FileService()
data_generator = ChatGPTDataGenerator()
logger = get_logger('routes')

# 진행 중인 작업 상태 저장
active_tasks = {}


def allowed_file(filename):
    """파일 확장자 검사"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf'}


# 메서드를 명시적으로 지정합니다.
@bp.route('/', methods=['GET'])
def index():
    """홈페이지"""
    try:
        logger.info('홈페이지 접속')
        return render_template('upload.html', config=config)
    except Exception as e:
        logger.error(f"홈페이지 렌더링 중 오류: {str(e)}")
        return str(e), 500


# POST 메서드를 명시적으로 지정합니다.
@bp.route('/upload', methods=['POST'])
def upload():
    """파일 업로드 처리"""
    try:
        logger.info("파일 업로드 요청 받음")

        # 파일 존재 확인
        if 'files[]' not in request.files:
            logger.error("요청에 'files[]'가 없음")
            return jsonify({'error': '파일이 없습니다.'}), 400

        uploaded_files = request.files.getlist('files[]')
        if not uploaded_files or not any(f.filename for f in uploaded_files):
            logger.error("선택된 파일 없음")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

        # 파일 정보 복사
        files_info = []
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # 파일 내용을 BytesIO 객체로 복사
                file_content = io.BytesIO(file.read())
                file_content.filename = file.filename
                files_info.append({
                    'file': file_content,
                    'original_filename': file.filename
                })
            file.close()  # 원본 파일 객체 닫기

        if not files_info:
            logger.error("처리 가능한 파일 없음")
            return jsonify({'error': '처리 가능한 파일이 없습니다.'}), 400

        # 태스크 정보 확인
        task_name = request.form.get('task_name')
        if not task_name or task_name not in config.TASK_TEMPLATES:
            logger.error(f"잘못된 태스크: {task_name}")
            return jsonify({'error': '잘못된 태스크입니다.'}), 400

        # 예제 수 확인
        try:
            num_examples = int(request.form.get('num_examples', config.DEFAULT_EXAMPLES_PER_TASK))
            if num_examples > config.MAX_EXAMPLES_PER_TASK:
                return jsonify({'error': f'최대 {config.MAX_EXAMPLES_PER_TASK}개까지만 생성 가능합니다.'}), 400
        except ValueError:
            return jsonify({'error': '잘못된 예제 수입니다.'}), 400

        # 작업 ID 생성
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 비동기 처리 시작
        thread = threading.Thread(
            target=process_files_unified,
            args=(task_id, files_info, task_name, num_examples)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': '파일 처리가 시작되었습니다.',
            'total_files': len(files_info),
            'status_url': url_for('main.check_status', task_id=task_id, _external=True)
        })

    except Exception as e:
        logger.error(f"업로드 처리 중 오류: {str(e)}")
        return jsonify({
            'error': '파일 업로드 중 오류가 발생했습니다.',
            'detail': str(e)
        }), 500


def process_files_unified(task_id: str, files_info: List[dict], task_name: str, num_examples: int):
    """
    통합된 파일 처리 및 데이터셋 생성 함수

    Args:
        task_id (str): 작업 식별자
        files (List[dict]): 처리할 파일 목록
        task_name (str): 생성할 태스크 이름
        num_examples (int): 생성할 예제 수
    """
    temp_dir = None
    try:
        # 작업 상태 초기화
        active_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '파일 처리 중...',
            'files_processed': 0,
            'total_files': len(files_info),
            'processed_list': [],
            'failed_files': [],
            'messages': [],
            'started_at': datetime.now().isoformat()
        }

        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp(prefix='chatgpt_data_')
        processed_files = []

        # 파일 처리
        for idx, file_info in enumerate(files_info):
            try:
                file_content = file_info.get('file')
                if not file_content:
                    continue

                # 파일명 처리
                original_filename = file_info['original_filename']
                safe_filename = secure_filename(original_filename)
                logger.info(f"파일 처리 시작: {original_filename} -> {safe_filename}")

                # 임시 파일 저장
                temp_path = Path(temp_dir) / safe_filename

                # BytesIO의 내용을 파일로 저장
                file_content.seek(0)
                with open(temp_path, 'wb') as f:
                    f.write(file_content.read())

                # uploads 폴더에 복사
                final_path = config.UPLOAD_FOLDER / safe_filename
                config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

                # 파일명 중복 처리
                if final_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_parts = os.path.splitext(safe_filename)
                    safe_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
                    final_path = config.UPLOAD_FOLDER / safe_filename

                # 임시 파일을 최종 위치로 복사
                shutil.copy2(temp_path, final_path)

                # 파일 메타데이터 수집
                file_meta = {
                    'size': os.path.getsize(final_path),
                    'created_at': datetime.fromtimestamp(os.path.getctime(final_path)).isoformat(),
                    'modified_at': datetime.fromtimestamp(os.path.getmtime(final_path)).isoformat()
                }

                # 처리된 파일 정보 저장
                processed_file = {
                    'original_filename': original_filename,
                    'safe_filename': safe_filename,
                    'path': str(final_path),
                    'meta': file_meta
                }
                processed_files.append(processed_file)

                # 작업 상태 업데이트
                active_tasks[task_id].update({
                    'files_processed': idx + 1,
                    'progress': int((idx + 1) * 50 / len(files)),
                    'processed_list': processed_files,
                    'message': f'파일 처리 중... ({idx + 1}/{len(files)})'
                })

            except Exception as e:
                error_msg = f"파일 '{original_filename}' 처리 실패: {str(e)}"
                logger.error(error_msg)
                active_tasks[task_id]['failed_files'].append({
                    'filename': original_filename,
                    'error': str(e)
                })
                active_tasks[task_id]['messages'].append(error_msg)
                continue

        if not processed_files:
            raise ValueError('처리할 수 있는 파일이 없습니다.')

        logger.info(f"총 {len(processed_files)}개 파일 처리 완료")

        # 데이터셋 생성 단계
        active_tasks[task_id].update({
            'status': 'generating',
            'message': '데이터셋 생성 중...',
            'progress': 50
        })

        # 파일 처리 및 텍스트 추출
        all_texts = []
        for file_info in processed_files:
            try:
                with open(file_info['path'], 'rb') as f:  # 바이너리 모드로 읽기
                    content = f.read()
                    if file_info['path'].endswith('.pdf'):
                        text = file_service.process_pdf(Path(file_info['path']))
                    else:
                        text = content.decode('utf-8')  # txt 파일인 경우

                    text = clean_text(text)
                    if text.strip():
                        all_texts.append(text)
            except Exception as e:
                logger.error(f"텍스트 추출 실패 ({file_info['original_filename']}): {str(e)}")
                continue

        if not all_texts:
            raise ValueError('텍스트를 추출할 수 있는 파일이 없습니다.')

        # 진행률 콜백 함수
        def progress_callback(progress):
            active_tasks[task_id].update({
                'progress': 50 + int(progress * 0.5),
                'message': f'데이터셋 생성 중... {progress}%'
            })

        # 데이터셋 생성
        stats = data_generator.generate_dataset(
            texts=all_texts,  # texts 직접 전달
            output_dir=config.GENERATED_FOLDER,
            task_name=task_name,
            num_examples=num_examples,
            progress_callback=progress_callback
        )

        # 작업 완료 상태 업데이트
        active_tasks[task_id].update({
            'status': 'complete',
            'progress': 100,
            'stats': stats,
            'message': '처리가 완료되었습니다.',
            'completed_at': datetime.now().isoformat(),
            'processing_time': (
                    datetime.fromisoformat(datetime.now().isoformat()) -
                    datetime.fromisoformat(active_tasks[task_id]['started_at'])
            ).total_seconds()
        })

    except Exception as e:
        error_msg = f"데이터셋 생성 중 오류: {str(e)}"
        logger.error(error_msg)
        active_tasks[task_id].update({
            'status': 'error',
            'message': error_msg,
            'error_detail': str(e),
            'failed_at': datetime.now().isoformat()
        })

    finally:
        # 임시 디렉토리 정리
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"임시 디렉토리 정리 중 오류: {str(e)}")


@bp.route('/status/<task_id>', methods=['GET'])
def check_status(task_id):
    """작업 상태 확인"""
    if task_id not in active_tasks:
        return jsonify({'status': 'not_found'})

    return jsonify(active_tasks[task_id])


@bp.route('/review', methods=['GET'])
def review_datasets():
    """생성된 데이터셋 검토"""
    try:
        logger.info('Accessing review page')
        datasets = []

        # 생성된 데이터셋 목록 조회
        if config.GENERATED_FOLDER.exists():
            for filepath in config.GENERATED_FOLDER.glob('*.jsonl'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        examples = []
                        for i, line in enumerate(f):
                            if i >= 10:  # 최대 10개 예제만 표시
                                break
                            examples.append(json.loads(line))

                        datasets.append({
                            'filename': filepath.name,
                            'created_at': datetime.fromtimestamp(filepath.stat().st_mtime),
                            'examples': examples,
                            'total_examples': sum(1 for _ in open(filepath, encoding='utf-8'))
                        })
                    logger.info(f'Loaded dataset: {filepath.name}')
                except Exception as e:
                    logger.error(f'Error loading dataset {filepath}: {str(e)}')
                    continue

        return render_template(
            'review.html',
            datasets=datasets,
            current_time=datetime.now()
        )

    except Exception as e:
        logger.error(f'Error in review page: {str(e)}')
        return render_template('error.html', error=str(e)), 500


@bp.route('/api/datasets/<filename>', methods=['GET'])
def get_dataset_examples(filename):
    """데이터셋의 예제 목록 조회 API"""
    try:
        filepath = config.GENERATED_FOLDER / filename
        if not filepath.exists():
            return jsonify({'error': 'Dataset not found'}), 404

        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        with open(filepath, 'r', encoding='utf-8') as f:
            # 페이지네이션
            start = (page - 1) * per_page
            examples = []

            for i, line in enumerate(f):
                if i >= start + per_page:
                    break
                if i >= start:
                    examples.append(json.loads(line))

            return jsonify({
                'examples': examples,
                'page': page,
                'per_page': per_page
            })

    except Exception as e:
        logger.error(f'Error getting dataset examples: {str(e)}')
        return jsonify({'error': str(e)}), 500


@bp.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """오래된 파일 정리"""
    try:
        days = int(request.form.get('days', 7))

        cleaned = file_service.clean_old_files(
            config.UPLOAD_FOLDER,
            max_age_days=days
        )

        return jsonify({
            'success': True,
            'message': f'{len(cleaned)} 개의 파일이 정리되었습니다.'
        })

    except Exception as e:
        logger.error(f"파일 정리 중 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/load_more/<filename>', methods=['GET'])
def load_more_examples(filename):
    """더 많은 예제 로드"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        logger.info(f"더보기 요청 - 파일: {filename}, 페이지: {page}, 개수: {per_page}")

        # 파일 경로 확인 및 검증
        file_path = config.GENERATED_FOLDER / filename
        if not file_path.exists():
            logger.error(f"파일을 찾을 수 없음: {file_path}")
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

        examples = []
        start_idx = (page - 1) * per_page
        total_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            # 전체 라인 수 계산
            all_lines = f.readlines()
            total_count = len(all_lines)

            # 요청된 페이지의 예제들 처리
            for idx, line in enumerate(all_lines[start_idx:start_idx + per_page]):
                try:
                    example = json.loads(line.strip())
                    example['id'] = start_idx + idx + 1  # 예제 ID 추가
                    examples.append(example)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류 (라인 {start_idx + idx}): {str(e)}")
                    continue

        # 다음 페이지 존재 여부 확인
        has_more = (start_idx + len(examples)) < total_count

        response_data = {
            'examples': examples,
            'has_more': has_more,
            'next_page': page + 1 if has_more else None,
            'total_examples': total_count,
            'current_page': page,
            'loaded_count': start_idx + len(examples),
            'per_page': per_page
        }

        logger.info(f"예제 로드 완료 - 로드된 예제: {len(examples)}, 전체: {total_count}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"예제 로드 중 오류 발생: {str(e)}")
        return jsonify({
            'error': '데이터를 불러오는 중 오류가 발생했습니다.',
            'detail': str(e)
        }), 500


@bp.route('/api/datasets/<filename>/stats', methods=['GET'])
def get_dataset_stats(filename):
    """데이터셋 통계 정보"""
    try:
        logger.info(f"데이터셋 통계 요청 - 파일: {filename}")

        # 파일 경로 확인
        dataset_path = config.GENERATED_FOLDER / filename
        feedback_path = config.FEEDBACK_FOLDER / f"feedback_{filename}.jsonl"

        if not dataset_path.exists():
            logger.error(f"데이터셋 파일 없음: {dataset_path}")
            return jsonify({
                'error': '데이터셋을 찾을 수 없습니다.'
            }), 404

        # 기본 통계 초기화
        stats = {
            'total_examples': 0,
            'total_feedback': 0,
            'quality_scores': [],
            'quality_distribution': {
                'excellent': 0,
                'good': 0,
                'acceptable': 0,
                'poor': 0,
                'unacceptable': 0
            },
            'issue_counts': {
                'grammar': 0,
                'context': 0,
                'consistency': 0,
                'naturalness': 0,
                'completeness': 0
            },
            'avg_quality': 0.0,
            'feedback_rate': 0.0,
            'processed_time': None
        }

        # 전체 예제 수 계산
        with open(dataset_path, 'r', encoding='utf-8') as f:
            stats['total_examples'] = sum(1 for _ in f)

        # 피드백 통계 계산 (피드백 파일이 있는 경우)
        if feedback_path.exists():
            with open(feedback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback = json.loads(line.strip())
                        stats['total_feedback'] += 1

                        # 품질 점수 변환
                        quality_scores = {
                            'excellent': 5,
                            'good': 4,
                            'acceptable': 3,
                            'poor': 2,
                            'unacceptable': 1
                        }

                        # 품질 분포 업데이트
                        rating = feedback.get('quality_rating')
                        if rating in stats['quality_distribution']:
                            stats['quality_distribution'][rating] += 1
                            if rating in quality_scores:
                                stats['quality_scores'].append(quality_scores[rating])

                        # 이슈 카운트 업데이트
                        for issue in feedback.get('issues', []):
                            if issue in stats['issue_counts']:
                                stats['issue_counts'][issue] += 1

                    except json.JSONDecodeError:
                        continue

        # 평균 품질 점수 계산
        if stats['quality_scores']:
            stats['avg_quality'] = sum(stats['quality_scores']) / len(stats['quality_scores'])

        # 피드백 비율 계산
        if stats['total_examples'] > 0:
            stats['feedback_rate'] = (stats['total_feedback'] / stats['total_examples']) * 100

        # 파일 처리 시간 정보
        file_stat = dataset_path.stat()
        stats['processed_time'] = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

        # 품질 분포 백분율 계산
        if stats['total_feedback'] > 0:
            stats['quality_distribution_percent'] = {
                rating: (count / stats['total_feedback'] * 100)
                for rating, count in stats['quality_distribution'].items()
            }
        else:
            stats['quality_distribution_percent'] = {
                rating: 0 for rating in stats['quality_distribution']
            }

        logger.info(f"통계 계산 완료: {stats}")
        return jsonify(stats)

    except Exception as e:
        logger.error(f"통계 계산 중 오류: {str(e)}")
        return jsonify({
            'error': '통계 계산 중 오류가 발생했습니다.',
            'detail': str(e)
        }), 500


@bp.route('/download/<filename>', methods=['GET'])
def download_dataset(filename):
    """생성된 데이터셋 다운로드"""
    try:
        file_path = config.GENERATED_FOLDER / filename
        if not file_path.exists():
            return "파일을 찾을 수 없습니다.", 404

        return send_file(
            file_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"파일 다운로드 중 오류: {str(e)}")
        return str(e), 500