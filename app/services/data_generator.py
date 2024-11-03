from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding  # 수정된 임포트 경로
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI as LlamaOpenAI  # 수정된 임포트 경로
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
from app.config import config
from app.services.quality_checker import QualityChecker
from app.utils.logging_config import get_logger
from app.utils.helpers import clean_text, save_jsonl, load_jsonl, check_memory_usage, clear_gpu_memory



class ChatGPTDataGenerator:
    def __init__(self):
        """ChatGPT 데이터 생성기 초기화"""
        # 로거 설정
        self.logger = get_logger('data_generator')
        self.logger.info('데이터 생성기 초기화')

        # OpenAI 설정
        try:
            import openai
            openai.api_key = config.OPENAI_API_KEY
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.logger.info("OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            self.logger.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
            raise

        # 품질 검사기 초기화
        self.quality_checker = QualityChecker()

        # GPU/CPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_GPU else 'cpu')
        self.logger.info(f"장치 설정: {self.device}")

        # GPU 메모리 정리 후 시작
        clear_gpu_memory()

        # KLUE RoBERTa 모델 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
            self.model = AutoModel.from_pretrained('klue/roberta-large').to(self.device)

            # 메모리 사용량 확인
            memory_usage = check_memory_usage()
            self.logger.info(f"모델 로드 후 메모리 사용량: {memory_usage}")

            if self.device.type == 'cuda':
                self.model = torch.compile(self.model)  # PyTorch 2.0 이상에서 성능 최적화
            self.logger.info("KLUE RoBERTa 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise

    def generate_conversation(
            self,
            context: str,
            task_template: Dict[str, str],
            max_retries: int = 2,
            timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """ChatGPT를 사용하여 대화 생성"""
        start_time = time.time()

        try:
            # 컨텍스트 전처리
            context = clean_text(context)  # helpers의 clean_text 활용
            if len(context) > 2000:  # 컨텍스트 길이 제한
                context = context[:2000] + "..."

            # 프롬프트 구성
            prompt = f"""다음 문서를 참고하여 자연스러운 한국어 대화를 생성해주세요.

규칙:
1. 시스템 역할: {task_template['system_prompt']}
2. 최소 1회 이상의 대화 (질문-답변)를 포함할 것
3. 실제 대화처럼 자연스러운 한국어 사용
4. 문서의 내용을 참고하되 단순 복사는 피할 것
5. 개인정보(이메일, 전화번호 등)를 포함하지 않을 것

문서 내용:
{context}

대화 예시 형식:
사용자: (질문/요청)
상담사: (답변/설명)
"""

            # GPT 호출 (최대 재시도)
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": task_template['system_prompt']
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=800,
                        timeout=timeout
                    )

                    content = response.choices[0].message.content.strip()

                    # 대화 형식으로 변환
                    conversation = {
                        "messages": [
                            {
                                "role": "system",
                                "content": task_template['system_prompt']
                            }
                        ]
                    }

                    # 대화 파싱
                    lines = content.split('\n')
                    current_role = None
                    current_content = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith('사용자:'):
                            if current_role and current_content:
                                conversation["messages"].append({
                                    "role": current_role,
                                    "content": '\n'.join(current_content).strip()
                                })
                            current_role = "user"
                            current_content = [line.replace('사용자:', '').strip()]
                        elif line.startswith('상담사:'):
                            if current_role and current_content:
                                conversation["messages"].append({
                                    "role": current_role,
                                    "content": '\n'.join(current_content).strip()
                                })
                            current_role = "assistant"
                            current_content = [line.replace('상담사:', '').strip()]
                        else:
                            if current_content:
                                current_content.append(line)

                    # 마지막에 메시지 추가
                    if current_role and current_content:
                        conversation["messages"].append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })

                    # 기본 검증
                    if len(conversation["messages"]) >= 3:  # system + 최소 1쌍의 대화
                        return conversation

                    self.logger.warning(f"대화 생성 실패 (시도 {attempt + 1}/{max_retries}): 형식 불일치")

                except Exception as e:
                    self.logger.error(f"GPT 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise

            return None

        except Exception as e:
            self.logger.error(f"대화 생성 중 오류: {str(e)}")
            return None

        finally:
            processing_time = time.time() - start_time
            self.logger.info(f"대화 생성 시도 완료 - 처리 시간: {processing_time:.2f}초")


    def generate_dataset(
            self,
            texts: List[str],
            output_dir: Path,
            task_name: str,
            num_examples: int,
            progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, Any]:
        """데이터셋 생성"""
        start_time = time.time()
        self.logger.info(f"데이터셋 생성 시작 - 태스크: {task_name}, 목표: {num_examples}개")

        try:
            # 문서 생성
            documents = [Document(text=text) for text in texts]
            self.logger.info(f"문서 로드 완료 - {len(documents)}개 파일")

            # OpenAI 임베딩 및 LLM 설정
            embed_model = OpenAIEmbedding(
                api_key=config.OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )

            llm = LlamaOpenAI(
                api_key=config.OPENAI_API_KEY,
                model="gpt-4-1106-preview",
                temperature=0.7
            )

            # ServiceContext 설정
            service_context = ServiceContext.from_defaults(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                llm=llm,
                embed_model=embed_model
            )

            # 청크 분할
            parser = SimpleNodeParser.from_defaults(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            nodes = parser.get_nodes_from_documents(documents)
            self.logger.info(f"청크 분할 완료 - {len(nodes)}개 청크")

            # 인덱스 생성
            index = VectorStoreIndex(
                nodes,
                service_context=service_context
            )
            self.logger.info("인덱스 생성 완료")

            # 태스크 템플릿
            task_template = config.TASK_TEMPLATES.get(task_name)
            if not task_template:
                raise ValueError(f"지원하지 않는 태스크입니다: {task_name}")

            conversations = []
            stats = {
                "generated": 0,
                "failed": 0,
                "duplicates": 0,
                "processing_time": 0
            }

            # 진행률 업데이트 함수
            def update_progress():
                if progress_callback:
                    progress = int(len(conversations) * 100 / num_examples)
                    progress_callback(progress)

            # 대화 생성
            with tqdm(total=num_examples) as pbar:
                while len(conversations) < num_examples:
                    try:
                        # 랜덤 청크 선택
                        node = np.random.choice(list(index.docstore.docs.values()))
                        iter_start = time.time()

                        # 대화 생성
                        conversation = self.generate_conversation(node.text, task_template)

                        if conversation:
                            # 품질 검사
                            if self.quality_checker.check_conversation(conversation):
                                # 중복 검사
                                if not self.quality_checker.is_duplicate(conversation, conversations):
                                    conversations.append(conversation)
                                    stats["generated"] += 1
                                    pbar.update(1)
                                    update_progress()
                                else:
                                    stats["duplicates"] += 1
                            else:
                                stats["failed"] += 1
                        else:
                            stats["failed"] += 1

                        # 처리 시간 기록
                        iter_time = time.time() - iter_start
                        stats["processing_time"] += iter_time

                        # 진행 상황 로깅
                        self.logger.info(
                            f"진행 상황 - 생성: {stats['generated']}, "
                            f"실패: {stats['failed']}, "
                            f"중복: {stats['duplicates']}, "
                            f"반복 시간: {iter_time:.2f}초"
                        )

                    except Exception as e:
                        self.logger.error(f"반복 중 오류: {str(e)}")
                        stats["failed"] += 1

            # 결과 저장
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{task_name}_{timestamp}.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + '\n')

            # 최종 통계
            total_time = time.time() - start_time
            stats.update({
                "output_file": str(output_file),
                "total_time": total_time,
                "avg_time_per_example": total_time / max(stats["generated"], 1)
            })

            self.logger.info(
                f"데이터셋 생성 완료:\n"
                f"- 생성된 예제: {stats['generated']}\n"
                f"- 실패: {stats['failed']}\n"
                f"- 중복 제외: {stats['duplicates']}\n"
                f"- 총 처리 시간: {total_time:.2f}초"
            )

            return stats

        except Exception as e:
            self.logger.error(f"데이터셋 생성 실패: {str(e)}")
            raise