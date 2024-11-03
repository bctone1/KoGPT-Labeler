import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from app.config import config


class QualityChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 한국어 패턴
        self.korean_pattern = re.compile('[가-힣]')

        # 개인정보 패턴
        self.patterns = {
            'email': re.compile(r'[\w\.-]+@[\w\.-]+\.\w+'),
            'phone': re.compile(r'\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        }

    def check_korean_ratio(self, text: str) -> float:
        """한글 비율 검사 (기준 완화)"""
        if not text:
            return 0
        korean_chars = len(self.korean_pattern.findall(text))
        total_chars = len(text.strip())
        ratio = korean_chars / total_chars if total_chars > 0 else 0
        self.logger.debug(f"한글 비율: {ratio:.2f}")
        return ratio

    def check_message_length(self, text: str) -> bool:
        """메시지 길이 검사 (기준 완화)"""
        if not text:
            return False
        words = text.strip().split()
        valid = 3 <= len(words) <= 200  # 최소 3단어, 최대 200단어
        self.logger.debug(f"메시지 길이: {len(words)} 단어, 유효: {valid}")
        return valid

    def check_personal_info(self, text: str) -> bool:
        """개인정보 포함 여부 검사"""
        if not text:
            return True

        for pattern_name, pattern in self.patterns.items():
            if pattern.search(text):
                self.logger.warning(f"개인정보 발견: {pattern_name}")
                return False
        return True

    def check_message_format(self, message: dict) -> bool:
        """메시지 형식 검사"""
        required_fields = {'role', 'content'}
        valid_roles = {'system', 'user', 'assistant'}

        # 필수 필드 확인
        if not all(field in message for field in required_fields):
            self.logger.warning("필수 필드 누락")
            return False

        # role 값 확인
        if message['role'] not in valid_roles:
            self.logger.warning(f"잘못된 role 값: {message['role']}")
            return False

        # content가 문자열인지 확인
        if not isinstance(message['content'], str):
            self.logger.warning("content가 문자열이 아님")
            return False

        return True

    def check_conversation_structure(self, conversation: dict) -> bool:
        """대화 구조 검사"""
        if 'messages' not in conversation:
            self.logger.warning("messages 필드 누락")
            return False

        messages = conversation['messages']
        if not messages:
            self.logger.warning("메시지가 비어있음")
            return False

        # system 메시지 확인
        if messages[0]['role'] != 'system':
            self.logger.warning("첫 메시지가 system이 아님")
            return False

        # 최소 한 쌍의 대화 확인
        if len(messages) < 3:
            self.logger.warning("대화 쌍이 부족함")
            return False

        return True

    def check_conversation(self, conversation: dict) -> bool:
        """대화 전체 품질 검사"""
        try:
            # 구조 검사
            if not self.check_conversation_structure(conversation):
                return False

            messages = conversation['messages']
            system_message = messages[0]

            # system 메시지 검사
            if not self.check_message_format(system_message):
                return False

            # 대화 메시지 검사
            for message in messages[1:]:
                # 메시지 형식 검사
                if not self.check_message_format(message):
                    return False

                content = message['content']

                # 기본 품질 검사
                if not content.strip():
                    self.logger.warning("빈 메시지")
                    return False

                # 한글 비율 검사 (30% 이상)
                if self.check_korean_ratio(content) < 0.3:
                    self.logger.warning("한글 비율 부족")
                    return False

                # 메시지 길이 검사
                if not self.check_message_length(content):
                    self.logger.warning("메시지 길이 부적절")
                    return False

                # 개인정보 검사
                if not self.check_personal_info(content):
                    self.logger.warning("개인정보 포함")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"품질 검사 중 오류 발생: {str(e)}")
            return False

    def is_duplicate(self, new_conv: dict, existing_convs: list, threshold: float = 0.85) -> bool:
        """중복 검사 (기준 완화)"""
        if not existing_convs:
            return False

        try:
            # 새 대화의 텍스트 추출
            new_text = ' '.join([m['content'] for m in new_conv['messages'][1:]])  # system 메시지 제외

            for conv in existing_convs:
                # 기존 대화의 텍스트 추출
                existing_text = ' '.join([m['content'] for m in conv['messages'][1:]])

                # 완전히 동일한 경우
                if new_text == existing_text:
                    self.logger.warning("완전히 동일한 대화 발견")
                    return True

                # 간단한 유사도 검사
                similarity = len(set(new_text.split()) & set(existing_text.split())) / len(set(new_text.split()))
                if similarity > threshold:
                    self.logger.warning(f"유사한 대화 발견 (유사도: {similarity:.2f})")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"중복 검사 중 오류 발생: {str(e)}")
            return False