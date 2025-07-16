"""텍스트 전처리 유틸리티"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespaces and normalizing format.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text or text.strip() == "":
        return ""
    
    # 문자열이 아니면 문자열로 변환
    text = str(text)
    
    # 여분의 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text for BLEU evaluation.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    text = clean_text(text)
    
    if not text:
        return ""
    
    # 일관된 비교를 위해 소문자로 변환
    text = text.lower()
    
    # 구두점 주변 여분의 공백 제거
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    
    # 최종 정리
    text = text.strip()
    
    return text