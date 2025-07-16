"""평가 시스템을 위한 설정 구성"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """BLEU 평가를 위한 설정"""
    
    # 입력 파일
    ground_truth_path: str
    predictions_path: str
    
    # 출력 설정
    output_csv_path: Optional[str] = None
    include_metadata: bool = True
    
    # 처리 설정
    normalize_text: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """초기화 후 설정 검증"""
        if not self.ground_truth_path:
            raise ValueError("ground_truth_path is required")
        if not self.predictions_path:
            raise ValueError("predictions_path is required")