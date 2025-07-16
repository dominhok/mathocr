"""데이터 검증 유틸리티"""

import pandas as pd
from typing import Dict, Any, List


def validate_csv_format(df: pd.DataFrame) -> bool:
    """
    Validate that CSV DataFrame has required format.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid format
        
    Raises:
        ValueError: If format is invalid
    """
    required_columns = ['new_custom_id', 'pert_a_cleaned']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    
    # null 이미지 ID 확인
    null_ids = df['new_custom_id'].isnull().sum()
    if null_ids > 0:
        raise ValueError(f"Found {null_ids} null image IDs")
    
    return True


def validate_json_format(data: Dict[str, Any]) -> bool:
    """
    Validate that JSON data has required format.
    
    Args:
        data: JSON data to validate
        
    Returns:
        True if valid format
        
    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("JSON data must be a dictionary")
    
    if 'images' not in data:
        raise ValueError("JSON must contain 'images' key")
    
    # 예측 항목 개수 카운트 (특수 키 제외)
    special_keys = {'images', 'not_parsed'}
    prediction_entries = [k for k in data.keys() if k not in special_keys]
    if len(prediction_entries) == 0:
        raise ValueError("JSON contains no prediction entries")
    
    # 몇 개 항목이 필수 구조를 가지고 있는지 검증
    sample_entries = prediction_entries[:5]  # Check first 5 entries
    for entry_key in sample_entries:
        entry = data[entry_key]
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {entry_key} must be a dictionary")
        
        if 'ocr' not in entry:
            raise ValueError(f"Entry {entry_key} missing 'ocr' key")
        
        if not isinstance(entry['ocr'], dict):
            raise ValueError(f"Entry {entry_key} 'ocr' must be a dictionary")
        
        if 'output' not in entry['ocr']:
            raise ValueError(f"Entry {entry_key} missing 'ocr.output' key")
    
    return True