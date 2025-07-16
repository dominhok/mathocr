"""CSV와 JSON 파일 로딩 기능"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


class DataLoader:
    """Ground Truth CSV와 예측 JSON 파일 로딩 처리"""
    
    def __init__(self):
        self.ground_truth_data: Optional[pd.DataFrame] = None
        self.prediction_data: Optional[Dict[str, Any]] = None
    
    def load_ground_truth(self, csv_path: str) -> pd.DataFrame:
        """
        Load ground truth data from CSV file.
        
        Args:
            csv_path: Path to the CSV file containing ground truth data
            
        Returns:
            DataFrame with ground truth data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # 필수 컨럼 검증
        required_columns = ['new_custom_id', 'orig_q', 'pert_a_cleaned']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.ground_truth_data = df
        print(f"Loaded {len(df)} ground truth records from {csv_path}")
        return df
    
    def load_predictions(self, json_path: str) -> Dict[str, Any]:
        """
        Load model predictions from JSON file.
        
        Args:
            json_path: Path to the JSON file containing model predictions
            
        Returns:
            Dictionary with prediction data
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON format is invalid
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read JSON file: {e}")
        
        # JSON 구조 검증
        if 'images' not in data:
            raise ValueError("JSON file must contain 'images' key")
        
        # 예측 데이터 개수 카운트 (특수 키 제외)
        special_keys = {'images', 'not_parsed'}
        prediction_count = len([k for k in data.keys() if k not in special_keys])
        
        self.prediction_data = data
        print(f"Loaded predictions for {prediction_count} images from {json_path}")
        return data
    
    def get_ground_truth_data(self) -> pd.DataFrame:
        """로드된 Ground Truth 데이터 얻기"""
        if self.ground_truth_data is None:
            raise ValueError("Ground truth data not loaded. Call load_ground_truth() first.")
        return self.ground_truth_data
    
    def get_prediction_data(self) -> Dict[str, Any]:
        """로드된 예측 데이터 얻기"""
        if self.prediction_data is None:
            raise ValueError("Prediction data not loaded. Call load_predictions() first.")
        return self.prediction_data