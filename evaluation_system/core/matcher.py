"""Ground Truth와 예측 데이터 간 이미지 ID 매칭 로직"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd


class DataMatcher:
    """Ground Truth CSV와 예측 JSON 데이터 간 매칭 처리"""
    
    def __init__(self):
        self.matched_pairs: List[Tuple[str, str, str]] = []  # (image_id, gt_text, pred_text)
        self.unmatched_gt: Set[str] = set()
        self.unmatched_pred: Set[str] = set()
    
    def extract_image_id_from_path(self, image_path: str) -> str:
        """
        Extract image ID from full image path.
        
        Args:
            image_path: Full path like "./benchmark_images/img_66_pert_5.3.png"
            
        Returns:
            Image ID like "img_66_pert_5.3"
        """
        # 확장자를 제외한 파일명 추출
        filename = Path(image_path).stem
        return filename
    
    def match_data(self, gt_df: pd.DataFrame, pred_data: Dict) -> List[Tuple[str, str, str]]:
        """
        Match ground truth and prediction data by image IDs.
        
        Args:
            gt_df: DataFrame with ground truth data
            pred_data: Dictionary with prediction data
            
        Returns:
            List of tuples (image_id, ground_truth_text, prediction_text)
        """
        self.matched_pairs = []
        self.unmatched_gt = set()
        self.unmatched_pred = set()
        
        # 룩업 사전 생성
        gt_lookup = {}
        for _, row in gt_df.iterrows():
            image_id = row['new_custom_id']
            
            # 공정한 비교를 위해 질문과 답안 결합
            orig_q_val = row['orig_q']
            pert_a_cleaned_val = row['pert_a_cleaned']
            orig_q = str(orig_q_val) if pd.notna(orig_q_val) else ""
            pert_a_cleaned = str(pert_a_cleaned_val) if pd.notna(pert_a_cleaned_val) else ""
            
            # 줄바꿈 구분자로 질문과 답안 결합
            if orig_q and pert_a_cleaned:
                gt_text = f"{orig_q}\n\n{pert_a_cleaned}"
            elif orig_q:
                gt_text = orig_q
            elif pert_a_cleaned:
                gt_text = pert_a_cleaned
            else:
                gt_text = ""
                
            gt_lookup[image_id] = gt_text
        
        pred_lookup = {}
        for path, data in pred_data.items():
            if path in ['images', 'not_parsed']:  # 특수 키 건너뛰기
                continue
            
            image_id = self.extract_image_id_from_path(path)
            
            # OCR 출력에서 예측 텍스트 추출 (전체 텍스트)
            if 'ocr' in data and 'output' in data['ocr']:
                pred_text = str(data['ocr']['output']) if data['ocr']['output'] is not None else ""
            else:
                pred_text = ""
            
            pred_lookup[image_id] = pred_text
        
        # Ground Truth와 예측 데이터 매칭
        gt_ids = set(gt_lookup.keys())
        pred_ids = set(pred_lookup.keys())
        
        # 매칭 찾기
        matched_ids = gt_ids.intersection(pred_ids)
        for image_id in matched_ids:
            gt_text = gt_lookup[image_id]
            pred_text = pred_lookup[image_id]
            self.matched_pairs.append((image_id, gt_text, pred_text))
        
        # 매칭되지 않은 항목 추적
        self.unmatched_gt = gt_ids - pred_ids
        self.unmatched_pred = pred_ids - gt_ids
        
        # 매칭 통계 출력
        print(f"Matching results:")
        print(f"  Total ground truth entries: {len(gt_ids)}")
        print(f"  Total prediction entries: {len(pred_ids)}")
        print(f"  Successfully matched: {len(matched_ids)}")
        print(f"  Unmatched ground truth: {len(self.unmatched_gt)}")
        print(f"  Unmatched predictions: {len(self.unmatched_pred)}")
        
        if self.unmatched_gt:
            print(f"  Sample unmatched GT IDs: {list(self.unmatched_gt)[:5]}")
        if self.unmatched_pred:
            print(f"  Sample unmatched pred IDs: {list(self.unmatched_pred)[:5]}")
        
        return self.matched_pairs
    
    def get_matched_pairs(self) -> List[Tuple[str, str, str]]:
        """매칭된 (image_id, ground_truth, prediction) 쌍 얻기"""
        return self.matched_pairs
    
    def get_match_statistics(self) -> Dict[str, int]:
        """매칭 과정에 대한 통계 얻기"""
        return {
            'total_matches': len(self.matched_pairs),
            'unmatched_ground_truth': len(self.unmatched_gt),
            'unmatched_predictions': len(self.unmatched_pred)
        }