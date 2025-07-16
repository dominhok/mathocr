"""evaluate 라이브러리를 사용한 BLEU 점수 평가"""

import evaluate
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from ..utils.preprocessors import normalize_text


class BLEUEvaluator:
    """Hugging Face evaluate 라이브러리를 사용한 메인 BLEU 평가 오케스트레이터"""
    
    def __init__(self):
        self.bleu_metric = evaluate.load("bleu")
        self.results: List[Dict[str, Any]] = []
    
    def calculate_bleu_score(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score for a single prediction-reference pair.
        
        Args:
            prediction: Model prediction text
            reference: Ground truth reference text
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        # 텍스트 정규화
        pred_normalized = normalize_text(prediction)
        ref_normalized = normalize_text(reference)
        
        # 빈 텍스트 처리
        if not pred_normalized and not ref_normalized:
            return 1.0  # Both empty = perfect match
        elif not pred_normalized or not ref_normalized:
            return 0.0  # One empty = no match
        
        try:
            # evaluate 라이브러리는 리스트를 기대
            result = self.bleu_metric.compute(
                predictions=[pred_normalized],
                references=[[ref_normalized]]
            )
            return result['bleu'] if result else 0.0
        except Exception as e:
            print(f"Warning: BLEU calculation failed for texts. Error: {e}")
            return 0.0
    
    def evaluate_pairs(self, matched_pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """
        Evaluate BLEU scores for all matched pairs.
        
        Args:
            matched_pairs: List of (image_id, ground_truth, prediction) tuples
            
        Returns:
            Dictionary with evaluation results
        """
        self.results = []
        total_bleu = 0.0
        valid_scores = 0
        
        print(f"Evaluating BLEU scores for {len(matched_pairs)} pairs...")
        
        for image_id, ground_truth, prediction in matched_pairs:
            bleu_score = self.calculate_bleu_score(prediction, ground_truth)
            
            result = {
                'image_id': image_id,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'bleu_score': bleu_score
            }
            
            self.results.append(result)
            total_bleu += bleu_score
            valid_scores += 1
        
        # 평균 BLEU 점수 계산
        avg_bleu = total_bleu / valid_scores if valid_scores > 0 else 0.0
        
        evaluation_summary = {
            'average_bleu': avg_bleu,
            'total_pairs': len(matched_pairs),
            'valid_scores': valid_scores,
            'results': self.results
        }
        
        print(f"Evaluation complete:")
        print(f"  Average BLEU score: {avg_bleu:.4f}")
        print(f"  Total pairs evaluated: {len(matched_pairs)}")
        
        return evaluation_summary
    
    def get_results(self) -> List[Dict[str, Any]]:
        """모든 평가된 쌍에 대한 상세 결과 얻기"""
        return self.results
    
    def export_results_csv(self, output_path: str, include_metadata: bool = False, 
                          gt_df: Optional[pd.DataFrame] = None) -> None:
        """
        Export results to CSV file.
        
        Args:
            output_path: Path to save CSV file
            include_metadata: Whether to include metadata from ground truth
            gt_df: Ground truth DataFrame for metadata
        """
        if not self.results:
            raise ValueError("No results to export. Run evaluation first.")
        
        # 결과로부터 DataFrame 생성
        df = pd.DataFrame(self.results)
        
        # 요청하고 사용 가능한 경우 메타데이터 추가
        if include_metadata and gt_df is not None:
            # 메타데이터 룩업 생성
            metadata_cols = ['grade', 'domain_code', 'subdomain_code']
            available_cols = [col for col in metadata_cols if col in gt_df.columns]
            
            if available_cols:
                gt_metadata = gt_df[['new_custom_id'] + available_cols].set_index('new_custom_id')
                
                # 메타데이터 병합
                df = df.set_index('image_id').join(gt_metadata, how='left').reset_index()
                df.rename(columns={'index': 'image_id'}, inplace=True)
        
        # CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """BLEU 점수의 요약 통계 얻기"""
        if not self.results:
            return {}
        
        scores = [r['bleu_score'] for r in self.results]
        
        return {
            'count': len(scores),
            'mean': sum(scores) / len(scores),
            'min': min(scores),
            'max': max(scores),
            'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
        }