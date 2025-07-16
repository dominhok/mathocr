#!/usr/bin/env python3
"""
MathOCR BLEU Evaluation System

Simple command-line interface for evaluating mathematical OCR models using BLEU scores.
"""

import argparse
import sys
import time
from pathlib import Path

from evaluation_system.core.data_loader import DataLoader
from evaluation_system.core.matcher import DataMatcher
from evaluation_system.core.evaluator import BLEUEvaluator
from evaluation_system.utils.validators import validate_csv_format, validate_json_format


def main():
    """평가 시스템의 메인 진입점"""
    parser = argparse.ArgumentParser(
        description="Evaluate mathematical OCR models using BLEU scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --gt data/fermat_meta_cleaned.csv --pred data/state_llama_ocr_cleaned.json
  python main.py --gt data/fermat_meta_cleaned.csv --pred data/gpt4_vision_results.json --output results.csv
        """
    )
    
    parser.add_argument(
        '--gt', '--ground-truth',
        required=True,
        help='Path to ground truth CSV file'
    )
    
    parser.add_argument(
        '--pred', '--predictions',
        required=True,
        help='Path to model predictions JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Path to save detailed results CSV (optional)'
    )
    
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Exclude metadata columns from output CSV'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # 입력 파일 존재 여부 확인
    if not Path(args.gt).exists():
        print(f"Error: Ground truth file not found: {args.gt}")
        sys.exit(1)
    
    if not Path(args.pred).exists():
        print(f"Error: Predictions file not found: {args.pred}")
        sys.exit(1)


    try:
        start_time = time.time()
        
        if not args.quiet:
            print("MathOCR BLEU Evaluation System")
            print("=" * 40)
        
        # 컴포넌트 초기화
        loader = DataLoader()
        matcher = DataMatcher()
        evaluator = BLEUEvaluator()
        
        # 데이터 로드
        if not args.quiet:
            print("\n1. Loading data...")
        
        gt_df = loader.load_ground_truth(args.gt)
        pred_data = loader.load_predictions(args.pred)
        
        # 데이터 형식 검증
        validate_csv_format(gt_df)
        validate_json_format(pred_data)
        
        # 데이터 매칭
        if not args.quiet:
            print("\n2. Matching data...")
        
        matched_pairs = matcher.match_data(gt_df, pred_data)
        
        if len(matched_pairs) == 0:
            print("Error: No matching pairs found between ground truth and predictions")
            sys.exit(1)
        
        # BLEU 점수 평가
        if not args.quiet:
            print("\n3. Evaluating BLEU scores...")
        
        results = evaluator.evaluate_pairs(matched_pairs)
        
        # 결과 출력
        if not args.quiet:
            print("\n4. Results Summary")
            print("-" * 20)
            print(f"Average BLEU Score: {results['average_bleu']:.4f}")
            print(f"Total Evaluated Pairs: {results['total_pairs']}")
            
            # 요약 통계 출력
            stats = evaluator.get_summary_statistics()
            if stats:
                print(f"Min BLEU Score: {stats['min']:.4f}")
                print(f"Max BLEU Score: {stats['max']:.4f}")
                print(f"Standard Deviation: {stats['std']:.4f}")
        else:
            # 조용한 모드 - 평균 점수만 출력
            print(f"{results['average_bleu']:.4f}")
        
        # 요청시 상세 결과 내보내기
        if args.output:
            if not args.quiet:
                print(f"\n5. Exporting results to {args.output}...")
            
            include_metadata = not args.no_metadata
            evaluator.export_results_csv(
                args.output, 
                include_metadata=include_metadata,
                gt_df=gt_df if include_metadata else None
            )
        
        # 실행 시간 출력
        elapsed_time = time.time() - start_time
        if not args.quiet:
            print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()