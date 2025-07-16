# MathOCR BLEU 평가 시스템

수학 OCR 모델의 성능을 BLEU 점수로 평가하는 Python 시스템입니다.

## 개요

이 시스템은 수학 문제 이미지에 대한 OCR 모델의 예측 결과를 정답 데이터와 비교하여 BLEU 점수를 계산합니다. 모듈러 구조로 설계되어 다양한 OCR 모델과 데이터 형식에 재사용 가능합니다.

## 주요 기능

- **데이터 매칭**: CSV 정답 데이터와 JSON 예측 데이터 간 이미지 ID 기반 매칭
- **BLEU 평가**: Hugging Face evaluate 라이브러리를 사용한 정확한 BLEU 점수 계산
- **상세 결과 출력**: 개별 이미지별 점수와 메타데이터를 포함한 CSV 결과 파일
- **텍스트 전처리**: 일관된 평가를 위한 텍스트 정규화

## 설치 및 설정

### 요구사항
- Python 3.12 이상
- uv 패키지 매니저

### 설치
```bash
# 의존성 설치
uv sync

# 가상환경 활성화
source venv/bin/activate
```

## 사용법

### 기본 평가
```bash
python main.py --gt data/fermat_meta_cleaned.csv --pred data/state_llama_ocr.json --output results.csv
```

### 매개변수 설명
- `--gt`: 정답 데이터 CSV 파일 경로
- `--pred`: 모델 예측 JSON 파일 경로  
- `--output`: 결과 CSV 파일 저장 경로

## 데이터 형식

### 정답 데이터 (CSV)
- **필수 컬럼**: `new_custom_id` (이미지 ID), `orig_q` (문제), `pert_a_cleaned` (정답)
- **선택 컬럼**: `grade`, `domain_code`, `subdomain_code` (메타데이터)

### 예측 데이터 (JSON)
```json
{
  "images": ["./benchmark_images/img_XXX_pert_Y.Y.png", ...],
  "./benchmark_images/img_XXX_pert_Y.Y.png": {
    "ocr": {
      "question": "추출된 문제 텍스트",
      "answer": "추출된 답안 텍스트", 
      "output": "전체 OCR 출력 텍스트"
    }
  }
}
```

## 결과 출력

### 콘솔 출력
- 데이터 로딩 상태
- 매칭 통계 (총 개수, 매칭 성공/실패)
- 평균 BLEU 점수

### CSV 결과 파일
각 행은 다음 정보를 포함합니다:
- `image_id`: 이미지 식별자
- `ground_truth`: 정답 텍스트
- `prediction`: 예측 텍스트
- `bleu_score`: BLEU 점수 (0.0~1.0)
- 메타데이터 컬럼들 (있는 경우)

## 프로젝트 구조

```
mathocr/
├── evaluation_system/          # 메인 평가 패키지
│   ├── core/                   # 핵심 모듈
│   │   ├── data_loader.py      # 데이터 로딩
│   │   ├── matcher.py          # ID 매칭
│   │   ├── evaluator.py        # BLEU 계산
│   │   └── reporter.py         # 결과 보고
│   ├── utils/                  # 유틸리티
│   │   ├── validators.py       # 데이터 검증
│   │   └── preprocessors.py    # 텍스트 전처리
│   └── config/
│       └── settings.py         # 설정 관리
├── main.py                     # 메인 실행 스크립트
├── pyproject.toml              # 프로젝트 설정
└── README.md                   # 프로젝트 문서
```

## 예제

### Python 스크립트에서 사용
```python
from evaluation_system.core.data_loader import DataLoader
from evaluation_system.core.matcher import DataMatcher
from evaluation_system.core.evaluator import BLEUEvaluator

# 데이터 로딩
loader = DataLoader()
gt_df = loader.load_ground_truth("data/fermat_meta_cleaned.csv")
pred_data = loader.load_predictions("data/state_llama_ocr.json")

# 데이터 매칭
matcher = DataMatcher()
matched_pairs = matcher.match_data(gt_df, pred_data)

# BLEU 평가
evaluator = BLEUEvaluator()
results = evaluator.evaluate_pairs(matched_pairs)

# 결과 출력
print(f"평균 BLEU 점수: {results['average_bleu']:.4f}")
evaluator.export_results_csv("results.csv", include_metadata=True, gt_df=gt_df)
```

## 기술적 세부사항

### 텍스트 전처리
- 공백 정규화
- 특수문자 처리
- 대소문자 통일

### 매칭 로직
- CSV의 `new_custom_id`와 JSON 경로의 파일명 매칭
- 예: `img_66_pert_5.3` ↔ `./benchmark_images/img_66_pert_5.3.png`

### BLEU 계산
- Hugging Face evaluate 라이브러리 사용
- 단어 레벨 토큰화
- 표준 BLEU-4 점수

## 라이선스

MIT License