# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MathOCR is a Python project implementing a BLEU Score Evaluation System for mathematical OCR models. The system evaluates OCR model predictions against ground truth annotations using BLEU scores, designed to be modular and reusable across different models.

## Development Environment

- **Python Version**: >=3.12 (specified in pyproject.toml)
- **Package Manager**: Uses `uv` for dependency management (uv.lock present)
- **Virtual Environment**: Standard Python venv in `venv/` directory

## Dependencies

Minimal dependencies for simple implementation:
- `evaluate>=0.4.5` - Hugging Face BLEU score calculation (includes NLTK internally)
- `pandas>=2.3.1` - Data manipulation and CSV processing

**Note**: NLTK is automatically installed as a dependency of `evaluate` library and handles tokenization internally. No need to explicitly use NLTK in our code.

## Data Structure

### Ground Truth Data (`data/fermat_meta_cleaned.csv`)
- **Key Column**: `new_custom_id` - Image identifier (e.g., "img_66_pert_5.3")
- **Answer Column**: `pert_a_cleaned` - Ground truth text
- **Metadata**: grade, domain_code, subdomain_code, image quality flags

### Model Predictions (`data/state_llama_ocr.json`)
```json
{
  "images": ["./benchmark_images/img_XXX_pert_Y.Y.png", ...],
  "./benchmark_images/img_XXX_pert_Y.Y.png": {
    "ocr": {
      "question": "extracted question text",
      "answer": "extracted answer text", 
      "output": "complete OCR output text"
    }
  }
}
```

### Data Matching Logic
- CSV `new_custom_id`: "img_66_pert_5.3" 
- JSON key: "./benchmark_images/img_66_pert_5.3.png"
- Matching: Extract filename from JSON path and match with CSV ID + ".png"

## Common Commands

```bash
# Install dependencies
uv sync

# Activate virtual environment  
source venv/bin/activate

# Run BLEU evaluation (when implemented)
python main.py --gt data/fermat_meta_cleaned.csv --pred data/state_llama_ocr.json

# Run with different model predictions
python main.py --gt data/fermat_meta_cleaned.csv --pred data/gpt4_vision_results.json

# Export detailed results
python main.py --gt data/fermat_meta_cleaned.csv --pred data/state_llama_ocr.json --output detailed_results.csv
```

## Project Architecture

### Current Project Structure
```
mathocr/
├── data/
│   ├── fermat_meta_cleaned.csv     # Ground truth data
│   └── state_llama_ocr.json        # Model predictions
├── evaluation_system/              # Main evaluation package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading and preprocessing
│   │   ├── matcher.py              # Image ID matching logic
│   │   ├── evaluator.py            # BLEU score calculation
│   │   └── reporter.py             # Results analysis and reporting
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validators.py           # Data validation
│   │   ├── preprocessors.py        # Text preprocessing
│   │   └── visualizers.py          # Visualization (optional)
│   └── config/
│       ├── __init__.py
│       └── settings.py             # Configuration management
├── main.py                         # Main execution script
├── pyproject.toml                  # Project dependencies
├── uv.lock                         # Locked dependencies
├── venv/                           # Virtual environment
├── CLAUDE.md                       # Project instructions
└── README.md                       # Project documentation
```

### Core Classes
1. **BLEUEvaluator** - Main evaluation orchestrator
2. **DataMatcher** - Handles image ID matching between GT and predictions
3. **TextPreprocessor** - Text normalization and cleaning

## Key Features

1. **Data Loading & Preprocessing**: CSV/JSON parsing with validation
2. **Image ID Matching**: Robust matching between GT and prediction data
3. **BLEU Score Calculation**: Using Hugging Face evaluate library (simple one-liner)
4. **Detailed Results Export**: CSV with individual scores and metadata
5. **Error Handling**: Comprehensive validation and logging

## Usage Patterns

### Basic Evaluation
```python
evaluator = BLEUEvaluator()
evaluator.load_ground_truth("fermat_meta_cleaned.csv")
evaluator.load_predictions("state_llama_ocr.json")
avg_bleu = evaluator.evaluate()
evaluator.export_detailed_results("results.csv")
```

### Expected Output Format
- **Console**: Average BLEU score, match statistics, timing
- **CSV**: Detailed results with columns: image_id, ground_truth, prediction, bleu_score, grade, domain_code, subdomain_code

## Implementation Priority

**Phase 1 (Current Focus)**:
1. Basic data loading and matching
2. BLEU score calculation  
3. Detailed results CSV generation

**Phase 2 (Future)**:
1. Support for different prediction formats
2. Additional evaluation metrics (ROUGE)
3. Domain/grade-specific filtering