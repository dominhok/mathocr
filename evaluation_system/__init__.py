"""
MathOCR BLEU Evaluation System

A simple and transparent system for evaluating mathematical OCR models using BLEU scores.
"""

__version__ = "1.0.0"
__author__ = "MathOCR Project"

from .core.evaluator import BLEUEvaluator
from .core.data_loader import DataLoader
from .core.matcher import DataMatcher

__all__ = [
    "BLEUEvaluator",
    "DataLoader", 
    "DataMatcher"
]