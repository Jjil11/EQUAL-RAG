"""Portable paths for the exported development snapshot (no credentials)."""
import os
from pathlib import Path

ROOT=Path(os.environ.get('EQUAL_RAG_ROOT',str(Path(__file__).resolve().parents[1]/'workspace'))).resolve()
MODELS_ROOT=Path(os.environ.get('EQUAL_RAG_MODELS',str(ROOT/'models'))).resolve()
QWEN_MODEL=Path(os.environ.get('EQUAL_RAG_QWEN_MODEL',str(MODELS_ROOT/'Qwen3-14B'))).resolve()
CACHE_ROOT=Path(os.environ.get('EQUAL_RAG_CACHE_ROOT',str(ROOT/'cache'))).resolve()
