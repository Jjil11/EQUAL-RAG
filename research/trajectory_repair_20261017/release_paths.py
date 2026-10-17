"""Configurable owner-supplied assets; no credentials, automatic downloads or training."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('EQUAL_RAG_ROOT', str(Path(__file__).resolve().parents[2] / 'workspace'))).resolve()
MODELS_ROOT = Path(os.environ.get('EQUAL_RAG_MODELS', str(ROOT / 'models'))).resolve()
QWEN_MODEL = Path(os.environ.get('EQUAL_RAG_QWEN_MODEL', str(MODELS_ROOT / 'Qwen3-14B'))).resolve()
MISTRAL_MODEL = Path(os.environ.get('EQUAL_RAG_MISTRAL_MODEL', str(MODELS_ROOT / 'Mistral-7B-Instruct-v0.3'))).resolve()
CACHE_ROOT = Path(os.environ.get('EQUAL_RAG_CACHE_ROOT', str(ROOT / 'cache'))).resolve()
