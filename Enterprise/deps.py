
# ── Standard Library ────────────────────────────────────────
import os, re, ast, sys, json, time, copy, queue, random
import logging, traceback, subprocess, threading
import io, base64, hashlib, uuid, unicodedata
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# ── Numeric / ML ──────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Web / Parsing ───────────────────────────────────────
import requests
from bs4 import BeautifulSoup

# ── HuggingFace ───────────────────────────────────────
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModel,
    TrainingArguments, Trainer,
    TrainerCallback, DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# ── Sentence Embeddings (Real) ───────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

# ── Vector Database (Real) ───────────────────────────────────
try:
    import chromadb
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

# ── Knowledge Graph ──────────────────────────────────────
try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False

# ── Browser Automation ───────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
    _PW_OK = True
except ImportError:
    _PW_OK = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    _SEL_OK = True
except ImportError:
    _SEL_OK = False

# ── GUI Control ───────────────────────────────────────

try:
    import pyautogui
    from PIL import Image, ImageGrab
    _GUI_OK = True
except ImportError:
    _GUI_OK = False

# ── Speech ───────────────────────────────────────
try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False

try:
    import pyttsx3
    _TTS_OK = True
except ImportError:
    _TTS_OK = False

# ── Vision ──────────────────────────────────────
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── PDF Text Extraction ──────────────────────────────────────
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ── Language Detection (used by DatasetQualityControl) ───────
try:
    from langdetect import detect as _langdetect_detect
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

# ── Accelerate (multi-GPU / DeepSpeed) ───────────────────────
try:
    from accelerate import Accelerator
    _ACCEL_OK = True
except ImportError:
    _ACCEL_OK = False

# ── TRL (PPO / DPO) ──────────────────────────────────────
try:
    from trl import PPOTrainer as TRLPPOTrainer, PPOConfig
    from trl import DPOTrainer as TRLDPOTrainer
    _TRL_OK = True
except ImportError:
    _TRL_OK = False

# ── Explicit __all__ ──────────────────────────────────────────
# IMPORTANT: every other module in this package does `from .deps import *`.
# Python's wildcard import silently DROPS underscore-prefixed names unless
# the source module defines __all__. Without this, every `_XXX_OK` flag
# above (_ST_OK, _NX_OK, _PIL_OK, etc.) is invisible to every file that
# imports this module with `*`, and the first `if _XXX_OK:` check anywhere
# else in the package raises NameError — even when the library in
# question is actually installed. This line fixes that for the whole
# package in one place.
__all__ = [name for name in dir() if not name.startswith("__")]
