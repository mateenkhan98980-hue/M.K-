# ðŸš€ M.K SETUP: Enterprise AI Full System Map

> **A comprehensive, enterprise-grade AI architecture spanning 20 files, 47 classes, and 8 architectural layers.**

![Architecture Layers](https://img.shields.io/badge/Architecture_Layers-8-blue)
![Total Files](https://img.shields.io/badge/Total_Files-20-success)
![Total Classes](https://img.shields.io/badge/Total_Classes-47-orange)
![Lines of Code](https://img.shields.io/badge/Size-~143K-purple)

Welcome to the **M.K SETUP Enterprise AI** repository. This project is a massive, highly structured AI framework designed to handle everything from secure tool execution and autonomous multi-agent orchestration to full Language Model pre-training, Reinforcement Learning from Human Feedback (RLHF), and Chain-of-Thought (CoT) synthesis.

---

## ðŸ“‘ Table of Contents

1. [System Overview](#-system-overview)
2. [Architectural Layers](#-architectural-layers)
   - [Layer 0: Package Entry Points](#layer-0-package-entry-points)
   - [Layer 1: Core Infrastructure](#layer-1-core-infrastructure)
   - [Layer 2: Perception & I/O](#layer-2-perception--io)
   - [Layer 3: Execution, Tools & Agent Orchestration](#layer-3-execution-tools--agent-orchestration)
   - [Layer 4: Data Quality Control](#layer-4-data-quality-control)
   - [Layer 5: Training Pipelines](#layer-5-training-pipelines)
   - [Layer 6: RLHF Pipeline](#layer-6-rlhf-pipeline)
3. [End-to-End Data Flows](#-end-to-end-data-flows)
4. [Feature Flags & Dependencies](#-feature-flags--dependencies)
5. [Getting Started & Smoke Tests](#-getting-started)

---

## ðŸ”­ System Overview

The M.K SETUP framework provides an end-to-end sandbox and production environment. It seamlessly integrates:
- **Agent Orchestration:** Thread-based multi-agent routing.
- **Knowledge Representation:** Vector Databases (Chroma/FAISS) and Directed Knowledge Graphs (NetworkX).
- **Perception/IO:** Computer Vision (BLIP-2), Audio (STT/TTS), and Web/GUI automation.
- **Model Training:** Full pre-training, LoRA fine-tuning, Sparse Mixture-of-Experts (MoE), and RL/RLHF.

---

## ðŸ— Architectural Layers

### Layer 0: Package Entry Points
The entry point of the application guarantees a clean global import surface and dynamic dependency resolution.
* **`__init__.py`**: Aggregates and re-exports public classes from all 20 modules. Provides a single wildcard import (`from enterprise_ai import *`).
* **`__main__.py`**: Houses the **Smoke Test Runner**, an 11-step test suite that covers all major components and prints a comprehensive component status table.
* **`deps.py`**: The **Dependency Hub**. Tries to load optional libraries and sets boolean feature flags (e.g., `_ST_OK`, `_CHROMA_OK`) to enable graceful degradation if modules are missing.

### Layer 1: Core Infrastructure
* **`logger.py`**: Dual-output structured logging (timestamped file + console), injected via constructors across the entire package.
* **`embeddings.py`**: Real dense text Autonomous desktop control. Loops through screenshots -> BLIP-2 descriptions -> LLM planning -> GUI execution.
* **`agents.py`**: A multi-agent framework built on a Thread-safe `MessageBus`. The orchestrator routes tasks dynamically: 
  * `http/url/web` -> **ResearchAgent**
  * `code/python` -> **CodingAgent**
  * `image/screenshot` -> **VisionAgentWorker**
  * *Other* -> **ReasoningAgent**

### Layer 4: Data Quality Control (`quality.py`)
A rigorous 6-stage data filtering pipeline used for corpus preparation:
1. **Normalize**: NFKC unicode normalization, whitespace collapsing.
2. **Structural**: Checks alpha ratio (â‰¥60%), repetition limits, and boilerplate phrase constraints.
3. **Language**: `langdetect` enforcing English (with heuristic fallbacks).
4. **Coherence & Topic**: Stopword ratio, avg word length, and optional regex keyword whitelisting.
5. **Safety + PII**: Strips unsafe patterns (weapons/drugs) and redacts PII (email/SSN/cards).
6. **Dedup**: SHA-256 exact hash + SimHash 64-bit fingerprinting + LSH 4-band routing.

### Layer 5: Training Pipelines
* **`pretraining.py`**: Handles **Full Pre-Training** (all parameters). Features block-packing of tokens into 2048-length chunks, bfloat16, and cosine LR scheduling. Includes `PDFCorpusBuilder` and `WebCorpusBuilder`.
* **`training.py`**: Focuses on **LoRA Fine-Tuning** (rank=8, targeting q/v projections) and injects **Sparse Mixture-of-Experts** (MoE) into FFN layers with top-2 load-balance routing.
* **`rl.py`**: REINFORCE-style policy gradient learning and a `SelfImprovementLoop` (Generate -> Evaluate -> Store -> Reflect -> Update).
* **`cot.py`**: The **Chain-of-Thought (CoT)** pipeline. Generates synthetic training data across math, logic, and commonsense domains. Uses `SmartDataQualityControl` to score reasoning quality.

### Layer 6: RLHF Pipeline (`rlhf.py`)
Supports two distinct alignment paradigms:
* **PPO PATH (Classic InstructGPT-style RLHF - 3 Stages):** 
  1. Supervised Fine-Tuning (SFTTrainer)
  2. Comparison Reward Model (Bradley-Terry online generation)
  3. Proximal Policy Optimization (PPOTrainer) with KL penalties.
* **DPO PATH (Direct Preference Optimization - 2 Stages):**
  1. Supervised Fine-Tuning (SFTTrainer)
  2. Direct Preference Optimization (DPOTrainer) utilizing log probabilities directly, eliminating the need for a separate reward model.

---

## ðŸ”„ End-to-End Data Flows

### 1. Pre-Training Pipeline
Raw Sources (PDFs/Web) -> Corpus Builders (`extract_pdf` / `scrape_urls`) -> Dataset Quality Control (6-stage filter) -> `save_chunks_jsonl()` -> Block-pack 2048-token chunks -> `FullPreTrainer.pretrain()`

### 2. Multi-Agent Routing
`MultiAgentOrchestrator.route(task)` -> Keyword Routing logic -> `MessageBus.send()` -> Target Agent's Inbox -> `Agent.handle(msg)` -> Output returned via Orchestrator polling.

### 3. Self-Improvement Loop
Input Prompt -> LLM `generate()` -> `RewardModel` scoring -> Stored in VectorDB/Graph -> LLM self-critique (if score < 0.6) -> `RLTrainer` Policy gradient update.

### 4. Computer Use Loop
Goal String -> `GUIController.screenshot()` -> `VisionAgent.caption()` -> LLM `plan()` (Generates `ACTION` and `PARAMS`) -> GUI action -> Loop.

---

## âš™ï¸ Feature Flags & Dependencies

To ensure maximum compatibility, the framework dynamically adjusts based on available libraries via `deps.py`:

| Flag | Library | Purpose |
| :--- | :--- | :--- |
| `_ST_OK` | `sentence-transformers` | Real dense embeddings |
| `_CHROMA_OK` | `chromadb` | Persistent vector DB (disk) |
| `_FAISS_OK` | `faiss` | In-memory vector DB (fast) |
| `_NX_OK` | `networkx` | Knowledge graph (DiGraph) |
| `_PW_OK` | `playwright` | Headless browser automation |
| `_SEL_OK` | `selenium` | Browser fallback |
| `_GUI_OK` | `pyautogui` | Desktop GUI control |
| `_PIL_OK` | `Pillow` | Image processing |
| `_SR_OK` | `speech_recognition` | Microphone text STT |
| `_TTS_OK` | `pyttsx3` | Speech offline |
| `_PDF_OK` | `pdfminer.six` | PDF text extraction |
| `_TRL_OK` | `trl` | PPO/DPO trainers |

---

## ðŸš€ Getting Started

Ensure you have Python 3.9+ installed. You can install all features or let the framework fallback automatically.

```bash
# Run the complete Smoke Test suite (11-step verification)
python -m enterprise_ai
```

### Basic Orchestrator Usage

```python
from enterprise_ai import MultiAgentOrchestrator, logger

orchestrator = MultiAgentOrchestrator()
logger.info("Sending task to agents...")

# Routed to ResearchAgent automatically
orchestrator.route("http://github.com Fetch the repo summary") 

# Routed to CodingAgent automatically
orchestrator.route("code/python Write a quick sort algorithm") 
```
*Build vua M.K SETUP - Enterprise-grade AI Architecture.*
