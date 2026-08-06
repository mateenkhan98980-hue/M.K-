"""Smoke test: python -m enterprise_ai"""
from .deps import *
from .logger import StructuredLogger
from .embeddings import RealEmbeddingModel, RealVectorDatabase
from .knowledge import KnowledgeGraph
from .sandbox import SecureSandbox
from .tools import calc_tool, python_tool, web_search_tool
from .speech import SpeechIO
from .vision import VisionAgent
from .rl import RewardModel
from .quality import DatasetQualityControl
import json
import torch

def main():
    logger = StructuredLogger()

    # ── 1. Real Embeddings ────────────────────────────────────
    logger.info("=== 1. Real Embeddings ===")
    embedder = RealEmbeddingModel()
    emb = embedder.encode(["Hello world", "AI systems"])
    logger.info(f"Embedding shape: {emb.shape}  backend={'sentence-transformers' if _ST_OK else 'fallback'}")

    # ── 2. Real Vector DB ─────────────────────────────────────
    logger.info("=== 2. Vector Database ===")
    vdb = RealVectorDatabase(embedder, persist_dir="./test_chroma")
    vdb.add(["Python is a language", "AI uses neural networks", "LoRA fine-tunes models"])
    results = vdb.query("machine learning", top_k=2)
    logger.info(f"Query results: {[r['text'] for r in results]}")
    logger.info(f"DB size: {len(vdb)}  backend={vdb._backend}")

    # ── 3. Knowledge Graph ────────────────────────────────────
    logger.info("=== 3. Knowledge Graph ===")
    kg = KnowledgeGraph()
    kg.ingest_text("Python is a programming language. Neural networks are machine learning models.")
    kg.add_triple("LoRA", "is", "fine-tuning method")
    logger.info(f"KG stats: {kg.stats()}")

    # ── 4. Sandbox ────────────────────────────────────────────
    logger.info("=== 4. Secure Sandbox ===")
    sandbox = SecureSandbox(timeout=5)
    r = sandbox.run_python("print(2 ** 10)")
    logger.info(f"Sandbox output: {r}")
    r2 = sandbox.run_python("import os; print(os.getcwd())")  # blocked
    logger.info(f"Blocked test: {r2}")

    # ── 5. Safe Calculator ────────────────────────────────────
    logger.info("=== 5. Tools ===")
    logger.info(f"calc_tool('99 * 3 + 7') = {calc_tool('99 * 3 + 7')}")
    logger.info(f"python_tool('print(42)') = {python_tool('print(42)')}")
    logger.info(f"web_search_tool('LoRA fine-tuning') =\n{web_search_tool('LoRA fine-tuning', max_results=2)}")

    # ── 6. Speech (availability check) ───────────────────────
    logger.info("=== 6. Speech IO ===")
    speech = SpeechIO(logger)
    logger.info(f"TTS available: {_TTS_OK} | STT available: {_SR_OK}")
    speech.speak("System initialized successfully.")   # silent if not installed

    # ── 7. Browser (availability check) ──────────────────────
    logger.info("=== 7. Browser ===")
    logger.info(f"Playwright: {_PW_OK} | Selenium: {_SEL_OK}")

    # ── 8. GUI (availability check) ──────────────────────────
    logger.info("=== 8. GUI Control ===")
    logger.info(f"pyautogui: {_GUI_OK}")

    # ── 9. Vision (availability check) ───────────────────────
    logger.info("=== 9. Vision Agent ===")
    vision = VisionAgent(logger)
    logger.info(f"PIL: {_PIL_OK} | BLIP auto-loads on first caption() call")

    # ── 10. RL Trainer (unit test) ────────────────────────────
    logger.info("=== 10. RL Reward Model (unit test) ===")
    reward_model = RewardModel(hidden_dim=embedder.dim)
    p = torch.randn(1, embedder.dim); r = torch.randn(1, embedder.dim)
    score = reward_model(p, r)
    logger.info(f"Reward model output: {score.item():.4f}")

    # ── 11. Dataset Quality Control (unit test) ───────────────
    logger.info("=== 11. Dataset Quality Control ===")
    qc = DatasetQualityControl(logger)
    qc_samples = [
        "Python is a high-level programming language known for its readability and broad ecosystem of libraries.",
        "Python is a high-level programming language known for its readability and broad ecosystem of libraries.",  # exact dup
        "click here click here subscribe subscribe buy now buy now sign up now",                                    # boilerplate
        "asdkj 123 !!! qwop xx zz zzzz",                                                                              # gibberish
        "Contact us at jane.doe@example.com or call 555-123-4567 for support.",                                      # PII
        "hi",                                                                                                         # too short
    ]
    for sample in qc_samples:
        result = qc.validate(sample)
        preview = sample[:45] + ("…" if len(sample) > 45 else "")
        logger.info(f"  [{('KEPT' if result['ok'] else 'REJECTED ' + result['reason']):<20}] {preview}")
    logger.info(f"QC report: {qc.report()}")

    # ── Summary ───────────────────────────────────────
    print("\n" + "="*60)
    print("  COMPONENT STATUS")
    print("="*60)
    components = [
        ("Real Embeddings",      _ST_OK),
        ("Vector DB (chroma)",   _CHROMA_OK),
        ("Vector DB (faiss)",    _FAISS_OK),
        ("Knowledge Graph",      _NX_OK),
        ("Browser (playwright)", _PW_OK),
        ("Browser (selenium)",   _SEL_OK),
        ("GUI Control",          _GUI_OK),
        ("Vision (PIL)",         _PIL_OK),
        ("Speech STT",           _SR_OK),
        ("Speech TTS",           _TTS_OK),
        ("PDF Extraction",       _PDF_OK),
        ("Language Detection",   _LANGDETECT_OK),
        ("Accelerate (multi-GPU)",_ACCEL_OK),
        ("TRL (PPO/DPO)",        _TRL_OK),
        ("Sandbox",              True),
        ("RL Trainer",           True),
        ("Multi-Agent Bus",      True),
        ("Self-Improvement",     True),
        ("Dataset Quality Control", True),
        ("Web Corpus Builder",   True),
        ("Full Pre-Trainer",     True),
        ("RLHF Pipeline",        True),
    ]
    for name, ok in components:
        status = "✅ Ready" if ok else "⚠️  Install required"
        print(f"  {name:<32} {status}")
    print("="*60)


    # ── Full Pre-Training example (uncomment to run) ───────────
    # model_name   = "meta-llama/Llama ALL MODEL"
    # pretrainer   = FullPreTrainer(model_name, logger, context_length=2048)
    #
    # Option A: pretrain from a list of text strings
    # texts = ["Large text chunk 1...", "Large text chunk 2...", ...]
    # pretrainer.pretrain(texts, output_dir="./pretrained", num_epochs=1)
    #
    # Option B: pretrain directly from a folder of PDFs
    # pretrainer.pretrain_from_pdfs("/path/to/pdfs/", output_dir="./pretrained")
    #
    # Option C: pretrain straight from a list of URLs — each page is
    # scraped, then run through DatasetQualityControl (structural,
    # language, coherence, safety, PII, dedup) before training ever sees it
    # pretrainer.pretrain_from_urls(
    #     ["https://example.com/article-1", "https://example.com/article-2"],
    #     output_dir="./pretrained",
    #     lang_whitelist=("en",),
    # )

    # ── RLHF Pipeline example (uncomment to run) ──────────────
    # model_name = "meta-llama/Llama ALL MODEL"
    # rlhf       = RLHFPipeline(model_name, logger)
    #
    # # Supervised data (Stage 1)
    # conversations = [
    #     {"prompt": "What is LoRA?",
    #      "response": "LoRA (Low-Rank Adaptation) is a fine-tuning method…"},
    #     {"prompt": "Explain RLHF",
    #      "response": "RLHF stands for Reinforcement Learning from Human Feedback…"},
    # ]
    #
    # # Preference data (Stage 2 / DPO)
    # prefs = PreferenceDataset()
    # prefs.add(
    #     prompt   = "What is the capital of France?",
    #     chosen   = "The capital of France is Paris.",
    #     rejected = "France has many cities.",
    # )
    #
    # # PPO path (3-stage, classic RLHF)
    # rlhf.full_pipeline_ppo(
    #     conversations=conversations,
    #     preference_data=prefs,
    #     ppo_prompts=["Tell me about neural networks", "What is fine-tuning?"],
    #     base_dir="./rlhf_output_ppo",
    # )
    #
    # # DPO path (2-stage, simpler — recommended for most use cases)
    # rlhf.full_pipeline_dpo(
    #     conversations=conversations,
    #     preference_data=prefs,
    #     base_dir="./rlhf_output_dpo",
    # )

    # ── LoRA fine-tuning + MoE (original, unchanged) ──────────
    # model_name = "meta-llama/Llama-3.2-1B"
    # trainer    = EnterpriseTrainer(model_name, logger)
    # trainer.apply_lora()
    # trainer.inject_moe()
    # inference  = InferencePipeline(trainer.model, trainer.tokenizer)
    # rl         = RLTrainer(trainer.model, trainer.tokenizer, reward_model, embedder, logger)
    # loop       = SelfImprovementLoop(inference, rl, vdb, kg, logger)
    # results    = loop.run(["Explain quantum computing", "What is LoRA?"], iterations=3)
    # print(loop.report())


if __name__ == "__main__":
    main()

