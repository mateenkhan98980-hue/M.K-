from .deps import *
from .logger import StructuredLogger
from .quality import DatasetQualityControl
from .browser import BrowserAgent


def save_chunks_jsonl(chunks: List[str], path: str, source_tag: str = "", keywords: Optional[List[str]] = None) -> str:
    """
    NEW: persist extracted/quality-filtered chunks to a .jsonl file —
    one JSON object per line: {"text": ..., "source": ..., "keywords": ...}.
    This is the 'Extraction: sirf high-value data ko JSON/DB mein save
    karo' step. Works with output from any builder below.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "text": chunk, "source": source_tag, "keywords": keywords,
            }, ensure_ascii=False) + "\n")
    return path

class PDFCorpusBuilder:
    """
    Extracts and cleans text from PDF files for pre-training.
    Supports single PDFs, entire directory trees, and streaming
    for large corpora (no full RAM load required).
    """
    def __init__(self, logger: StructuredLogger, chunk_size: int = 2048):
        self.logger     = logger
        self.chunk_size = chunk_size
        self.qc         = DatasetQualityControl(logger)

    def extract_pdf(self, pdf_path: str) -> str:
        """Extract + normalize raw text from one PDF (quality filtering happens at chunk level)."""
        if _PDF_OK:
            try:
                text = pdf_extract_text(pdf_path)
                return self.qc.normalize(text)
            except Exception as e:
                self.logger.warn(f"pdfminer failed [{pdf_path}]: {e}")
        # Fallback: pdftotext system binary
        try:
            result = subprocess.run(
                ["pdftotext", pdf_path, "-"],
                capture_output=True, text=True, timeout=30,
            )
            return self.qc.normalize(result.stdout)
        except Exception as e:
            self.logger.warn(f"pdftotext fallback failed [{pdf_path}]: {e}")
            return ""

    # ── NEW: stream a PDF straight from a URL — no disk write, ever.
    # Bytes go HTTP response → in-memory buffer → pdfminer → discarded.
    def extract_pdf_from_url(self, url: str) -> str:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
        except Exception as e:
            self.logger.warn(f"PDF download failed [{url}]: {e}")
            return ""
        if _PDF_OK:
            try:
                text = pdf_extract_text(buf)
                return self.qc.normalize(text)
            except Exception as e:
                self.logger.warn(f"pdfminer failed on streamed PDF [{url}]: {e}")
                return ""
        self.logger.warn("pdfminer not installed — can't parse a PDF buffer without it "
                          "(the pdftotext binary fallback needs an actual file on disk).")
        return ""

    # ── NEW: a whole list of PDF URLs, fully in-memory, quality-filtered ─
    def build_from_urls(
        self, pdf_urls: List[str], keywords: Optional[List[str]] = None,
        save_to: Optional[str] = None,
    ) -> List[str]:
        """Same contract as build_from_directory(), but for PDFs that
        live on the web — nothing touches local disk. Pass save_to="out.jsonl"
        to persist the result automatically."""
        chunks = []
        self.logger.info(f"Streaming {len(pdf_urls)} PDFs from URLs (no disk writes)")
        for i, url in enumerate(pdf_urls, 1):
            self.logger.info(f"[{i}/{len(pdf_urls)}] Streaming PDF {url}")
            text = self.extract_pdf_from_url(url)
            if not text:
                continue
            for rough_chunk in self.qc.semantic_chunking(text, self.chunk_size):
                result = self.qc.validate(rough_chunk, keywords=keywords)
                if result["ok"]:
                    chunks.append(result["text"])

        report = self.qc.report()
        self.logger.info(
            f"Streamed-PDF corpus built: {report['accepted']} clean chunks kept "
            f"({report['rejected']} rejected) from {len(pdf_urls)} PDF URLs "
            f"| accept_rate={report['accept_rate']*100:.1f}%"
        )
        self.logger.info(f"Rejection breakdown: {report['breakdown']}")
        if save_to:
            save_chunks_jsonl(chunks, save_to, source_tag="pdf_url", keywords=keywords)
            self.logger.info(f"Saved {len(chunks)} chunks → {save_to}")
        return chunks

    def build_from_directory(
        self, pdf_dir: str, keywords: Optional[List[str]] = None,
        save_to: Optional[str] = None,
    ) -> List[str]:
        """Load ALL PDFs in a directory tree → list of QUALITY-FILTERED text chunks.
        Pass keywords=["algorithm","function","formula"] to keep only
        chunks that actually mention those topics. Pass save_to="out.jsonl"
        to persist the result automatically."""
        chunks    = []
        pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")
        for path in pdf_files:
            text = self.extract_pdf(str(path))
            if not text:
                continue
            for rough_chunk in self.qc.semantic_chunking(text, self.chunk_size):
                result = self.qc.validate(rough_chunk, keywords=keywords)
                if result["ok"]:
                    chunks.append(result["text"])

        report = self.qc.report()
        self.logger.info(
            f"Corpus built: {report['accepted']} clean chunks kept "
            f"({report['rejected']} rejected) from {len(pdf_files)} PDFs "
            f"| accept_rate={report['accept_rate']*100:.1f}%"
        )
        self.logger.info(f"Rejection breakdown: {report['breakdown']}")
        if save_to:
            save_chunks_jsonl(chunks, save_to, source_tag="pdf_dir", keywords=keywords)
            self.logger.info(f"Saved {len(chunks)} chunks → {save_to}")
        return chunks

    def stream_chunks(self, pdf_dir: str, keywords: Optional[List[str]] = None):
        """
        Generator — yields only QUALITY-FILTERED chunks, one at a time.
        Use this for very large corpora to avoid RAM exhaustion.
        """
        for path in Path(pdf_dir).rglob("*.pdf"):
            text = self.extract_pdf(str(path))
            if not text:
                continue
            for rough_chunk in self.qc.semantic_chunking(text, self.chunk_size):
                result = self.qc.validate(rough_chunk, keywords=keywords)
                if result["ok"]:
                    yield result["text"]


class WebCorpusBuilder:
    """
    Scrapes a list of URLs via BrowserAgent and runs everything through
    the SAME DatasetQualityControl pipeline as PDFCorpusBuilder, so
    web-sourced and PDF-sourced training data are held to one identical
    quality + safety bar before a model ever trains on it.
    """
    def __init__(
        self,
        logger:         StructuredLogger,
        chunk_size:     int = 2048,
        lang_whitelist: Tuple[str, ...] = ("en",),
        headless:       bool = True,
        delay_seconds:  float = 1.0,
        keywords:       Optional[List[str]] = None,
    ):
        self.logger        = logger
        self.chunk_size    = chunk_size
        self.browser       = BrowserAgent(logger, headless=headless)
        self.qc            = DatasetQualityControl(logger, lang_whitelist=lang_whitelist)
        self.delay_seconds = delay_seconds   # be polite — don't hammer servers
        self.keywords      = keywords        # e.g. ["algorithm","function","formula"]
        self._pdf_builder  = PDFCorpusBuilder(logger, chunk_size=chunk_size)

    def scrape_url(self, url: str) -> str:
        try:
            return self.browser.visit(url)
        except Exception as e:
            self.logger.warn(f"Scrape failed [{url}]: {e}")
            return ""

    def scrape_relevant(self, url: str) -> Dict:
        """Keyword-targeted version — returns {matched_text, code_blocks}
        instead of one flat string. Needs self.keywords to be set."""
        try:
            return self.browser.extract_relevant(url, self.keywords or [])
        except Exception as e:
            self.logger.warn(f"Targeted scrape failed [{url}]: {e}")
            return {"url": url, "keywords": self.keywords, "matched_text": [], "code_blocks": []}

    def build_from_urls(self, urls: List[str], save_to: Optional[str] = None) -> List[str]:
        """Visit every URL, quality-filter the result, return only clean
        chunks. URLs ending in .pdf are streamed straight through
        PDFCorpusBuilder (in-memory, no disk write) instead of HTML
        cleaning. If self.keywords is set, only chunks mentioning at
        least one keyword survive — that's the 'high value only' filter.
        Pass save_to="out.jsonl" to persist the result automatically."""
        chunks = []
        for i, url in enumerate(urls, 1):
            self.logger.info(f"[{i}/{len(urls)}] Scraping {url}")

            if url.lower().endswith(".pdf"):
                text = self._pdf_builder.extract_pdf_from_url(url)
            else:
                text = self.scrape_url(url)

            if not text:
                continue
            for rough_chunk in self.qc.semantic_chunking(text, self.chunk_size):
                result = self.qc.validate(rough_chunk, keywords=self.keywords)
                if result["ok"]:
                    chunks.append(result["text"])
            if self.delay_seconds:
                time.sleep(self.delay_seconds)

        report = self.qc.report()
        self.logger.info(
            f"Web corpus built: {report['accepted']} clean chunks kept "
            f"({report['rejected']} rejected) from {len(urls)} URLs "
            f"| accept_rate={report['accept_rate']*100:.1f}%"
        )
        self.logger.info(f"Rejection breakdown: {report['breakdown']}")
        if save_to:
            save_chunks_jsonl(chunks, save_to, source_tag="web_url", keywords=self.keywords)
            self.logger.info(f"Saved {len(chunks)} chunks → {save_to}")
        return chunks

    def quality_report(self) -> Dict:
        return self.qc.report()


class FullPreTrainer:
    """
    Full causal LM pre-training — ALL parameters are trained.
    (Compare with EnterpriseTrainer which does LoRA fine-tuning only.)

    Key design choices vs fine-tuning:
      context_length : 2048 tokens (vs 512)
      LR             : 3e-4 with cosine schedule + linear warmup
      weight_decay   : 0.1  (regularises large weights)
      precision      : bfloat16 preferred (stabler than fp16)
      block packing  : texts concatenated → no wasted padding tokens
      optim          : fused AdamW (faster on CUDA)
      gradients      : checkpointing always on (saves ~40% VRAM)
    """
    def __init__(
        self,
        model_name:     str,
        logger:         StructuredLogger,
        context_length: int  = 2048,
        from_scratch:   bool = False,
    ):
        self.logger         = logger
        self.context_length = context_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = context_length

        dtype = (torch.bfloat16
                 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                 else torch.float16)

        if from_scratch:
            from transformers import AutoConfig
            config     = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(config).to(dtype)
            self.logger.info("Model initialised from SCRATCH (random weights)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, device_map="auto"
            )
            self.logger.info(f"Continuing pre-training from: {model_name}")

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        self.logger.info(f"Total params: {total_params:.2f}B  |  dtype={dtype}")

    # ── Dataset preparation ──────────────────────────────────
    def prepare_pretrain_dataset(self, texts: List[str]) -> "Dataset":
        """
        Tokenise and pack texts into fixed-length blocks.
        Block packing = no padding waste = ~2× data efficiency vs naïve truncation.
        """
        self.logger.info(f"Tokenising {len(texts)} documents…")

        raw_ds    = Dataset.from_dict({"text": texts})
        tokenized = raw_ds.map(
            lambda ex: self.tokenizer(ex["text"], truncation=False, padding=False),
            batched=True, remove_columns=["text"], desc="Tokenising",
        )

        def group_texts(examples):
            concat = {k: sum(examples[k], []) for k in examples}
            total  = (len(concat["input_ids"]) // self.context_length) * self.context_length
            result = {
                k: [v[i : i + self.context_length]
                    for i in range(0, total, self.context_length)]
                for k, v in concat.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        packed = tokenized.map(group_texts, batched=True, desc="Block packing")
        self.logger.info(
            f"Pre-train dataset: {len(packed)} blocks × {self.context_length} tokens"
        )
        return packed

    # ── Main training entry ──────────────────────────────────
    def pretrain(
        self,
        texts:         List[str],
        output_dir:    str   = "./pretrained_model",
        num_epochs:    int   = 1,
        batch_size:    int   = 1,
        grad_accum:    int   = 16,
        learning_rate: float = 3e-4,
        warmup_ratio:  float = 0.03,
        weight_decay:  float = 0.1,
        save_steps:    int   = 500,
    ):
        """
        Full pre-training loop.
          Effective batch size = batch_size × grad_accum × num_gpus
          For a 70B model: batch_size=1, grad_accum=128, 8×A100 80GB needed.
        """
        dataset = self.prepare_pretrain_dataset(texts)
        eval_n  = min(100, max(1, int(len(dataset) * 0.01)))
        split   = dataset.train_test_split(test_size=eval_n)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        optim    = ("adamw_torch_fused" if torch.cuda.is_available()
                    else "adamw_torch")

        args = TrainingArguments(
            output_dir                  = output_dir,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            num_train_epochs            = num_epochs,
            learning_rate               = learning_rate,
            lr_scheduler_type           = "cosine",
            warmup_ratio                = warmup_ratio,
            weight_decay                = weight_decay,
            bf16                        = use_bf16,
            fp16                        = use_fp16,
            gradient_checkpointing      = True,
            optim                       = optim,
            logging_steps               = 10,
            save_steps                  = save_steps,
            save_strategy               = "steps",
            eval_strategy               = "steps",
            eval_steps                  = save_steps,
            load_best_model_at_end      = True,
            dataloader_num_workers      = 2,
            remove_unused_columns       = False,
            report_to                   = "none",
        )

        self.logger.info("══ FULL PRE-TRAINING STARTED ══")
        self.logger.info(f"  Train blocks : {len(split['train'])}")
        self.logger.info(f"  Eval  blocks : {len(split['test'])}")
        self.logger.info(f"  Context len  : {self.context_length} tokens")
        self.logger.info(f"  Eff. batch   : {batch_size * grad_accum}  |  LR={learning_rate} cosine")

        trainer = Trainer(
            model         = self.model,
            args          = args,
            train_dataset = split["train"],
            eval_dataset  = split["test"],
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.logger.error("CUDA OOM — reduce batch_size or context_length")

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Pre-trained model saved → {output_dir}")
        return trainer

    # ── Convenience: pretrain straight from PDF folder ───────
    def pretrain_from_pdfs(
        self,
        pdf_dir:    str,
        output_dir: str = "./pretrained_model",
        **kwargs,
    ):
        """Build corpus from a PDF directory then run pretrain()."""
        builder = PDFCorpusBuilder(self.logger, chunk_size=self.context_length)
        texts   = builder.build_from_directory(pdf_dir)
        if not texts:
            self.logger.error(f"No text extracted from {pdf_dir}")
            return None
        return self.pretrain(texts, output_dir=output_dir, **kwargs)

    # ── Convenience: pretrain straight from a list of URLs ───
    def pretrain_from_urls(
        self,
        urls:           List[str],
        output_dir:     str = "./pretrained_model",
        lang_whitelist: Tuple[str, ...] = ("en",),
        **kwargs,
    ):
        """Scrape + quality/safety-filter a list of URLs, then run pretrain()."""
        builder = WebCorpusBuilder(
            self.logger, chunk_size=self.context_length, lang_whitelist=lang_whitelist,
        )
        texts = builder.build_from_urls(urls)
        if not texts:
            self.logger.error("No training-quality text survived the QC pipeline for the given URLs")
            return None
        return self.pretrain(texts, output_dir=output_dir, **kwargs)
