from .deps import *

class DatasetQualityControl:
    """
    Elite multi-stage data-quality + safety pipeline for ANY raw text
    source (scraped web pages, PDFs, etc.) before it is allowed anywhere
    near a training run.

    Every chunk must pass ALL of these gates:
        1. normalize   – encoding / whitespace cleanup
        2. structural  – length, alphabet ratio, repetition, boilerplate
        3. language    – keep only the language(s) you actually want
        4. coherence   – heuristically reject keyword-soup / nav-menu /
                          link-dump junk that isn't real prose
        5. safety      – reject clearly unsafe content, redact PII
        6. dedup       – exact + near-duplicate removal (SimHash + LSH)

    Note on "checking 100 times": re-running an identical, deterministic
    check on the same text 100 times always returns the same answer, so
    it adds nothing. What actually raises quality is (a) many DIFFERENT
    kinds of checks — which is what stages 1-5 are — and (b) comparing
    every new chunk against EVERYTHING seen so far, which is exactly
    what the dedup stage does. On a corpus of 10,000 pages that's tens
    of millions of real comparisons, not a meaningless loop.
    """

    # Junk/nav phrases that often survive HTML boilerplate stripping
    _BOILERPLATE_PHRASES = [
        "click here", "subscribe to our newsletter", "all rights reserved",
        "cookie policy", "terms of service", "privacy policy", "sign up now",
        "accept all cookies", "javascript is disabled", "404 not found",
        "page not found", "skip to content", "add to cart", "buy now",
        "related articles", "share this post", "leave a comment",
    ]

    # Coarse first-pass safety net. NOT a substitute for a dedicated
    # toxicity/safety classifier in a production pipeline (pair this with
    # something like Detoxify / Perspective API for anything real) — this
    # just stops the most obvious unsafe categories from slipping through.
    _UNSAFE_PATTERNS = [
        re.compile(r"\bhow (?:to|do i)\b[^.?!\n]{0,60}\b(?:build|make|synthesi[sz]e)\b[^.?!\n]{0,60}\b(?:bomb|explosive|nerve agent|biological weapon|chemical weapon)\b"),
        re.compile(r"\bchild\s+(?:sexual|porn|abuse)\b"),
        re.compile(r"\bhow to (?:make|cook|synthesi[sz]e)\b[^.?!\n]{0,40}\b(?:meth|methamphetamine|fentanyl)\b"),
    ]

    _PII_PATTERNS = {
        "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SSN":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CARD":  re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    }

    _STOPWORDS = {
        "the", "is", "a", "an", "and", "or", "of", "to", "in", "that", "it",
        "for", "on", "with", "as", "this", "be", "are", "was", "were", "at",
        "by", "from", "but", "not", "have", "has", "had", "i", "you", "we",
        "they", "he", "she",
    }

    def __init__(
        self,
        logger,
        lang_whitelist: Tuple[str, ...] = ("en",),
        min_words: int = 8,
        max_repetition_ratio: float = 0.30,
        redact_pii: bool = True,
        near_dup_hamming_threshold: int = 4,
    ):
        self.logger                = logger
        self.lang_whitelist        = set(lang_whitelist) if lang_whitelist else None
        self.min_words             = min_words
        self.max_repetition_ratio  = max_repetition_ratio
        self.redact_pii            = redact_pii
        self.near_dup_hamming_threshold = near_dup_hamming_threshold

        self._exact_hashes = set()
        self._lsh_buckets  = defaultdict(set)   # (band_idx, band_value) -> {simhash, ...}
        self._num_bands    = 4
        self._band_bits    = 16                 # 4 * 16 = 64-bit simhash

        self.stats = Counter()

    # ── Stage 1: normalize ────────────────────────────────────
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean(self, text: str) -> str:
        """Kept for backward compatibility with older callers."""
        return self.normalize(text)

    # ── Stage 2: structural sanity ────────────────────────────
    @staticmethod
    def _alpha_ratio(text: str) -> float:
        if not text:
            return 0.0
        alpha = sum(1 for c in text if c.isalpha() or c.isspace())
        return alpha / len(text)

    @staticmethod
    def _repetition_ratio(words: List[str]) -> float:
        if len(words) < 4:
            return 0.0
        counts = Counter(w.lower() for w in words)
        return counts.most_common(1)[0][1] / len(words)

    def is_structurally_sound(self, text: str) -> Tuple[bool, str]:
        words = text.split()
        if len(words) < self.min_words:
            return False, "too_short"
        if self._alpha_ratio(text) < 0.6:
            return False, "low_alpha_ratio"
        if self._repetition_ratio(words) > self.max_repetition_ratio:
            return False, "high_repetition"
        lower = text.lower()
        if sum(1 for p in self._BOILERPLATE_PHRASES if p in lower) >= 3:
            return False, "boilerplate"
        return True, "ok"

    # ── Stage 3: language ─────────────────────────────────────
    def detect_language(self, text: str) -> str:
        if _LANGDETECT_OK:
            try:
                return _langdetect_detect(text)
            except Exception:
                return "unknown"
        # Heuristic fallback when langdetect isn't installed
        letters = re.findall(r"[A-Za-z]", text)
        return "en" if letters and len(letters) / max(len(text), 1) > 0.5 else "unknown"

    def is_allowed_language(self, text: str) -> bool:
        if not self.lang_whitelist:
            return True
        return self.detect_language(text) in self.lang_whitelist

    # ── Stage 4: logical coherence ────────────────────────────
    def is_logically_coherent(self, text: str) -> bool:
        """
        Heuristically reject text that doesn't read like real prose:
        keyword dumps, nav menus, link lists, scraped-junk word salad.
        Real writing almost always has a healthy stopword share and a
        sane average word length; junk usually doesn't.
        """
        words = re.findall(r"[A-Za-z']+", text.lower())
        if len(words) < self.min_words:
            return False
        stop_ratio = sum(1 for w in words if w in self._STOPWORDS) / len(words)
        if stop_ratio < 0.04:
            return False
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 12 or avg_len < 2.5:
            return False
        if len(words) > 40 and not re.search(r"[.!?]", text):
            return False
        return True

    # ── Stage 4b: topic relevance (NEW — opt-in, off by default) ─
    def is_topically_relevant(self, text: str, keywords: Optional[List[str]]) -> bool:
        """
        'High-value content' filter: keep a chunk only if it actually
        mentions at least one of your target keywords/topics (e.g.
        ["algorithm", "function", "formula"]). Word-boundary matched,
        case-insensitive, so "formula" doesn't match "formulaic" by
        accident but does match "Formula:" or "the formula is".
        If keywords is None/empty, every chunk passes (old behavior,
        100% backward compatible — nothing breaks for existing callers).
        """
        if not keywords:
            return True
        lower = text.lower()
        return any(re.search(rf"\b{re.escape(kw.lower())}\b", lower) for kw in keywords)

    # ── Stage 5: safety + PII ─────────────────────────────────
    def contains_unsafe_content(self, text: str) -> bool:
        lower = text.lower()
        return any(p.search(lower) for p in self._UNSAFE_PATTERNS)

    def redact_pii_text(self, text: str) -> str:
        for label, pattern in self._PII_PATTERNS.items():
            text = pattern.sub(f"[REDACTED_{label}]", text)
        return text

    # ── Stage 6: dedup — SimHash + LSH banding ────────────────
    @staticmethod
    def _simhash(text: str, bits: int = 64) -> int:
        tokens = re.findall(r"\w+", text.lower())
        v = [0] * bits
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(bits):
                v[i] += 1 if (h >> i) & 1 else -1
        fp = 0
        for i in range(bits):
            if v[i] >= 0:
                fp |= (1 << i)
        return fp

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def _bands(self, fingerprint: int) -> List[int]:
        mask = (1 << self._band_bits) - 1
        return [(fingerprint >> (i * self._band_bits)) & mask for i in range(self._num_bands)]

    def is_duplicate(self, text: str) -> bool:
        """Exact dedup via hash, near-dup via SimHash + LSH banding
        (so we don't have to compare against every prior chunk one by one)."""
        exact_key = hashlib.sha256(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()
        if exact_key in self._exact_hashes:
            return True

        fp = self._simhash(text)
        bands = self._bands(fp)
        candidates = set()
        for i, b in enumerate(bands):
            candidates |= self._lsh_buckets[(i, b)]
        for other_fp in candidates:
            if self._hamming(fp, other_fp) <= self.near_dup_hamming_threshold:
                return True

        # Not a duplicate — register it so future chunks get checked against it too
        self._exact_hashes.add(exact_key)
        for i, b in enumerate(bands):
            self._lsh_buckets[(i, b)].add(fp)
        return False

    # ── Master pipeline ───────────────────────────────────────
    def validate(self, raw_text: str, keywords: Optional[List[str]] = None) -> Dict:
        """
        Runs ONE chunk of text through every gate above.
        `keywords`: optional list like ["algorithm","function","formula"].
        If given, chunks that don't mention ANY of them are rejected as
        "off_topic" — this is the "high-value content only" filter.
        Leave it as None (default) for the original generic-quality-only
        behavior.
        Returns {"ok": bool, "text": str|None, "reason": str}
        """
        self.stats["seen"] += 1
        text = self.normalize(raw_text)
        if not text:
            self.stats["rejected_empty"] += 1
            return {"ok": False, "text": None, "reason": "empty"}

        sound, reason = self.is_structurally_sound(text)
        if not sound:
            self.stats[f"rejected_{reason}"] += 1
            return {"ok": False, "text": None, "reason": reason}

        if not self.is_allowed_language(text):
            self.stats["rejected_language"] += 1
            return {"ok": False, "text": None, "reason": "language"}

        if not self.is_logically_coherent(text):
            self.stats["rejected_incoherent"] += 1
            return {"ok": False, "text": None, "reason": "incoherent"}

        if not self.is_topically_relevant(text, keywords):
            self.stats["rejected_off_topic"] += 1
            return {"ok": False, "text": None, "reason": "off_topic"}

        if self.contains_unsafe_content(text):
            self.stats["rejected_unsafe"] += 1
            return {"ok": False, "text": None, "reason": "unsafe_content"}

        if self.is_duplicate(text):
            self.stats["rejected_duplicate"] += 1
            return {"ok": False, "text": None, "reason": "duplicate"}

        if self.redact_pii:
            text = self.redact_pii_text(text)

        self.stats["accepted"] += 1
        return {"ok": True, "text": text, "reason": "ok"}

    def batch_validate(self, texts: List[str], keywords: Optional[List[str]] = None) -> List[str]:
        """Convenience: returns only the chunks that pass every gate."""
        out = []
        for t in texts:
            r = self.validate(t, keywords=keywords)
            if r["ok"]:
                out.append(r["text"])
        return out

    def semantic_chunking(self, text: str, chunk_size: int = 500) -> List[str]:
        """Splits on sentence boundaries (./?/!) so abbreviations and
        decimals don't fragment text as badly as a naive '.'-split would."""
        sentences = re.split(r"(?<=[.?!])\s+", text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) < chunk_size:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
        return chunks

    def report(self) -> Dict:
        """Summary of what's been seen/accepted/rejected (and why) so far."""
        seen = self.stats["seen"] or 1
        accepted = self.stats["accepted"]
        return {
            "seen":        self.stats["seen"],
            "accepted":    accepted,
            "rejected":    self.stats["seen"] - accepted,
            "accept_rate": round(accepted / seen, 4),
            "breakdown":   dict(self.stats),
        }

    def reset(self):
        """Clears dedup memory + stats — use this to start a fresh corpus run."""
        self._exact_hashes.clear()
        self._lsh_buckets.clear()
        self.stats.clear()
