from .deps import *
from .logger import StructuredLogger

class BrowserAgent:
    """
    Playwright (preferred) → Selenium fallback → requests fallback.
    """
    def __init__(self, logger: StructuredLogger, headless: bool = True):
        self.logger   = logger
        self.headless = headless
        self._backend = None

    # ── internal: detect backend ─────────────────────────────
    def _detect(self) -> str:
        if _PW_OK:   return "playwright"
        if _SEL_OK:  return "selenium"
        return "requests"

    # ── navigate + get text ──────────────────────────────────
    def visit(self, url: str) -> str:
        backend = self._detect()
        self.logger.info(f"Browser [{backend}] → {url}")

        if backend == "playwright":
            return self._playwright_visit(url)
        if backend == "selenium":
            return self._selenium_visit(url)
        return self._requests_visit(url)

    # ── internal: fetch RAW html only, no cleaning (shared by
    #    visit() and extract_relevant() so there's one fetch path) ──
    def _get_raw_html(self, url: str) -> str:
        backend = self._detect()
        self.logger.info(f"Browser [{backend}] → {url}")
        if backend == "playwright":
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                page    = browser.new_page()
                page.goto(url, timeout=30_000)
                html = page.content()
                browser.close()
            return html
        if backend == "selenium":
            opts = ChromeOptions()
            if self.headless:
                opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            driver = webdriver.Chrome(options=opts)
            try:
                driver.get(url)
                time.sleep(2)
                return driver.page_source
            finally:
                driver.quit()
        return requests.get(url, timeout=30).text

    def _clean_html_content(self, html_content: str) -> str:
        """Web page se ads, scripts, footers aur navigation links ka kachra saaf karne ke liye helper"""
        soup = BeautifulSoup(html_content, "html.parser")
        # Boilerplate tags ko filter out karein
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()
        return soup.get_text(separator="\n")

    # ── NEW: keyword-targeted extraction, code blocks kept separate ─
    def extract_relevant(
        self,
        url: str,
        keywords: List[str],
        max_chars_per_block: int = 2000,
    ) -> Dict:
        """
        'Sirf wahi text/code copy karo jahan algorithm/function/formula
        jaise keywords milein' — this is that. Two passes over the same
        page, kept SEPARATE on purpose:

          - code_blocks : every <pre>/<code> tag's content, kept
                           UNCONDITIONALLY (real code rarely contains the
                           literal word "algorithm" — filtering code by
                           keyword match would throw away the exact thing
                           you actually want).
          - matched_text: prose paragraphs that mention at least one of
                           your keywords (word-boundary, case-insensitive).

        Boilerplate tags (script/style/nav/footer/header) are stripped
        before either pass, same as visit().
        """
        html = self._get_raw_html(url)
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()

        code_blocks = []
        for tag in soup.find_all(["pre", "code"]):
            if tag.find_parent(["pre", "code"]) is not None:
                continue   # nested <code> inside <pre> — already covered by the parent
            code_text = tag.get_text()
            if code_text.strip():
                code_blocks.append(code_text.strip()[:max_chars_per_block])
            tag.extract()   # don't let it also show up duplicated in matched_text

        patterns = [re.compile(rf"\b{re.escape(kw.lower())}\b") for kw in keywords]
        matched_text = []
        for block in soup.get_text(separator="\n").split("\n\n"):
            block = block.strip()
            if block and any(p.search(block.lower()) for p in patterns):
                matched_text.append(block[:max_chars_per_block])

        self.logger.info(
            f"extract_relevant[{url}]: {len(matched_text)} matched text blocks, "
            f"{len(code_blocks)} code blocks (keywords={keywords})"
        )
        return {"url": url, "keywords": keywords, "matched_text": matched_text, "code_blocks": code_blocks}

    def _playwright_visit(self, url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page    = browser.new_page()
            page.goto(url, timeout=30_000)
            html = page.content()
            browser.close()
        return self._clean_html_content(html)

    def _selenium_visit(self, url: str) -> str:
        opts = ChromeOptions()
        if self.headless:
            opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=opts)
        try:
            driver.get(url)
            time.sleep(2)
            html = driver.page_source
            return self._clean_html_content(html)
        finally:
            driver.quit()

    def _requests_visit(self, url: str) -> str:
        r = requests.get(url, timeout=30)
        return self._clean_html_content(r.text)

    # ── click element by text ────────────────────────────────
    def click(self, url: str, selector: str) -> str:
        if not _PW_OK:
            return "Playwright required for click()."
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page    = browser.new_page()
            page.goto(url, timeout=30_000)
            page.click(selector)
            time.sleep(1)
            text = page.inner_text("body")
            browser.close()
        return text[:3000]

    # ── fill form ────────────────────────────────────────────
    def fill_form(self, url: str, fields: Dict[str, str]) -> str:
        if not _PW_OK:
            return "Playwright required for fill_form()."
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page    = browser.new_page()
            page.goto(url, timeout=30_000)
            for selector, value in fields.items():
                page.fill(selector, value)
            text = page.inner_text("body")
            browser.close()
        return text[:3000]

    # ── screenshot ───────────────────────────────────────
    def screenshot(self, url: str, save_path: str = "/tmp/page.png") -> str:
        if not _PW_OK:
            return "Playwright required for screenshot()."
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page    = browser.new_page()
            page.goto(url, timeout=30_000)
            page.screenshot(path=save_path, full_page=True)
            browser.close()
        return save_path
