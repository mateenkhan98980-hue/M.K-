from .deps import *
from .sandbox import SecureSandbox

def calc_tool(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                   ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Constant)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError(f"Unsafe: {type(node).__name__}")
        return str(eval(compile(tree, "<string>", "eval")))
    except Exception as e:
        return f"Calc error: {e}"

def python_tool(code: str) -> str:
    sandbox = SecureSandbox(timeout=10)
    result  = sandbox.run_python(code)
    return result["stdout"] or result["stderr"]

def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Real web search — no API key required.
    Queries DuckDuckGo's HTML endpoint (https://html.duckduckgo.com/html/),
    parses the result list with BeautifulSoup, and returns a clean,
    numbered summary of titles + snippets + links.

    Falls back to a clear error string (not a fake "Searching: ..." stub)
    if the request fails — e.g. no network access, DNS blocked, etc.
    """
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; EnterpriseAIBot/1.0)"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Web search failed (network/HTTP error): {e}"

    soup    = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result")[:max_results]:
        title_tag = result.select_one(".result__title a")
        snip_tag  = result.select_one(".result__snippet")
        if not title_tag:
            continue
        title   = title_tag.get_text(strip=True)
        link    = title_tag.get("href", "")
        snippet = snip_tag.get_text(strip=True) if snip_tag else ""
        results.append(f"{len(results)+1}. {title}\n   {snippet}\n   {link}")

    if not results:
        return f"No results found for: {query}"

    return f"Search results for '{query}':\n" + "\n".join(results)

TOOLS: Dict[str, callable] = {
    "calculator": calc_tool,
    "python":     python_tool,
    "web":        web_search_tool,
}
