from .deps import *

class KnowledgeGraph:
    """
    NetworkX-based knowledge graph with NLP triple extraction.
    Falls back gracefully when networkx is absent.
    """
    def __init__(self):
        if _NX_OK:
            self._g    = nx.DiGraph()
            self._real = True
        else:
            self._edges: List[Tuple] = []
            self._real = False

    # ── add triple ───────────────────────────────────────
    def add_triple(self, subject: str, relation: str, obj: str, weight: float = 1.0):
        if self._real:
            self._g.add_edge(subject, obj, relation=relation, weight=weight)
        else:
            self._edges.append((subject, relation, obj, weight))

    # ── query neighbours ─────────────────────────────────────
    def neighbours(self, entity: str) -> List[Dict]:
        if self._real:
            results = []
            for s, o, data in self._g.out_edges(entity, data=True):
                results.append({"from": s, "relation": data.get("relation"), "to": o})
            return results
        return [(s, r, o) for s, r, o, _ in self._edges if s == entity]

    # ── shortest path ───────────────────────────────────────
    def path(self, src: str, dst: str) -> List[str]:
        if self._real:
            try:
                return nx.shortest_path(self._g, src, dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        return []

    # ── extract triples from text (rule-based) ───────────────
    def ingest_text(self, text: str):
        """Simple subject-verb-object extraction via regex patterns."""
        patterns = [
            r"(\w[\w\s]+?)\s+(is|are|was|were|has|have|contains|includes)\s+([\w\s]+?)(?:\.|,|$)",
            r"(\w[\w\s]+?)\s+(->|→)\s+([\w\s]+?)(?:\.|,|$)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                if s and r and o:
                    self.add_triple(s, r, o)

    def stats(self) -> Dict:
        if self._real:
            return {"nodes": self._g.number_of_nodes(), "edges": self._g.number_of_edges()}
        return {"triples": len(self._edges)}
