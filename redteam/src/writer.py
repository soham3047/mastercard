"""
writer.py — Stage A0 (stub output writer) / used by every later stage.

Central place that writes the final graph JSON to disk in the exact
interface-contract shape, plus a companion identities.json (extra
content, not part of the strict nodes/edges schema, that Person D's
dashboard uses for the "explain this account" panel).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict

from .schema import Edge, Node, TransactionGraph, write_graph_json


def write_output(graph: TransactionGraph, identities: Dict[str, Dict[str, str]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "transaction_graph.json")
    identities_path = os.path.join(out_dir, "identities.json")

    write_graph_json(graph, graph_path)
    with open(identities_path, "w") as f:
        json.dump(identities, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(graph.nodes)} nodes / {len(graph.edges)} edges -> {graph_path}")
    print(f"Wrote {len(identities)} identities -> {identities_path}")


def stub_output(out_dir: str) -> None:
    """Stage A0: emits placeholder data matching the schema exactly, so
    downstream teammates (B, D) can build against the real file shape
    from day one, before any real generator exists."""
    now = datetime.now(timezone.utc).isoformat()
    graph = TransactionGraph(
        nodes=[
            Node(id="acct_001", type="ring", created_at=now),
            Node(id="acct_002", type="ring", created_at=now),
            Node(id="acct_003", type="decoy", created_at=now),
        ],
        edges=[
            Edge(from_="acct_001", to="acct_002", timestamp=now, amount=452.30, label="ring"),
            Edge(from_="acct_002", to="acct_001", timestamp=now, amount=9800.00, label="ring"),
        ],
    )
    identities = {
        "acct_001": {
            "account_id": "acct_001", "name": "PLACEHOLDER", "occupation": "PLACEHOLDER",
            "income_bracket": "PLACEHOLDER", "address_plausibility": "PLACEHOLDER",
            "narrative": "PLACEHOLDER — real identities land in Stage A5",
        },
    }
    write_output(graph, identities, out_dir)


if __name__ == "__main__":
    stub_output("output/stub")
