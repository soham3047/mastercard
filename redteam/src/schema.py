"""
schema.py — Shared output schema for VINEYARD Red Team (Person A).

This is the exact interface contract consumed by Person B (detection),
Person C (adversarial loop), and Person D (dashboard):

    {
      "nodes": [{"id": "acct_001", "type": "ring|decoy|hub", "created_at": "ISO8601"}],
      "edges": [{"from": "acct_001", "to": "acct_002", "timestamp": "ISO8601",
                 "amount": 452.30, "label": "ring|decoy"}]
    }

Keep this file as the single source of truth for that shape — every
generator imports Node/Edge/TransactionGraph from here rather than
building dicts by hand, so a schema change only has to happen in one
place.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

NodeType = Literal["ring", "decoy", "hub"]
EdgeLabel = Literal["ring", "decoy"]


@dataclass
class Node:
    id: str
    type: NodeType
    created_at: str  # ISO8601


@dataclass
class Edge:
    from_: str
    to: str
    timestamp: str  # ISO8601
    amount: float
    label: EdgeLabel

    def to_dict(self) -> Dict[str, Any]:
        # "from" is a Python keyword, hence from_ internally but "from" on the wire.
        return {
            "from": self.from_,
            "to": self.to,
            "timestamp": self.timestamp,
            "amount": round(self.amount, 2),
            "label": self.label,
        }


@dataclass
class TransactionGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def merge(self, other: "TransactionGraph") -> None:
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)

    def validate(self) -> None:
        node_ids = [n.id for n in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            dupes = {i for i in node_ids if node_ids.count(i) > 1}
            raise ValueError(f"Duplicate node ids in graph: {dupes}")
        node_id_set = set(node_ids)
        for e in self.edges:
            if e.from_ not in node_id_set:
                raise ValueError(f"Edge references unknown node '{e.from_}'")
            if e.to not in node_id_set:
                raise ValueError(f"Edge references unknown node '{e.to}'")
            if e.amount <= 0:
                raise ValueError(f"Non-positive amount on edge {e.from_}->{e.to}: {e.amount}")
            if e.from_ == e.to:
                raise ValueError(f"Self-loop edge not allowed: {e.from_}->{e.to}")


def write_graph_json(graph: TransactionGraph, path: str) -> None:
    graph.validate()
    with open(path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)
