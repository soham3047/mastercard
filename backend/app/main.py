"""VINEYARD backend — FastAPI app skeleton (D0).

D1 adds stub endpoints here for: graph data, persistence diagram, RPS,
round results, cross-bank verification — returning hardcoded JSON matching
the schemas in the root README until A/B/C's real output is wired in (D3/D4).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VINEYARD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before submission
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# --- D1: add stub endpoints below, e.g. ---
# @app.get("/api/graph")
# def get_graph():
#     return {"nodes": [...], "edges": [...]}
