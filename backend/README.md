# Backend — Person D

FastAPI service. D1 stubs endpoints with hardcoded JSON matching the shared
schemas so frontend work doesn't wait on teammates; D3/D4 swap each endpoint
to real output from A/B/C as their synopses confirm interfaces are live.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Log progress in `PROGRESS_D.md` (shared with `../frontend/`).
