# PROGRESS_C.md — Adversarial Loop + Cross-Bank Layer (Person C)

## SYNOPSIS — Person C Stage 0: Scaffold + stub loop

- Files created/changed:
  - `adversarial/requirements.txt`
  - `adversarial/src/__init__.py`
  - `adversarial/src/stub_generator.py`
  - `adversarial/src/stub_scorer.py`
  - `adversarial/src/loop.py`
  - `adversarial/tests/test_smoke.py`

- What it does: Builds the round-loop skeleton (generate → score → log
  round result) entirely against stub functions, so the loop's structure
  is provable independent of A/B being finished. `stub_generator.py`
  fakes a theta dict, `stub_scorer.py` fakes a `score_batch()`-shaped
  result, and `loop.py` runs N rounds, writing each round's result plus
  an aggregate file to disk in the exact "Round Result" JSON shape from
  the root README.

- Key decisions/assumptions made:
  - `generate_fn`/`score_fn` are injectable parameters on `run_loop()`
    specifically so C1 (GP + acquisition function) and C2 (real A/B
    wiring) can swap in real implementations without touching the loop's
    control flow.
  - The stub's theta keys (`ring_size`, `decoy_density`,
    `hop_timing_params`, `fan_out_ratio`) intentionally follow the master
    prompt's rough sketch, not A's real `Theta` — see open questions below,
    since A's synopsis (already available) shows the real keys differ.
  - `bottleneck_distance_from_prev_round` is a random placeholder (0.0 on
    round 1, random after) since no real persistence diagrams exist yet at
    this stage — it's structurally present in the output so D can build
    the dashboard field-mapping now, but the number itself is meaningless
    until C2.
  - Round results are written both incrementally (`round_N.json` per
    round) and as one aggregate (`all_rounds.json`), since it wasn't clear
    from the README which D would prefer — flagged as an open question
    rather than guessed silently.

- Interface it EXPOSES:
  `run_loop(n_rounds, out_dir, seed, generate_fn, score_fn) ->` list of
  ```json
  {"round": 3, "theta_used": {...}, "recall": 0.61,
   "bottleneck_distance_from_prev_round": 0.44}
  ```
  — matches the root README's Round Result contract exactly.

- Interface it CONSUMES:
  `stub_generator.fake_generate_theta` and `stub_scorer.fake_score` —
  both local stubs, not A's or B's real output yet.

- Commands to run:
  ```bash
  cd adversarial
  pip install -r requirements.txt
  python -m src.loop
  python tests/test_smoke.py
  ```

- Open questions for the team:
  - A's real `Theta` (her Stage 6 synopsis) uses different key names than
    this stub (`ring_size_lambda`, `num_rings`, `hub_out_degree_lambda`,
    `hop_timing_lambda`, `hawkes_baseline_mu/kappa/beta`,
    `ring_amount_mu/sigma`, `decoy_amount_mu/sigma`,
    `reporting_threshold_inr`, `structuring_margin`, `decoy_density`,
    `num_decoy_accounts`, `seed` — `fan_out_ratio` is the one name that's
    actually shared). C1/C2 should import `Theta`/`DEFAULT_THETA` directly
    from `redteam.src.theta` per A's own note, not hardcode either this
    stub's keys or a guess at A's.
  - B's real `update_diagram(prev_diagram, new_edges)` returns
    `(diagram, state)`, where `state` is a live `PersistentGraph` object
    that must be kept alive across rounds (per her Stage 3 synopsis) — not
    the bare-diagram-in/bare-diagram-out shape this stage assumes.
    `loop.py`'s current bottleneck-distance stub doesn't track diagram
    state at all; C2 needs to decide whether the loop owns that state
    object across its `for round_num in range(...)` iterations, or asks B
    for a stateless wrapper around it.
  - Confirm with D whether round JSON should be written per-round
    (`round_N.json`, current default) or only as the aggregate
    `all_rounds.json` — happy to drop one if it's redundant for the
    dashboard's polling model.

## SYNOPSIS — Person C Stage 1: Bayesian optimization core

- Files created/changed:
  - `adversarial/src/bo.py`
  - `adversarial/tests/test_bo_smoke.py`

- What it does: Adds the GP surrogate + Expected Improvement acquisition
  core (`BayesOptimizer`, built on scikit-optimize's `Optimizer`) that
  proposes each round's theta instead of drawing one randomly like C0's
  stub. `make_bo_generate_and_score_fns()` wires one `BayesOptimizer`'s
  `ask()`/`tell()` around a scorer into a `(generate_fn, score_fn)` pair
  that plugs directly into `loop.run_loop()` — no changes to the loop's
  control flow were needed, confirming C0's injectable-parameter design
  works as intended. Still uses the stub scorer by default, per the
  master prompt's explicit allowance for C1 to do so.

- Key decisions/assumptions made:
  - The GP's search space (`SEARCH_SPACE` in `bo.py`) mirrors
    `stub_generator.fake_generate_theta`'s ranges and keys exactly (not
    A's real `Theta`), so `BayesOptimizer.ask()` output is a drop-in
    `generate_fn` for `loop.run_loop`. This is a stopgap, flagged as an
    open question below, not a resolution of the C0/README open question
    about A's real key names.
  - `hop_timing_params.lambda` is flattened to a single `hop_timing_lambda`
    dimension for the GP (skopt spaces are flat), then re-nested back into
    the `{"lambda": ...}` dict shape on `ask()` so the output still matches
    the stub's theta shape exactly.
  - Objective handed to skopt is `J(theta) = 1 - recall`, implemented
    literally per the master prompt's spec — NOT re-signed to match my own
    reading of "attacker wants low recall." Flagged as an open question
    below rather than silently flipped, since guessing wrong here would
    silently invert what every future round optimizes for.
  - `ask()`/`tell()` are separate calls (not a single `optimize(theta,
    recall)` call) so the same `BayesOptimizer` instance can be wired
    into `loop.run_loop`'s existing `generate_fn`-then-`score_fn`
    call order via closures, without `loop.py` needing to know a GP is
    involved at all.
  - Still uses `stub_scorer.fake_score` by default (master prompt
    explicitly allows this for C1) — but `fake_score` ignores theta's
    actual values, so the GP is currently fitting a surrogate to pure
    noise. The optimizer's mechanics (ask/tell, space, acquisition) are
    real and tested; the *optimization itself* is only meaningful once
    C2 swaps in a theta-dependent score function.

- Interface it EXPOSES:
  - `BayesOptimizer(seed, n_initial_points)` with:
    - `.ask(seed=None, round_num=None) -> theta dict` (same shape as
      `stub_generator.fake_generate_theta`'s output — accepts and ignores
      `seed`/`round_num` for signature-compatibility with `generate_fn`)
    - `.tell(recall, x=None)` — feeds the most recent `ask()`'s theta and
      the observed recall back into the GP as `1 - recall`
  - `make_bo_generate_and_score_fns(seed, score_fn=None) ->
    (generate_fn, score_fn, bo)` — the `(generate_fn, score_fn)` pair is
    directly usable as `loop.run_loop(generate_fn=..., score_fn=...)`
    without modification; `bo` is returned too in case a caller wants to
    inspect `bo.opt` (e.g. D's dashboard, or C2 debugging convergence).

- Interface it CONSUMES:
  - `stub_scorer.fake_score` by default (still a stub, not B's real
    `score_batch` — swappable via `make_bo_generate_and_score_fns(score_fn=...)`)
  - `scikit-optimize`'s `Optimizer`/`Integer`/`Real` (already in
    `requirements.txt` from C0)

- Commands to run:
  ```bash
  cd adversarial
  pip install -r requirements.txt
  python -m src.bo               # runs 8 GP-driven rounds, writes output/bo_rounds/
  python tests/test_bo_smoke.py
  ```

- Open questions for the team:
  - **Objective sign** — `J(theta) = 1 - recall` is implemented exactly as
    the master prompt specifies, but minimizing it (skopt's default)
    pushes the GP towards *higher* recall each round. The stated concept
    ("tunes Red Team's attack parameters specifically against what the
    detector missed") reads like the attacker wants recall to go *down*
    (evade detection), which would mean minimizing `J(theta) = recall`
    instead, or maximizing `1 - recall`. Need a decision before C2 wires
    this against B's real, theta-dependent `score_batch` — right now
    it's a coin flip which direction the loop actually converges.
    **STILL OPEN going into C2 — see below, not resolved this stage
    either.**
  - Same A-real-`Theta`-vs-stub-keys gap from C0 still applies to
    `SEARCH_SPACE` in `bo.py` — it needs to be re-keyed against
    `redteam.src.theta.Theta`/`DEFAULT_THETA`'s real bounds once C2 wires
    in A's real generator, not just the stub's rough-sketch keys.
    **Resolved in C2 — see `bo_real_space.py`.**
  - Should `n_initial_points` (currently 3, pure-random exploration
    before the GP model kicks in) scale with the 3-5 real rounds C2 plans
    to run? With only 3-5 total rounds, most or all of a real run could
    be random exploration before the GP model has enough data to matter
    — worth deciding together whether C2 needs a lower `n_initial_points`
    or a larger round budget. **Still not really resolved in C2 — see
    below, dropped to 2 as a guess, not a team decision.**

## SYNOPSIS — Person C Stage 2: Round loop, wired for real

- Files created/changed:
  - `adversarial/src/loop.py` (modified — added optional `bottleneck_fn`
    param, backward-compatible with C0/C1)
  - `adversarial/src/bo.py` (modified — `BayesOptimizer` now takes
    configurable `search_space`/`vector_to_theta`, defaulting to C1's
    original stub values; `make_bo_generate_and_score_fns` unchanged)
  - `adversarial/src/bo_real_space.py` (new — real search space re-keyed
    against A's actual `Theta`)
  - `adversarial/src/redteam_adapter.py` (new — real adapter around A's
    `build_graph`/`Theta`)
  - `adversarial/src/blueteam_adapter.py` (new — adapter around B's real
    `score_batch`, with a flagged, clearly-labeled fallback)
  - `adversarial/src/bottleneck.py` (new — real bottleneck distance from
    B7's diagram schema, via `persim`)
  - `adversarial/src/run_real_rounds.py` (new — C2's actual entry point)
  - `adversarial/tests/test_c2_smoke.py` (new)

- What it does: Wires the loop for real — generate (A's real
  `build_graph`) → score (B's real `score_batch`, import-verified this
  stage) → optimize (this track's `BayesOptimizer`, now over A's real
  `Theta` fields) → repeat, for 3-5 rounds, with bottleneck distance
  computed directly from B's real persistence diagrams each round. Round
  results are written to disk via the unchanged `loop.run_loop`, in the
  exact Round Result schema.

- Key decisions/assumptions made:
  - **Tuned search space is a 4-field subset of A's 16-field `Theta`**
    (`ring_size_lambda`, `decoy_density`, `hop_timing_lambda`,
    `fan_out_ratio`), matching the master prompt's original rough sketch
    as closely as the real key names allow. The other 12 fields
    (`num_rings`, `num_hubs`, Hawkes params, A's calibrated amount/
    threshold fields, `seed`) are held at `DEFAULT_THETA`'s values. This
    is a decision made to keep the loop real and runnable now, not a
    guess left unflagged — widening the search space is a team call. See
    `bo_real_space.py`'s docstring.
  - **`decoy_density`'s upper bound of 1.0 in `REAL_SEARCH_SPACE` is this
    file's own choice, not one `Theta.validate()` itself enforces** (it
    only checks `>= 0`) — flagged explicitly rather than silently
    assuming a bound A didn't actually set.
  - **Deviated from the master prompt's "use B's `update_diagram`"
    instruction, explicitly flagged, not silent.** `update_diagram`'s
    actual module was never provided to this track, and more importantly
    its `(prev_diagram, new_edges)` contract doesn't map onto this loop's
    real round structure: each round calls A's `build_graph(theta)`,
    which samples an entirely new graph from `theta.seed` — there's no
    "new_edges since last round" to hand it, since rounds aren't
    accumulating one graph. Computed bottleneck distance directly from
    consecutive rounds' full diagrams instead (`bottleneck.py`, via
    `persim`). No incremental-recompute benefit is actually being lost
    here, since `score_batch` already recomputes the diagram from scratch
    every call under this round design regardless of which path is used.
    Real open question for the team below if incremental updates are
    wanted: the round loop's design (not just this file) would need to
    change so rounds mutate a shared graph instead of resampling fresh.
  - **`b7_score_batch.py`'s real file content was never received** —
    only its prose synopsis. `blueteam_adapter.py` attempts a real import
    two ways (package-style `blueteam.src.b7_score_batch`, then a flat
    sys.path-based import matching B's own documented `cd blueteam/src &&
    python b7_score_batch.py` command style) and exposes
    `SCORE_BATCH_IS_REAL` so the caller — and D's dashboard, if it wants
    to check — knows which path actually fired. `run_real_rounds.py`
    prints a loud warning to stderr if it fell back to the stub. This
    needs to be confirmed against the real file/repo layout before a
    demo, not assumed from prose.
  - `persim` is a new dependency (bottleneck distance), not yet added to
    `adversarial/requirements.txt` since that file's current contents
    weren't available to edit directly this stage — see Commands below.
  - `BayesOptimizer.n_initial_points` dropped from C1's 3 to 2 for the
    real run, as a guess (not a team decision) at trading off pure-random
    exploration against the GP model actually having enough data across
    only 3-5 total rounds — same open question carried forward from C1,
    still not really resolved, just nudged.

- Interface it EXPOSES: unchanged from C0/C1's Round Result contract —
  `run_real_rounds.make_real_generate_and_score_fns(seed) ->
  (generate_fn, score_fn, bo)`, directly usable with `loop.run_loop`.
  `loop.run_loop`'s own signature gained one new optional parameter,
  `bottleneck_fn`, which is backward-compatible (defaults to `None`,
  preserving C0/C1's exact behavior and passing test suite).

- Interface it CONSUMES:
  - A's REAL `redteam.src.main.build_graph` / `redteam.src.theta.Theta` /
    `redteam.src.schema.TransactionGraph.to_dict()` — verified against
    the actual files this stage, not guessed.
  - B's `score_batch` — REAL if `blueteam_adapter.SCORE_BATCH_IS_REAL` is
    `True` at runtime; a clearly-labeled, non-real fallback otherwise
    (see above). `update_diagram` — NOT consumed, see deviation above.
  - `persim.bottleneck` — new dependency this stage.

- Commands to run:
  ```bash
  cd adversarial
  pip install persim              # new this stage — add to requirements.txt
  pip install -r requirements.txt
  python -m src.run_real_rounds   # runs 5 real rounds, writes output/c2_real_rounds/
  python tests/test_c2_smoke.py
  ```

- Open questions for the team:
  - **Objective sign, carried forward from C1, still unresolved.**
    `J(theta) = 1 - recall`, minimized, still pushes the GP towards
    higher recall — backwards from "evade the detector" as read from the
    concept description. This is now live against B's real, theta-
    dependent scorer (assuming `SCORE_BATCH_IS_REAL`), so it's no longer
    a theoretical concern — whichever direction is correct should be
    confirmed before D's dashboard/writeup reports round-over-round
    "improvement."
  - **Incremental diagram updates**: is it worth changing the round
    loop's design so rounds mutate a shared/accumulating graph (making
    `update_diagram`'s real contract usable), or is independent-sample-
    per-round (this stage's actual behavior) fine for the demo? See the
    deviation note above.
  - **Confirm `b7_score_batch.py`'s real import path/package layout with
    B** — this stage guessed two reasonable options based on her
    documented run command, but neither has been verified against the
    real file.
  - **Confirm the 4-field tuned search space is the right scope** — happy
    to widen it (e.g. add `num_rings`/`num_hubs`) if the team wants a
    richer optimization surface for the demo.
  - `n_initial_points=2` for a 3-5 round real run — still just a guess,
    not a resolved decision; open since C1.

## SYNOPSIS — Person C Stage 3: Mocked cross-bank commitment layer

- Files created/changed:
  - `adversarial/src/crossbank/__init__.py`
  - `adversarial/src/crossbank/commitment.py`
  - `adversarial/src/crossbank/simulate.py`
  - `adversarial/src/run_cross_bank.py`
  - `adversarial/tests/test_crossbank_smoke.py`

- What it does: Simulates 2-3 banks proving fraud-ring overlap via salted
  hash commitments without sharing raw account data with each other —
  each bank hashes its flagged-fingerprint list with a shared salt,
  publishes only the commitment set, and overlap is estimated from
  commitment-set intersection (Jaccard by default). Entirely
  self-contained per the master prompt's note that this stage doesn't
  depend on A/B/C1/C2 — built with synthetic fingerprints with a
  known-by-construction ground-truth overlap fraction, plus an optional
  hook to swap in B's real `detected_ring_nodes` per bank later.

- Key decisions/assumptions made:
  - **The salt is SHARED/public across all banks, not per-bank-private —
    flagged explicitly, not a shortcut taken quietly.** A literal reading
    of "salted hash commitments" could mean each bank salts its own
    fingerprints independently, but that makes cross-bank matching
    impossible by construction: `H(x||salt_A) != H(x||salt_B)` for the
    same raw `x`, so naive intersection would always be empty regardless
    of true overlap (proven directly in
    `test_independent_salts_never_match`, not just asserted in a
    comment). A shared salt is the only way to make matching work at all
    with plain hashing — but a shared, publicly-known salt is itself a
    real privacy break (dictionary/rainbow-table attack over any
    guessable fingerprint space). `commitment.py`'s docstring spells out
    both failure modes and what real PSI protocols (DH-PSI, OPRF-PSI) do
    instead to avoid needing a shared secret at all. This is exactly why
    every output record carries `"simulated": true` and the module
    docstrings say, unambiguously, "not real ZK/PSI cryptography."
  - `overlap_estimate` defaults to Jaccard (`|A∩B|/|A∪B|`), with a
    `method="containment"` (`|A∩B|/min(|A|,|B|)`) alternative exposed for
    when bank flagged-list sizes differ a lot — not calibrated to any
    real statistic (there's nothing to calibrate against here, unlike
    A's amount distributions), just a documented modeling choice.
  - Ground-truth overlap is baked into the synthetic data generator
    (`known_overlap_fraction`, default 0.35) specifically so the
    computed `overlap_estimate` can be sanity-range-checked against a
    known answer in tests, rather than just producing a
    plausible-looking number with nothing to verify it against.
  - `fingerprints_from_score_batch_result()` is a real-data hook (pulls
    B's `detected_ring_nodes`) but is NOT wired in by default — mixing
    one real bank against synthetic banks wouldn't produce a meaningful
    overlap number anyway (different fingerprint namespaces), so this is
    left as an explicit opt-in via `real_fingerprints` rather than
    silently half-wired.

- Interface it EXPOSES:
  `simulate_cross_bank_verification(bank_names, n_flagged_per_bank,
  known_overlap_fraction, shared_salt, method, seed, real_fingerprints) ->`
  ```json
  {"bank_pairs": [{"bank_a": "Bank1", "bank_b": "Bank2",
                    "overlap_estimate": 0.73, "simulated": true}, ...]}
  ```
  — matches the root README's Cross-Bank Verification schema exactly.
  `run_cross_bank.py` also writes this to `output/cross_bank/bank_pairs.json`
  for D to read directly.

- Interface it CONSUMES: none by default (fully self-contained, synthetic
  data) — optional hook for B's real `detected_ring_nodes` via
  `fingerprints_from_score_batch_result` / `real_fingerprints`, not wired
  in.

- Commands to run:
  ```bash
  cd adversarial
  python -m src.run_cross_bank    # writes output/cross_bank/bank_pairs.json
  python tests/test_crossbank_smoke.py
  ```

- Open questions for the team:
  - Confirm 2 vs 3 banks for the actual demo — code supports either
    (any `bank_names` length ≥ 2), default is 3.
  - Jaccard vs. containment for `overlap_estimate` — worth a team look at
    which reads better on the dashboard; no real-world calibration exists
    either way for this metric, it's a definitional choice.
  - Should real `detected_ring_nodes` (once B's real run is available)
    actually get wired in via `real_fingerprints`, or is this layer meant
    to stay fully synthetic/illustrative for the demo? Left as an
    explicit opt-in rather than guessed.

## SYNOPSIS — Person C Stage 4: Finalize output interfaces

- Files created/changed:
  - `adversarial/tests/test_crossbank_smoke.py` (bugfix — import path)
  - `adversarial/tests/test_c4_output_contract.py` (new)

- What it does: Validates C's two real, runnable output interfaces
  (round-loop output, cross-bank output) against the root README's
  documented schemas by actually generating the files and running the
  whole pipeline end to end, rather than checking prose against prose.
  No new production code — this stage is a verification pass, plus one
  bugfix surfaced by actually running a previously-untested command.

- Key decisions/assumptions made:
  - **Ran everything for real, since no output existed yet.**
    `python -m src.bo` (C1's stub-space GP loop) and
    `python -m src.run_cross_bank` (C3) were both executed fresh; the
    findings below come from their actual output, not synopsis prose.
  - **Found and fixed a real bug in `test_crossbank_smoke.py`**, not
    just a design nit: its own documented run command
    (`cd adversarial && python tests/test_crossbank_smoke.py`, per C3's
    synopsis) crashed with `ModuleNotFoundError: No module named
    'adversarial'`. It imported via `from adversarial.src.crossbank...
    import`, which needs the repo root (parent of `adversarial/`) on
    `sys.path` — nothing in the codebase sets that up, and it's
    inconsistent with `test_bo_smoke.py`'s own working convention
    (`sys.path.insert` to the test's parent dir, then `from src.xxx
    import`). Fixed to match that convention; reran the exact documented
    command afterward and all 5 assertions pass.
  - **Confirmed one harmless schema divergence, not fixed because
    there's nothing to fix:** README's illustrative `theta_used` example
    shows 3 keys (`ring_size`, `decoy_density`, `fan_out_ratio`); real
    output has a 4th, `hop_timing_params` (nested dict), since the
    example was never meant to be exhaustive. Noted explicitly so D
    doesn't assume only 3 keys.
  - **Flagged, not fixed, a real asymmetry between the two adapters:**
    `blueteam_adapter.py` has a documented try/except fallback if B's
    real module isn't importable; `redteam_adapter.py` has none.
    Confirmed by running `python -m src.run_real_rounds` with neither
    `redteam/` nor `blueteam/` present — it hard-crashes at import time
    (`ModuleNotFoundError: No module named 'redteam'`) before reaching
    any fallback logic. Practical effect: C2's "wired for real" round
    loop cannot be exercised at all right now, not even in a
    partial/flagged-fallback way like B's side already supports. Left
    as a team decision (add the same fallback treatment to
    `redteam_adapter.py`, or accept that A's side fails loud by design)
    rather than silently patched, since it changes failure-mode
    philosophy, not just plumbing.
  - **Documented the currently-real file-path contract for D**, since
    the README specifies JSON shapes but not disk locations:
    - `output/bo_rounds/round_N.json` + `all_rounds.json` — schema-real,
      but **values are placeholder** (`stub_scorer`'s random recall,
      `loop.py`'s random bottleneck-distance stub for rounds after the
      first). Not meaningful demo data yet.
    - `output/cross_bank/bank_pairs.json` — schema-real **and**
      semantically meaningful within its documented `"simulated": true`
      caveat.
    - `output/c2_real_rounds/` — does not exist yet; blocked on the
      adapter gap above.

- Interface it EXPOSES: no schema changes this stage.
  `test_c4_output_contract.py` newly exposes two standalone checks —
  `test_round_result_files_match_contract()` and
  `test_cross_bank_file_matches_contract()` — that read the on-disk JSON
  directly (not C's Python objects) and assert it against the README's
  Round Result and Cross-Bank Verification schemas.

- Interface it CONSUMES: the on-disk files written by `src.bo` and
  `src.run_cross_bank` (C1/C3) — read as data, not re-imported as code,
  so this test would catch a future regression even if the producing
  code still "looked" correct.

- Commands to run:
  ```bash
  cd adversarial
  python -m src.bo                         # writes output/bo_rounds/
  python -m src.run_cross_bank              # writes output/cross_bank/bank_pairs.json
  python tests/test_bo_smoke.py
  python tests/test_crossbank_smoke.py      # now passes as documented — see bugfix above
  python tests/test_c4_output_contract.py   # new — validates the on-disk files against the README
  ```

- Open questions for the team:
  - Should `redteam_adapter.py` get `blueteam_adapter.py`'s fallback
    treatment, so `run_real_rounds.py` can execute end-to-end (loudly
    flagged non-real for A's part) before A's real files exist — or is
    a hard crash the intended behavior for a genuinely-missing hard
    dependency?
  - `output/c2_real_rounds/` doesn't exist yet — worth telling D
    explicitly not to expect it until the above is resolved, so D
    doesn't build a dashboard path against a file that isn't there.
  - `hop_timing_params` divergence from the README's illustrative
    `theta_used` example — confirmed harmless; flagging only so it's
    written down rather than rediscovered later.

## ADDENDUM — Person C Stage 4: two more test files found (`test_smoke.py`, `test_c2_smoke.py`)

These weren't included in the original C4 pass — surfaced afterward when
checking the `tests/` directory contents. Same verification approach as
the rest of C4: run for real, don't infer from the filename.

- `test_smoke.py` (C0's original stub-loop test) — **runs clean, as-is,
  no fix needed.** Self-contained: calls `run_loop()` directly and
  asserts on the in-memory result, so it needs no output generated
  beforehand. `cd adversarial && python tests/test_smoke.py` passes.

- `test_c2_smoke.py` (C2's real-pipeline smoke test) — **had the same
  import bug `test_crossbank_smoke.py` had**, independently: imported via
  `from adversarial.src.X import Y`, which needs the repo root on
  `sys.path` and isn't how anything else in the suite is invoked. Fixed
  to the same `sys.path.insert` + `from src.X import Y` convention used
  by `test_bo_smoke.py` / `test_smoke.py` / the fixed
  `test_crossbank_smoke.py`.
  - After that fix, it fails on exactly one thing, cleanly:
    `ModuleNotFoundError: No module named 'redteam'` — this is the
    already-known, already-flagged gap (`redteam_adapter.py` has no
    fallback, unlike `blueteam_adapter.py`), not a new bug. Confirms the
    C4 finding rather than adding a new one.
  - Net effect of the fix: once A's real `redteam/` package exists (or
    `redteam_adapter.py` gets a fallback), this test will run
    immediately with no further changes needed — it was only the wrong
    bug standing in front of the real, known blocker.

- Updated full test inventory (5 files total):

  | Test | Needs generation first? | Runs today? |
  |---|---|---|
  | `test_smoke.py` | No — self-contained | Yes |
  | `test_bo_smoke.py` | No — self-contained | Yes |
  | `test_crossbank_smoke.py` | No — self-contained | Yes (after C4's fix) |
  | `test_c4_output_contract.py` | Yes — reads `output/bo_rounds/all_rounds.json`, `output/cross_bank/bank_pairs.json` | Yes, after `python -m src.bo` and `python -m src.run_cross_bank` |
  | `test_c2_smoke.py` | Yes — writes `output/c2_smoke_test/` | No — blocked on A's `redteam/` package |

- Commands to run (full suite, in order):
  ```bash
  cd adversarial
  pip install scikit-optimize numpy persim

  python -m src.bo                          # writes output/bo_rounds/
  python -m src.run_cross_bank              # writes output/cross_bank/bank_pairs.json

  python tests/test_smoke.py
  python tests/test_bo_smoke.py
  python tests/test_crossbank_smoke.py
  python tests/test_c4_output_contract.py   # needs the two -m commands above first
  python tests/test_c2_smoke.py             # will fail until redteam/ exists — expected
  ```

## ADDENDUM 2 — Person C: redteam_adapter.py fallback added

Resolves the open question from the C4 addendum. `redteam_adapter.py`
now mirrors `blueteam_adapter.py`'s existing pattern exactly: the two
hard imports (`redteam.src.theta`, `redteam.src.main`) are wrapped in
`try/except ImportError`, with a new `BUILD_GRAPH_IS_REAL` flag (same
convention as `SCORE_BATCH_IS_REAL`) so callers can check which path
fired rather than being told silently.

- Files changed: `adversarial/src/redteam_adapter.py` only. No changes
  to `loop.py`, `bo.py`, `bo_real_space.py`, `run_real_rounds.py`, or
  `blueteam_adapter.py` — the fix is fully contained.
- Size: net +62 lines (45 → 107). ~12 lines for the try/except +
  `BUILD_GRAPH_IS_REAL` flag, ~35 lines for the new
  `_fallback_build_graph()` synthetic generator, ~15 lines of
  docstring/comments.
- What `_fallback_build_graph()` does: builds a TransactionGraph JSON
  matching the root README's schema exactly (ring-cycle edges +
  randomly-attached decoy edges), loosely driven by the tuned theta
  dict's `ring_size_lambda` / `decoy_density` / `fan_out_ratio` so the
  GP isn't optimizing against pure noise. Deliberately crude — no
  Hawkes timing, no decoy realism, no ring-closure logic — exists only
  so downstream code has something schema-shaped to run against. NOT a
  substitute for A's real generator; must never be mistaken for one.
- Verified for real: ran `test_c2_smoke.py` with neither `redteam/` nor
  `blueteam/` present. All 4 checks pass (`check_build_real_graph`,
  `check_score_batch`, `check_bottleneck`, `check_full_loop`) —
  `ALL C2 SMOKE TESTS PASSED`. Full 5-test suite now passes end to end
  with zero external dependencies.
- Still open, and still a team call, not resolved here: whether this
  fallback should ever be allowed to fire in the actual demo, or
  whether it's purely a development convenience so C's pipeline is
  runnable while waiting on A. Recommend `run_real_rounds.py` print the
  same kind of loud stderr warning it already prints for
  `SCORE_BATCH_IS_REAL=False` when `BUILD_GRAPH_IS_REAL=False` too —
  not added here since it's a few-line, low-risk follow-up rather than
  something blocking this fix.

## ADDENDUM 3 — Person C: redteam_adapter.py sys.path fix (real, confirmed root cause)

After A committed real files to `redteam/` (sibling of `adversarial/` and
`blueteam/` under the repo root, `redteam/src/__init__.py` present),
`BUILD_GRAPH_IS_REAL` was still `False` under the documented invocation
(`cd adversarial && python -m src.run_real_rounds`). This was NOT a
problem with A's files.

- Root cause, isolated and confirmed before touching anything: under
  `python -m src.run_real_rounds` run from inside `adversarial/`,
  Python puts only `adversarial/` on `sys.path` — never the repo root,
  where `redteam/` actually lives. So `from redteam.src.theta import
  ...` fails regardless of whether A's files are correct. Verified
  directly with a minimal, structurally-valid dummy `redteam` package
  before concluding this wasn't A's code at fault: the same import
  failed against the dummy package too, in the same environment where
  `blueteam_adapter.py`'s import of B's real module already succeeded.
- Why blueteam_adapter.py didn't have this problem: it already computes
  an absolute path from its own `__file__` and inserts it into
  `sys.path` before importing (see its own docstring). `redteam_adapter.py`
  never got the equivalent treatment — it did a bare `from redteam...
  import`, hoping the path was already right. It never was, under the
  documented invocation.
- Fix: added the same kind of `__file__`-relative `sys.path` insertion
  to `redteam_adapter.py`, computing the repo root
  (`os.path.dirname(__file__)/../..`) and inserting it before the
  import attempt. ~6 lines, contained entirely in this one file.
- Verified for real, both directions:
  - With a dummy-but-correct `redteam` package present:
    `BUILD_GRAPH_IS_REAL` now prints `True` under the exact documented
    command, and `run_real_rounds.py` runs end to end.
  - With `redteam/` removed entirely: `BUILD_GRAPH_IS_REAL` still
    correctly falls back to `False`, `test_c2_smoke.py` still passes
    via the fallback path — the fix doesn't break the fallback
    behavior added in Addendum 2.
- Action for the real repo: apply this same fix to the actual
  `redteam_adapter.py`, then rerun the `BUILD_GRAPH_IS_REAL` check —
  it should now flip to `True` with A's real files in place, assuming
  A's `Theta`/`build_graph`/`TransactionGraph.to_dict()` interfaces
  match what the adapter expects (not yet independently verified against
  A's actual `theta.py`/`main.py`/`schema.py`/`writer.py` — separate
  from this fix).