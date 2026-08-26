# Adversarial Loop + Cross-Bank Layer — Person C

Bayesian-optimization loop that tunes Red Team's attack parameters `θ`
against what the Blue Team detector missed each round (Gaussian Process
surrogate + Expected Improvement acquisition), tracked via bottleneck
distance between rounds' persistence diagrams. Also owns a mocked
cross-bank commitment layer (salted hash commitments, explicitly simulated,
not real ZK/PSI).

Stages: C0 stub loop → C1 Bayesian optimization core → C2 real round loop
(wired to A + B) → C3 mocked cross-bank layer (independent, can be built any
time) → C4 finalize output schemas.

Output schemas (consumed by Dashboard) — see root `README.md` § Shared
interface contracts, contracts 5 and 6.

Log progress in `PROGRESS_C.md` in this folder.
