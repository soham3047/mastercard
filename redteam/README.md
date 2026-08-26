# Red Team / Attack Generation — Person A

Generates directed, time-respecting transaction graphs for both attack
shapes (ring/layering and fan-out/mule), with GenAI identity content and
Hawkes-process decoy noise so fraud isn't distinguishable transaction-by-transaction.

Stages: A0 scaffold → A1 ring generator → A2 fan-out generator → A3 decoy
realism → A4 calibration to public aggregate stats → A5 LLM identity content
→ A6 expose `θ` config for the adversarial loop.

Output schema (consumed by Blue Team and Dashboard) — see root `README.md`
§ Shared interface contracts, contract 1.

Log progress in `PROGRESS_A.md` in this folder.
