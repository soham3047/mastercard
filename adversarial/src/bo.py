"""
Stage C1 — Bayesian optimization core. (Modified in C2 — see note below.)

Wraps scikit-optimize's `Optimizer` (Gaussian Process surrogate +
Expected Improvement acquisition) around a theta search space, so each
round's theta is chosen by the GP instead of drawn randomly like C0's
stub_generator.

Objective handed to the GP: J(theta) = 1 - recall, per the master
prompt's spec, taken literally (NOT silently re-signed — see the open
question in C1's synopsis, STILL UNRESOLVED going into C2, about what
direction this actually optimizes).

C2 CHANGE (backward-compatible): `BayesOptimizer.__init__` now accepts
optional `search_space` and `vector_to_theta` parameters. When omitted,
both default to exactly C1's original stub-shaped values (SEARCH_SPACE /
_vector_to_theta below), so C1's existing tests and `python -m src.bo`
entry point are unaffected. C2 passes in `bo_real_space.REAL_SEARCH_SPACE`
/ `real_vector_to_theta_dict` (re-keyed against A's real Theta) instead of
adding a second optimizer class.
"""
from skopt import Optimizer
from skopt.space import Integer, Real

# Bounds intentionally mirror stub_generator.fake_generate_theta's ranges
# exactly, so BayesOptimizer.ask() produces theta dicts that are a
# drop-in generate_fn for loop.run_loop (same shape stub_generator emits).
# This remains the DEFAULT search space (backward-compat with C1) — C2's
# real run passes bo_real_space.REAL_SEARCH_SPACE instead.
SEARCH_SPACE = [
    Integer(3, 8, name="ring_size"),
    Real(0.1, 0.9, name="decoy_density"),
    Real(0.5, 5.0, name="hop_timing_lambda"),  # flattened; nested back into
                                                # hop_timing_params.lambda below
    Real(0.0, 1.0, name="fan_out_ratio"),
]


def _stub_vector_to_theta(x):
    ring_size, decoy_density, hop_timing_lambda, fan_out_ratio = x
    return {
        "ring_size": int(ring_size),
        "decoy_density": float(decoy_density),
        "hop_timing_params": {"lambda": float(hop_timing_lambda)},
        "fan_out_ratio": float(fan_out_ratio),
    }


class BayesOptimizer:
    """GP surrogate + Expected Improvement over a configurable theta space.

    ask()  -> theta dict, shaped by whatever `vector_to_theta` this
              instance was constructed with, so it's a drop-in generate_fn
              for loop.run_loop regardless of which theta shape is in use.
    tell(recall) -> feeds this round's observed recall back into the GP,
              using the theta from the most recent ask().

    Objective given to skopt: J(theta) = 1 - recall (skopt minimizes by
    default). NOTE — flagged, not silently resolved: minimizing
    (1 - recall) means the GP is pushing towards HIGHER recall, which is
    backwards from the master prompt's framing ("tunes Red Team's attack
    parameters specifically against what the detector missed" implies an
    attacker wants recall to go DOWN, i.e. evade detection). This class
    implements the formula exactly as specified rather than guessing the
    intended sign — carried over from C1, still open going into C2.
    """

    def __init__(self, seed=42, n_initial_points=3, search_space=None,
                 vector_to_theta=None):
        self.opt = Optimizer(
            dimensions=search_space if search_space is not None else SEARCH_SPACE,
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=n_initial_points,
            random_state=seed,
        )
        self._vector_to_theta = vector_to_theta or _stub_vector_to_theta
        self._last_x = None

    def ask(self, seed=None, round_num=None):
        """generate_fn-compatible signature (loop.run_loop calls
        generate_fn(seed=..., round_num=...)); both args accepted but
        ignored — the GP's own random_state governs exploration, not a
        per-round seed."""
        x = self.opt.ask()
        self._last_x = x
        return self._vector_to_theta(x)

    def tell(self, recall, x=None):
        """Feed this round's observed recall back into the GP. Uses the
        x from the most recent ask() unless one is passed explicitly."""
        if recall is None:
            # Surfaced by C2: B's real score_batch can return recall=None
            # for a round whose graph has no computable ground-truth
            # positives (recall is undefined, not 0 -- 0/0). Without this
            # guard, `1.0 - recall` below raised a cryptic
            # "TypeError: unsupported operand type(s) for -: 'float' and
            # 'NoneType'" with no hint at the actual cause. The known,
            # expected case is now handled by run_real_rounds.py's
            # score_fn (which substitutes a value and logs a warning
            # before ever calling tell()) -- this raise is a safety net
            # for any OTHER caller that reaches tell() without a guard,
            # so it fails loud and explains itself instead of crashing
            # confusingly.
            raise ValueError(
                "BayesOptimizer.tell() received recall=None. This usually "
                "means the score function's recall was undefined for this "
                "round's graph (e.g. no ground-truth positives to compute "
                "recall against). Callers must decide what to substitute "
                "before calling tell() -- see run_real_rounds.py's score_fn "
                "for the current handling -- rather than passing None "
                "through to the GP."
            )
        x = x if x is not None else self._last_x
        if x is None:
            raise RuntimeError("tell() called before any ask()")
        objective = 1.0 - recall
        self.opt.tell(x, objective)


def make_bo_generate_and_score_fns(seed=42, score_fn=None):
    """
    C1's original stub-wired convenience function — UNCHANGED in C2.
    Wires one BayesOptimizer (default stub search space) around a score
    function into a (generate_fn, score_fn) pair for loop.run_loop.
    C2's real run uses run_real_rounds.make_real_generate_and_score_fns
    instead, which passes the real search space explicitly.
    """
    from .stub_scorer import fake_score
    score_fn = score_fn or fake_score
    bo = BayesOptimizer(seed=seed)

    def generate_fn(seed=None, round_num=None):
        return bo.ask(seed=seed, round_num=round_num)

    def wrapped_score_fn(theta, seed=None):
        scored = score_fn(theta, seed=seed)
        bo.tell(scored["recall"])
        return scored

    return generate_fn, wrapped_score_fn, bo


if __name__ == "__main__":
    from .loop import run_loop

    generate_fn, score_fn, bo = make_bo_generate_and_score_fns(seed=42)
    results = run_loop(
        n_rounds=8,
        out_dir="output/bo_rounds",
        seed=42,
        generate_fn=generate_fn,
        score_fn=score_fn,
    )
    for r in results:
        print(
            f"Round {r['round']}: recall={r['recall']}, "
            f"theta={r['theta_used']}"
        )
