# companion/models.py — pluggable forecasters. Numeric baselines ignore text (arm A=B=C by
# construction — that's the point: a purely numeric base is text-blind, so it establishes the
# no-text floor). A TEXT-CONDITIONED forecaster needs a semantic base model (Moirai-MoE / Time-MMD
# / the team model) + GPU — left as a documented stub so the pipeline is complete and the real
# base plugs in later without touching split/arms/eval.
from statistics import mean


class Forecaster:
    uses_text = False
    name = "base"

    def predict(self, history, text, H):
        raise NotImplementedError


class LastValue(Forecaster):
    name = "last_value"

    def predict(self, history, text, H):
        return [history[-1]] * H


class LinearTrend(Forecaster):
    name = "linear_trend"

    def predict(self, history, text, H):
        n = len(history)
        xs = list(range(n))
        mx, my = mean(xs), mean(history)
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        b = sum((xs[i] - mx) * (history[i] - my) for i in range(n)) / den
        a = my - b * mx
        return [a + b * (n + h) for h in range(H)]


class TextConditionedStub(Forecaster):
    """Placeholder for the real text-reading base model. Raises so run.py marks arms B/C 'pending'
    rather than silently faking a gain (宁缺毋滥). Swap in a wrapper over Moirai-MoE/Time-MMD that
    consumes `text` to produce a GPU-backed forecast."""
    uses_text = True
    name = "text_conditioned(base_model)"

    def predict(self, history, text, H):
        raise NotImplementedError("needs a semantic base model + GPU — see companion/README.md")


class LLMTimeForecaster(Forecaster):
    """Zero-GPU text-reading arm: LLMTime-style forecasting via the team's vLLM (core.compress).
    Feeds the normalized history + the conditioning text and parses the continuation. Predictions
    are cached (keyed on text+history+H) so re-runs are offline-deterministic despite the API.
    Not a substitute for a real multimodal TS base model — it's the cheap first signal on whether
    the text helps, and it exercises the exact interface a GPU base model will plug into."""
    uses_text = True
    name = "llmtime(team_vllm)"

    def __init__(self, cache_path):
        import json, os
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self._dirty = False

    def _key(self, history, text, H):
        import hashlib
        return hashlib.sha1(f"{text}|{[round(v, 3) for v in history]}|{H}".encode()).hexdigest()[:16]

    def predict(self, history, text, H):
        import re
        k = self._key(history, text, H)
        if k in self.cache:
            return self.cache[k]
        from core.compress import complete
        sys_p = ("You are a forecasting engine. Given optional real-world context and a series of "
                 "recent normalized numeric values, predict the next N values. Respond with ONLY N "
                 "numbers separated by commas — no words, no explanation.")
        hist_s = ", ".join(f"{v:.2f}" for v in history)
        user_p = (f"Context: {text or '(none)'}\n"
                  f"Recent normalized values ({len(history)}): {hist_s}\n"
                  f"Predict the next {H} values (comma-separated, exactly {H} numbers):")
        try:
            out = complete(sys_p, user_p, max_tokens=16 * H + 48, temperature=0.0)
            nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", out)][:H]
        except Exception:
            nums = []
        if len(nums) < H:                                   # robustness: pad short/garbled output
            nums = (nums + [history[-1]] * H)[:H]
        self.cache[k] = nums
        self._dirty = True
        return nums

    def save(self):
        import json
        if self._dirty:
            json.dump(self.cache, open(self.cache_path, "w"))


NUMERIC_BASELINES = [LastValue(), LinearTrend()]
