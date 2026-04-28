from qwen_think.router import ComplexityRouter, RuleBasedClassifier
from qwen_think.types import Complexity, ThinkingMode


class TestRuleBasedClassifier:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_simple_queries(self):
        for q in ["yes", "What is Python?", "translate hello"]:
            c = self.clf.classify(q)
            assert c == Complexity.SIMPLE, f"{q} → {c}"

    def test_moderate_queries(self):
        for q in ["How do I set up a virtual environment for this project?"]:
            c = self.clf.classify(q)
            assert c == Complexity.MODERATE, f"{q} → {c}"

    def test_complex_queries(self):
        for q in [
            "Refactor this module to use dependency injection",
            "Debug and rewrite the authentication middleware that is leaking sessions step by step",
        ]:
            c = self.clf.classify(q)
            assert c == Complexity.COMPLEX, f"{q} → {c}"

    def test_code_boosts_score(self):
        q = "```python\ndef foo():\n    return bar\n```\nfix this function"
        c = self.clf.classify(q)
        assert c == Complexity.COMPLEX

    def test_word_count_boost(self):
        short = "hello"
        long_q = "word " * 55
        assert self.clf.classify(short) != Complexity.COMPLEX
        c = self.clf.classify(long_q + "implement something")
        assert c == Complexity.COMPLEX

    def test_context_adds_score(self):
        q = "implement this"
        without = self.clf.classify(q, context=None)
        with_ctx = self.clf.classify(q, context=["a", "b", "c", "d", "e"])
        # Context should only push the score up, never down
        assert with_ctx.value >= without.value or with_ctx == without


class TestComplexityRouter:
    def setup_method(self):
        self.router = ComplexityRouter()

    def test_simple_routes_to_no_think(self):
        d = self.router.route("What is 2+2?")
        assert d.complexity == Complexity.SIMPLE
        assert d.mode == ThinkingMode.NO_THINK
        assert d.preserve_thinking is False

    def test_complex_routes_to_think_with_preserve(self):
        d = self.router.route("Refactor this entire module for async")
        assert d.complexity == Complexity.COMPLEX
        assert d.mode == ThinkingMode.THINK
        assert d.preserve_thinking is True
        assert d.sampling.temperature == 1.0

    def test_force_thinking_overrides(self):
        router = ComplexityRouter(force_thinking=True)
        d = router.route("yes")
        assert d.mode == ThinkingMode.THINK
        assert d.preserve_thinking is True

    def test_override_mode(self):
        d = self.router.route("Refactor this", override_mode=ThinkingMode.NO_THINK)
        assert d.mode == ThinkingMode.NO_THINK
        assert d.preserve_thinking is False

    def test_sampling_matches_mode(self):
        think = self.router.route("Implement a REST API with auth")
        assert think.sampling.temperature == 1.0
        no_think = self.router.route("yes", override_mode=ThinkingMode.NO_THINK)
        assert no_think.sampling.temperature == 0.7

    def test_confidence_is_always_1(self):
        d = self.router.route("Implement something complex step by step")
        assert d.confidence == 1.0

    def test_reasoning_includes_classification(self):
        d = self.router.route("What is 2+2?")
        assert "simple" in d.reasoning.lower()
