import unittest

from atlasrag.agent import AtlasAgent
from atlasrag.evaluation import evaluate
from atlasrag.index import HybridIndex
from atlasrag.models import EvaluationSample


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_computed(self):
        index = HybridIndex()
        index.add_text("policy.md", "# Retention\nAudit logs are kept for 180 days.")
        report = evaluate(
            AtlasAgent(index),
            [EvaluationSample("How long are audit logs kept?", "policy.md", ["180 days"])],
        )
        self.assertEqual(1.0, report["recall_at_k"])
        self.assertEqual(1.0, report["citation_rate"])


if __name__ == "__main__":
    unittest.main()
