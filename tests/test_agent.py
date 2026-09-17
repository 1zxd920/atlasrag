import unittest

from atlasrag.agent import AtlasAgent
from atlasrag.index import HybridIndex


class AgentTests(unittest.TestCase):
    def setUp(self):
        index = HybridIndex()
        index.add_text("runbook.md", "# Failed Refund\nAfter a failed refund, retry once. Then transfer the case to a human.")
        self.agent = AtlasAgent(index)

    def test_answer_has_citation(self):
        result = self.agent.query("What happens after a failed refund?")
        self.assertFalse(result.clarification_needed)
        self.assertTrue(result.citations)
        self.assertIn("[1]", result.answer)

    def test_ambiguous_query_requests_clarification(self):
        result = self.agent.query("What about it?")
        self.assertTrue(result.clarification_needed)

    def test_history_rewrites_follow_up(self):
        result = self.agent.query("What about it?", history=["Explain failed refunds"])
        self.assertIn("Follow-up question", result.rewritten_question)


if __name__ == "__main__":
    unittest.main()

