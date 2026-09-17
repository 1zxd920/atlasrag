import tempfile
import unittest
from pathlib import Path

from atlasrag.index import HybridIndex


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = HybridIndex()
        self.index.add_text("refund.md", "# Retry\nA failed refund is retried once with an idempotency key.")
        self.index.add_text("security.md", "# Logs\nAudit logs are retained for 180 days.")

    def test_expected_source_is_ranked_first(self):
        hits = self.index.search("refund retry idempotency", limit=2)
        self.assertEqual("refund.md", hits[0].source)

    def test_index_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            self.index.save(path)
            loaded = HybridIndex.load(path)
            self.assertEqual("security.md", loaded.search("audit log retention")[0].source)


if __name__ == "__main__":
    unittest.main()

