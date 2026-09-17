import unittest

from atlasrag.text import chunk_document, tokenize


class TextTests(unittest.TestCase):
    def test_parent_child_relationships(self):
        parents, children = chunk_document(
            "guide.md",
            "# First\n" + "alpha " * 100 + "\n# Second\nbeta policy",
            child_size=100,
            child_overlap=10,
        )
        self.assertEqual(2, len(parents))
        self.assertGreater(len(children), 2)
        parent_ids = {item.id for item in parents}
        self.assertTrue(all(item.parent_id in parent_ids for item in children))

    def test_tokenizer_handles_chinese_and_english(self):
        tokens = tokenize("Refund 退款流程")
        self.assertIn("refund", tokens)
        self.assertIn("退款", tokens)


if __name__ == "__main__":
    unittest.main()

