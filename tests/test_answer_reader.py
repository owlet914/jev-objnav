import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jev_obj"))
from llm.answer_reader.answer_reader import read_answer


class AnswerReaderTests(unittest.TestCase):
    def test_neutral_prior_does_not_read_or_create_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "answers.txt"
            cache.write_text("mug: ['cup', 0.3, 'kitchen']\n", encoding="utf-8")
            result = read_answer(str(cache), str(Path(tmp) / "responses.txt"), "mug", SimpleNamespace(llm_client="none"))
            self.assertEqual(result, ([], "everywhere", 0.5))
            self.assertEqual(cache.read_text(encoding="utf-8"), "mug: ['cup', 0.3, 'kitchen']\n")

    def test_cached_answer_parsed_without_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "answers.txt"
            cache.write_text("mug: ['cup', 0.45, 'kitchen']\n", encoding="utf-8")
            result = read_answer(str(cache), str(Path(tmp) / "responses.txt"), "mug", SimpleNamespace(llm_client="ollama"))
            self.assertEqual(result, (["cup"], "kitchen", 0.45))
            cache.write_text("mug: __import__('os').system('exit 7')\n", encoding="utf-8")
            with self.assertRaises((ValueError, SyntaxError)):
                read_answer(str(cache), str(Path(tmp) / "responses.txt"), "mug", SimpleNamespace(llm_client="ollama"))


if __name__ == "__main__":
    unittest.main()
