"""트렌드 추천어는 저장된 추천 주제 파일에서 온다. 파일이 틀리면 근거 없는 일반 예시로 구분한다."""

import json
import tempfile
import unittest
from pathlib import Path

from services import keyword_service


class TrendTopicsTest(unittest.TestCase):
    def test_shipped_file_is_valid(self) -> None:
        # 배포되는 파일: 6개 이하, 주제마다 근거 논문 2편 이상, 응답 계약을 만족
        data = json.loads(keyword_service._TOPICS_FILE.read_text(encoding="utf-8"))
        self.assertLessEqual(len(data["topics"]), 6)
        for topic in data["topics"]:
            self.assertGreaterEqual(len(topic["evidence"]), 2, topic["keyword"])
            for ev in topic["evidence"]:
                self.assertTrue(ev["url"].startswith("https://huggingface.co/papers/"))
        result = keyword_service.get_trend_topics()
        self.assertIsNotNone(result.basis)
        self.assertEqual(result.keywords, [t["keyword"] for t in data["topics"]])

    def test_missing_or_broken_file_falls_back_without_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "t.json"
            broken.write_text('{"topics": []}', encoding="utf-8")
            good = json.loads(keyword_service._TOPICS_FILE.read_text(encoding="utf-8"))
            bad_count = Path(tmp) / "bad_count.json"
            bad_count.write_text(json.dumps({**good, "paper_count": "oops"}), encoding="utf-8")
            null_keyword = Path(tmp) / "null_keyword.json"
            null_keyword.write_text(json.dumps({**good, "topics": [{"keyword": None}]}), encoding="utf-8")
            for path in (Path(tmp) / "missing.json", broken, bad_count, null_keyword):
                result = keyword_service.get_trend_topics(path)
                self.assertIsNone(result.basis, path.name)
                self.assertEqual(result.keywords, keyword_service._FALLBACK_KEYWORDS, path.name)


if __name__ == "__main__":
    unittest.main()
