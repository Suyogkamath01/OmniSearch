import unittest
from pathlib import Path

from omnisearch.bootstrap import check_repository
from omnisearch.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_default_config_loads_with_image_group_split(self) -> None:
        config = load_config(ROOT / "configs" / "default.toml")
        self.assertEqual(config.dataset_id, "coco2017_val")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.split_group, "image_id")

    def test_repository_contract_is_complete(self) -> None:
        report = check_repository(ROOT)
        self.assertTrue(report.ok, report.missing_directories)


if __name__ == "__main__":
    unittest.main()
