import unittest
from pathlib import Path

from omnisearch.preprocessing import IdentityImagePreprocessor


class PreprocessingInterfaceTests(unittest.TestCase):
    def test_identity_image_preprocessor_does_not_change_path(self) -> None:
        path = Path("data/raw/example.jpg")
        self.assertEqual(IdentityImagePreprocessor()(path), path)
