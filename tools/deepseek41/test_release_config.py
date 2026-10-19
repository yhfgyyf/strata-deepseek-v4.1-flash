import json
from pathlib import Path
import tempfile
import unittest

from tools.deepseek41.serve_experimental import load_config


class ReleaseConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        asset = self.root / "asset"
        asset.write_text("fixture")
        asset.chmod(0o700)
        self.cfg = {key: "asset" for key in ("engine", "model", "draft", "tokenizer", "integrity",
                    "profile", "heat", "costs", "draft_input_vocab", "draft_output_vocab")}
        self.path = self.root / "config.json"

    def load(self):
        self.path.write_text(json.dumps(self.cfg))
        return load_config(self.path)

    def test_paths_are_relative_to_config_and_vision_is_optional(self):
        self.cfg["log"] = "new-directory/engine.log"
        result = self.load()
        self.assertEqual(result["model"], str(self.root / "asset"))
        self.assertEqual(result["log"], str(self.root / "new-directory/engine.log"))
        self.assertFalse((self.root / "new-directory").exists())

    def test_public_bind_requires_authentication(self):
        self.cfg["host"] = "0.0.0.0"
        with self.assertRaisesRegex(ValueError, "requires api_key"):
            self.load()
        self.cfg["api_key"] = "local-test-key"
        self.assertEqual(self.load()["host"], "0.0.0.0")

    def test_invalid_port_rejected(self):
        for port in (True, 0, 65536, "30011"):
            with self.subTest(port=port):
                self.cfg["port"] = port
                with self.assertRaisesRegex(ValueError, "port"):
                    self.load()

    def test_missing_asset_rejected_before_loading_engine(self):
        self.cfg["draft"] = "missing.gguf"
        with self.assertRaisesRegex(ValueError, "draft file is missing"):
            self.load()


if __name__ == "__main__":
    unittest.main()
