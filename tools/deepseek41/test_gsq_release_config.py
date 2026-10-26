import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from tools.deepseek41 import hash_proof, startup_assets
from tools.deepseek41.serve_experimental import load_config


class GsqReleaseConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        asset = self.root / "asset"
        asset.write_bytes(b"fixture")
        asset.chmod(0o700)
        self.files = []
        for i in range(1, 5):
            path = self.root / f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-{i:05d}-of-00004.gguf"
            path.write_bytes(b"GSQ shard " + bytes([i]))
            self.files.append(path)
        self.engram = self.root / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
        self.engram.write_bytes(b"external Engram")
        self.cfg = dict(engine="asset", draft="asset", tokenizer="asset", mmproj="asset",
                        variant="GSQ-RCO-IQ2_XS", model=self.files[0].name,
                        runtime_dir="runtime/gsq")

    def load(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.cfg))
        return load_config(path)

    def test_four_shards_and_external_engram_use_auto_budget_and_text_api(self):
        cfg = self.load()
        self.assertEqual(cfg["hot_bytes"], "auto")
        self.assertFalse(cfg["vision"])
        self.assertEqual(cfg["mmproj"], str(self.root / "asset"))
        self.assertTrue(cfg["_auto_profile"])
        self.assertTrue(cfg["_auto_integrity"])
        self.assertFalse((self.root / "runtime").exists())

    def test_missing_engram_and_main_shard_fail_before_start(self):
        self.engram.unlink()
        with self.assertRaisesRegex(ValueError, "Engram"):
            self.load()
        self.engram.write_bytes(b"external Engram")
        self.files[-1].unlink()
        with self.assertRaisesRegex(ValueError, "model shard is missing"):
            self.load()

    def test_gsq_does_not_expose_an_unsupported_image_api(self):
        self.cfg["vision"] = True
        with self.assertRaisesRegex(ValueError, "GSQ.*text-only"):
            self.load()

    def test_proof_hashes_all_five_files_and_rejects_changed_engram(self):
        cfg = self.load()
        files = self.files + [self.engram]
        before = {p: p.read_bytes() for p in files}
        digests = [hashlib.sha256(before[p]).hexdigest() for p in files]
        with patch.object(hash_proof, "GSQ_RCO_SHA256", digests), \
                patch.object(hash_proof, "SIDECARS_1017", {"draft": hashlib.sha256(b"fixture").hexdigest(),
                                                         "vision": hashlib.sha256(b"fixture").hexdigest()}), \
                patch.object(startup_assets, "TOKENIZER_SHA256", hashlib.sha256(b"fixture").hexdigest()):
            startup_assets.prepare(cfg)
            paths = {"main": files[0], "draft": self.root / "asset", "vision": self.root / "asset"}
            approved = hash_proof.approved_files(paths, sidecars_1017=True, main_gsq_rco=True)
            self.assertEqual([v[0] for v in approved],
                             ["main.1", "main.2", "main.3", "main.4", "engram", "draft", "vision"])
            proof = Path(cfg["integrity"])
            self.assertTrue(startup_assets.proof_matches(proof, approved))
            with patch.object(hash_proof, "verify", side_effect=AssertionError("unnecessary full rehash")):
                startup_assets.prepare(cfg)
            self.assertEqual(before, {p: p.read_bytes() for p in files})
            old_proof = proof.read_bytes()
            self.engram.write_bytes(b"changed external Engram")
            with self.assertRaisesRegex(ValueError, "SHA-256 failed"):
                startup_assets.prepare(cfg)
            self.assertEqual(proof.read_bytes(), old_proof)

    def test_python_approved_digests_match_native_loader(self):
        root = Path(__file__).resolve().parents[2]
        header = root / "include/strata/models/deepseek41/gsq_rco_1017_digests.hpp"
        native = re.findall(r'"([0-9a-f]{64})"', header.read_text())
        self.assertEqual(hash_proof.GSQ_RCO_SHA256, native)


if __name__ == "__main__":
    unittest.main()
