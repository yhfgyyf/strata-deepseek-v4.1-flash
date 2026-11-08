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
                        variant="GSQ-RCO-IQ2_XS", model=self.files[0].name, vision=False,
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

    def test_image_start_requires_verified_bias_and_selected_shard(self):
        self.cfg["vision"] = True
        for variant, fourth in (("GSQ-RCO-IQ2_XS", hash_proof.GSQ_RCO_SHA256[3]),
                                ("GSQ-RCO-IQ2_XS-Uncensored", hash_proof.GSQ_UNCENSORED_SHARD4_SHA256)):
            with self.subTest(variant=variant):
                self.cfg.update(variant=variant, integrity="original.proof")
                original = self.root / "original.proof"
                identity = " ".join(map(str, hash_proof.identity(self.files[3])))
                source = f"STRATA_DSV41_SHA256_V1\nmain.4 {fourth} {identity}\n"
                original.write_text(source)
                bias = self.root / startup_assets.GSQ_VISION_BIAS
                bias.unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError, "image router bias is missing"):
                    self.load()
                data = bytes(40 * 384 * 4)
                bias.write_bytes(data)
                cfg = self.load()
                self.assertTrue(cfg["vision"])
                with patch.object(startup_assets, "TOKENIZER_SHA256", hashlib.sha256(b"fixture").hexdigest()):
                    with self.assertRaisesRegex(ValueError, "bias SHA-256"):
                        startup_assets.prepare(cfg)
                    digest = hashlib.sha256(data).hexdigest()
                    with patch.object(startup_assets, "GSQ_VISION_BIAS_SHA256", digest):
                        proof = Path(startup_assets.prepare(cfg))
                        self.assertEqual(proof, Path(startup_assets.prepare(cfg)))
                        self.assertIn("vision_bias " + digest + " ", proof.read_text())
                        self.assertEqual(original.read_text(), source)
                        original.write_text(source.replace(fourth, "0" * 64))
                        with self.assertRaisesRegex(ValueError, "fourth-shard"):
                            startup_assets.prepare(cfg)

    def test_legacy_text_configs_check_variant_without_visual_bias(self):
        self.cfg.pop("vision")
        original = self.root / "original.proof"
        self.cfg["integrity"] = original.name
        identity = " ".join(map(str, hash_proof.identity(self.files[3])))
        with patch.object(startup_assets, "TOKENIZER_SHA256", hashlib.sha256(b"fixture").hexdigest()):
            for variant, digest in ((startup_assets.GSQ_VARIANT, hash_proof.GSQ_RCO_SHA256[3]),
                                    (startup_assets.GSQ_UNCENSORED_VARIANT, hash_proof.GSQ_UNCENSORED_SHARD4_SHA256)):
                with self.subTest(variant=variant):
                    self.cfg["variant"] = variant
                    original.write_text(f"STRATA_DSV41_SHA256_V1\nmain.4 {digest} {identity}\n")
                    cfg = self.load()
                    self.assertFalse(cfg["vision"])
                    self.assertNotIn("_gsq_vision_bias", cfg)
                    self.assertEqual(Path(startup_assets.prepare(cfg)), original)
                    other = (hash_proof.GSQ_RCO_SHA256[3] if variant == startup_assets.GSQ_UNCENSORED_VARIANT
                             else hash_proof.GSQ_UNCENSORED_SHARD4_SHA256)
                    original.write_text(f"STRATA_DSV41_SHA256_V1\nmain.4 {other} {identity}\n")
                    with self.assertRaisesRegex(ValueError, "fourth-shard"):
                        startup_assets.prepare(cfg)

    def test_uncensored_auto_proof_uses_original_names_and_exact_fourth_digest(self):
        self.cfg["variant"] = "GSQ-RCO-IQ2_XS-Uncensored"
        cfg = self.load()
        self.assertEqual(cfg["hot_bytes"], "auto")
        self.assertFalse(cfg["vision"])
        digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.files + [self.engram]]
        fixture_digest = hashlib.sha256(b"fixture").hexdigest()
        with patch.object(hash_proof, "GSQ_RCO_SHA256", digests[:3] + ["0" * 64, digests[4]]), \
                patch.object(hash_proof, "GSQ_UNCENSORED_SHARD4_SHA256", digests[3]), \
                patch.object(hash_proof, "SIDECARS_1017", {"draft": fixture_digest, "vision": fixture_digest}), \
                patch.object(startup_assets, "TOKENIZER_SHA256", fixture_digest):
            startup_assets.prepare(cfg)
            proof = Path(cfg["integrity"])
            before = proof.read_bytes()
            self.assertIn("main.4 " + digests[3], proof.read_text())
            with patch.object(hash_proof, "verify", side_effect=AssertionError("unnecessary full rehash")):
                startup_assets.prepare(cfg)
            original = dict(cfg, variant="GSQ-RCO-IQ2_XS")
            with self.assertRaisesRegex(ValueError, "SHA-256 failed"):
                startup_assets.prepare(original)
            self.assertEqual(proof.read_bytes(), before)

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
        self.assertEqual(hash_proof.GSQ_RCO_SHA256, native[:5])
        self.assertEqual(hash_proof.GSQ_PLE_SHA256, native[5:11])
        self.assertEqual([hash_proof.GSQ_UNCENSORED_SHARD4_SHA256, startup_assets.GSQ_VISION_BIAS_SHA256], native[11:])

    def test_uncensored_proof_uses_exact_fourth_shard_and_preserves_original(self):
        paths = {"main": self.files[0]}
        original = hash_proof.approved_files(paths, main_gsq_rco=True)
        candidate = hash_proof.approved_files(paths, main_gsq_rco_uncensored=True)
        self.assertEqual(original[:3] + original[4:], candidate[:3] + candidate[4:])
        self.assertNotEqual(original[3][2], candidate[3][2])
        digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.files + [self.engram]]
        with patch.object(hash_proof, "GSQ_RCO_SHA256", digests[:3] + ["0" * 64, digests[4]]), \
                patch.object(hash_proof, "GSQ_UNCENSORED_SHARD4_SHA256", digests[3]):
            proof = self.root / "uncensored.proof"
            hash_proof.verify(paths, proof, main_gsq_rco_uncensored=True)
            before = proof.read_bytes()
            with self.assertRaisesRegex(ValueError, "SHA-256 failed"):
                hash_proof.verify(paths, proof, main_gsq_rco=True)
            self.assertEqual(proof.read_bytes(), before)
            self.files[3].write_bytes(b"changed fourth shard")
            with self.assertRaisesRegex(ValueError, "SHA-256 failed"):
                hash_proof.verify(paths, proof, main_gsq_rco_uncensored=True)
            self.assertEqual(proof.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "choose one"):
            hash_proof.approved_files(paths, main_q2_k_xl=True, main_gsq_rco_uncensored=True)


if __name__ == "__main__":
    unittest.main()
