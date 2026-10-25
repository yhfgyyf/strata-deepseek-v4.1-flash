import json
from pathlib import Path
import tempfile
import unittest
import struct
import hashlib
from unittest.mock import patch

from tools.deepseek41.serve_experimental import load_config
from tools.deepseek41.startup_assets import VARIANTS, prepare, proof_matches, starter_profile
from tools.deepseek41 import hash_proof
from tools.deepseek41 import startup_assets


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

    def test_explicit_hot_budget_is_checked(self):
        for value in (True, 0, -1, "123", 1 << 63):
            self.cfg["hot_bytes"] = value
            with self.assertRaisesRegex(ValueError, "hot_bytes"):
                self.load()
        self.cfg["hot_bytes"] = 64 << 30
        self.assertEqual(self.load()["hot_bytes"], 64 << 30)

    def test_split_model_symlink_directory_is_preserved(self):
        link = self.root / "first-shard.gguf"
        link.symlink_to(self.root / "asset")
        self.cfg["model"] = link.name
        self.assertEqual(self.load()["model"], str(link))

    def test_prefix_cache_budget(self):
        for value in (True, -1, 4097, "1024", 1.5):
            self.cfg["prefix_cache_mib"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prefix_cache_mib"):
                self.load()
        for value in (0, 1024, 4096):
            self.cfg["prefix_cache_mib"] = value
            self.assertEqual(self.load()["prefix_cache_mib"], value)

    def variant_config(self, variant):
        self.cfg = {key: "asset" for key in ("engine", "draft", "tokenizer")}
        self.cfg["variant"] = variant
        for i in range(1, 8):
            name = f"DeepSeek-V4.1-Flash-1017-{variant}-{i:05d}-of-00007.gguf"
            (self.root / name).write_bytes(b"test shard " + bytes([i]))
            if i == 1:
                self.cfg["model"] = name

    def test_both_variants_need_only_original_model_assets(self):
        for variant, hot_bytes in VARIANTS.items():
            with self.subTest(variant=variant):
                self.variant_config(variant)
                cfg = self.load()
                self.assertEqual(cfg["hot_bytes"], hot_bytes)
                self.assertTrue(cfg["_auto_profile"])
                self.assertTrue(cfg["_auto_integrity"])
                self.assertFalse((self.root / "runtime").exists())
                self.assertTrue(Path(cfg["draft_input_vocab"]).is_file())
                self.assertTrue(Path(cfg["draft_output_vocab"]).is_file())

    def test_incomplete_shards_and_wrong_variant_fail_before_start(self):
        self.variant_config("UD-Q2_K_XL")
        last = self.root / self.cfg["model"].replace("00001", "00007")
        last.unlink()
        with self.assertRaisesRegex(ValueError, "model shard is missing"):
            self.load()
        self.cfg["variant"] = "UD-IQ3_XXS"
        with self.assertRaisesRegex(ValueError, "first of the seven"):
            self.load()

    def test_calibration_requires_a_pair_and_preserves_explicit_paths(self):
        self.variant_config("UD-Q2_K_XL")
        self.cfg["profile"] = "asset"
        with self.assertRaisesRegex(ValueError, "both profile and heat"):
            self.load()
        self.cfg.update(heat="asset", integrity="asset")
        cfg = self.load()
        self.assertFalse(cfg["_auto_profile"])
        self.assertFalse(cfg["_auto_integrity"])
        self.assertEqual(cfg["profile"], str(self.root / "asset"))

    def test_profile_has_every_pair_once_and_correct_inverse(self):
        blob = starter_profile()
        self.assertEqual(blob[:4], b"STRP")
        self.assertEqual(struct.unpack_from("<5I", blob, 4), (1, 40, 384, 15360, 15360))
        pairs = list(struct.iter_unpack("<HH", blob[24:24 + 4 * 15360]))
        inverse = struct.unpack_from("<15360i", blob, 24 + 4 * 15360)
        self.assertEqual(set(pairs), {(l, e) for l in range(40) for e in range(384)})
        for slot, (layer, expert) in enumerate(pairs):
            self.assertEqual(slot, inverse[layer * 384 + expert])

    def test_wrong_tokenizer_is_rejected_before_metadata_or_model_reads(self):
        self.variant_config("UD-Q2_K_XL")
        with self.assertRaisesRegex(ValueError, "tokenizer SHA-256 mismatch"):
            prepare(self.load())
        self.assertFalse((self.root / "runtime").exists())

    def test_prepare_hashes_original_files_and_reuses_only_same_identity(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.variant_config(variant)
                self.cfg["runtime_dir"] = "runtime/" + variant
                cfg = self.load()
                files = [self.root / cfg["model"].split("/")[-1].replace("00001", f"{i:05d}") for i in range(1, 8)]
                before = {p: p.read_bytes() for p in files}
                digests = [hashlib.sha256(before[p]).hexdigest() for p in files]
                sidecars = {"draft": hashlib.sha256(b"fixture").hexdigest()}
                with patch.object(hash_proof, "IQ3XXS_SHA256", digests), \
                        patch.object(hash_proof, "Q2K_XL_SHA256", digests), \
                        patch.object(hash_proof, "SIDECARS_1017", sidecars), \
                        patch.object(startup_assets, "TOKENIZER_SHA256", hashlib.sha256(b"fixture").hexdigest()):
                    prepare(cfg)
                    approved = hash_proof.approved_files({"main": files[0], "draft": self.root / "asset"},
                                                         variant == "UD-Q2_K_XL", True)
                    self.assertTrue(proof_matches(Path(cfg["integrity"]), approved))
                    with patch.object(hash_proof, "verify", side_effect=AssertionError("unnecessary full rehash")):
                        prepare(cfg)
                    self.assertEqual(before, {p: p.read_bytes() for p in files})
                    files[3].write_bytes(b"changed model")
                    self.assertFalse(proof_matches(Path(cfg["integrity"]), approved))
                    with self.assertRaisesRegex(ValueError, "SHA-256 failed"):
                        prepare(cfg)

    def test_bundled_vocab_maps_are_sorted_and_output_is_an_input_subset(self):
        self.variant_config("UD-Q2_K_XL")
        cfg = self.load()
        maps = []
        for key, count in (("draft_input_vocab", 114801), ("draft_output_vocab", 90000)):
            ids = [v[0] for v in struct.iter_unpack("<i", Path(cfg[key]).read_bytes())]
            self.assertEqual(len(ids), count)
            self.assertEqual(ids, sorted(set(ids)))
            self.assertTrue({0, 1, 2, 128799}.issubset(ids))
            self.assertTrue(all(0 <= v < 129280 for v in ids))
            maps.append(set(ids))
        self.assertTrue(maps[1].issubset(maps[0]))


if __name__ == "__main__":
    unittest.main()
