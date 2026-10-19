"""The supplied tokenizer.json with Strata's existing byte BPE implementation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import regex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from strata_tokenizer import Tokenizer


class SplitSequence:
    def __init__(self, patterns):
        self.patterns = [regex.compile(pattern) for pattern in patterns]

    def findall(self, text):
        pieces = [text]
        for pattern in self.patterns:
            output = []
            for piece in pieces:
                start = 0
                for match in pattern.finditer(piece):
                    if match.start() > start:
                        output.append(piece[start:match.start()])
                    output.append(match.group())
                    start = match.end()
                if start < len(piece):
                    output.append(piece[start:])
            pieces = output
        return pieces


class DeepSeekTokenizer(Tokenizer):
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        model = data["model"]
        if model["type"] != "BPE" or data.get("normalizer") != {"type": "Sequence", "normalizers": []}:
            raise ValueError("unsupported DeepSeek tokenizer model or normalizer")
        vocab = model["vocab"]
        added = data["added_tokens"]
        n = max([*vocab.values(), *(t["id"] for t in added)]) + 1
        if n != 129280:
            raise ValueError("DeepSeek vocabulary must have 129280 IDs")
        tokens, types = [None] * n, [1] * n
        for token, index in vocab.items():
            tokens[index] = token
        for entry in added:
            if entry["single_word"] or entry["lstrip"] or entry["rstrip"]:
                raise ValueError("unsupported added-token matching")
            if tokens[entry["id"]] not in (None, entry["content"]):
                raise ValueError("added token changes an existing token ID")
            tokens[entry["id"]] = entry["content"]
            types[entry["id"]] = 3 if entry["special"] else 4
        if any(token is None for token in tokens):
            raise ValueError("DeepSeek tokenizer has missing token IDs")
        super().__init__(tokens, model["merges"], types, pre="deepseek-v4.1")
        pre = data["pre_tokenizer"]
        if pre["type"] != "Sequence":
            raise ValueError("DeepSeek tokenizer needs sequential pre-tokenizers")
        patterns = []
        for step in pre["pretokenizers"]:
            if step["type"] == "Split" and step["behavior"] == "Isolated" and not step["invert"]:
                patterns.append(step["pattern"]["Regex"])
            elif step != {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}:
                raise ValueError("unsupported DeepSeek pre-tokenizer step")
        self._re = SplitSequence(patterns)
        self._added_ids = set(self.special_tokens.values())
        self.eos_id = self.ids["<｜end▁of▁sentence｜>"]

    def token_bytes(self, i):
        if i in self._added_ids:
            return self.tokens[i].encode("utf-8")
        return super().token_bytes(i)
