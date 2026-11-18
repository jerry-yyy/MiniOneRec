"""Offline integration with real tokenizers and CPU tensors, without model weights.

The small WordLevel tokenizer is deliberately NOT a Qwen simulation. The separate
validate_hierarchical_integration CLI tests the pinned actual Qwen tokenizer.
"""

import json
from pathlib import Path
import tempfile
import unittest

from sid_generation.validate_hierarchical_integration import (
    evaluator_constraints, token_extender, validate_dataset_contract, validate_index,
)


def local_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "###": 2, " Response": 3, ":\n": 4, "\n": 5}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"###| Response|:\n|\n|."), behavior="isolated")
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")


class HierarchicalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "Demo.index.json"
        self.index = {
            "0": ["<leaf_000000>"],
            "1": ["<node_000001>", "<leaf_000001>"],
            "2": ["<node_000001>", "<node_000002>", "<leaf_000002>"],
            "3": ["<node_000001>", "<node_000002>", "<node_000003>", "<leaf_000003>"],
            "4": ["<node_000001>", "<node_000002>", "<node_000003>", "<leaf_000004>"],
        }
        self.path.write_text(json.dumps(self.index), encoding="utf-8")

    def test_token_extender_collects_namespaces_and_tokenizer_keeps_adjacency(self):
        extender = token_extender(self.path)
        tokens = extender.get_new_tokens()
        self.assertEqual(tokens, sorted({token for path in self.index.values() for token in path}))
        self.assertIs(extender.get_new_tokens(), tokens)
        tokenizer = local_tokenizer()
        self.assertEqual(tokenizer.add_tokens(tokens), len(tokens))
        for path in self.index.values():
            ids = [tokenizer.convert_tokens_to_ids(token) for token in path]
            self.assertEqual(tokenizer.encode("".join(path), add_special_tokens=False), ids)

    def test_mixed_length_catalog_actual_evaluator_and_cpu_masks(self):
        report = validate_index(self.path, local_tokenizer())
        self.assertEqual(report["catalog_items"], 5)
        self.assertEqual(report["sid_lengths"], [1, 2, 3, 4])
        self.assertEqual(report["atomic_tokens_checked"], 8)
        self.assertEqual(report["adjacent_sid_encodings_checked"], 5)
        self.assertTrue(report["invalid_prefix_forces_eos"])
        self.assertEqual(report["tensor_device"], "cpu")
        self.assertGreater(report["tensor_mask_calls"], 5)

    def test_actual_dataset_classes_preserve_all_paths(self):
        tokenizer = local_tokenizer()
        tokenizer.add_tokens(token_extender(self.path).get_new_tokens())
        report = validate_dataset_contract(self.index, tokenizer)
        self.assertEqual(report["mapped_items"], 5)
        self.assertGreaterEqual(report["actual_tokenized_pre_calls"], 4)

    def test_terminal_leaf_permits_newline_then_eos_only(self):
        tokenizer = local_tokenizer()
        tokenizer.add_tokens(token_extender(self.path).get_new_tokens())
        callback = evaluator_constraints(self.index, tokenizer)["prefix_allowed_tokens_fn_semantic"]
        for path in self.index.values():
            ids = tokenizer.encode("".join(path), add_special_tokens=False)
            newline = tokenizer.encode("\n", add_special_tokens=False)
            self.assertEqual(callback(0, ids), newline)
            self.assertEqual(callback(0, ids + newline), [tokenizer.eos_token_id])
        wrong = tokenizer.encode("<node_000001><leaf_000000>", add_special_tokens=False)
        self.assertEqual(callback(0, wrong), [])

    def test_rejects_sid_prefix_collision(self):
        self.index["5"] = ["<node_000001>"]
        self.path.write_text(json.dumps(self.index), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "prefix collision"):
            validate_index(self.path, local_tokenizer())

    def test_rejects_duplicate_final_sid(self):
        self.index["5"] = list(self.index["0"])
        self.path.write_text(json.dumps(self.index), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "duplicate SID"):
            validate_index(self.path, local_tokenizer())


if __name__ == "__main__":
    unittest.main()
