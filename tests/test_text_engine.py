"""The LTX prompt presentation (BOS, left padding to 1024, every layer, projection) and the Gemma tokenizers."""

import os
import unittest
from dataclasses import dataclass
from pathlib import Path

from forge_env import FORGE

if FORGE:
    import torch
    from backend.operations import using_forge_operations
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from forge_ltx.native import gemma4, projection, text_engine

# the official Gemma 3 tokenizer files of Lightricks/LTX-2 (tokenizer/tokenizer.model and tokenizer.json), downloaded
# outside the repository
HF_DIR = Path(os.environ.get("LTX_HF_DIR", "D:/ltx-forge-dev/hf"))
GEMMA3_FILES = HF_DIR / "LTX-2" / "tokenizer"

PROMPTS = [
    "A golden retriever running through a sunny meadow, cinematic lighting",
    "  Two   spaces and\nnewlines\t tabs ",
    "Ação! Câmera lenta - 4K HDR, 50 FPS. 日本語のテキスト 🎬 x²",
    "The camera slowly dollies out as wind moves through the grass.",
    "",
]


def build(factory):
    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        module = factory()
    torch.manual_seed(0)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.05 + (1.0 if param.ndim == 1 else 0.0))
    return module


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @dataclass
        class Tiny(gemma4.Gemma4_12BConfig):
            vocab_size: int = 16
            hidden_size: int = 32
            intermediate_size: int = 64
            num_hidden_layers: int = 6
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim: int = 8
            global_head_dim: int = 16
            sliding_attention = [8, 8, 8, 8, 8, False]

        vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "a": 4, "red": 5, "fox": 6, "runs": 7}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        cls.model = build(lambda: gemma4.Gemma4_12B(config=Tiny()))
        spec = {"kind": "dual_linear", "in_dim": 7 * 32, "video_dim": 12, "audio_dim": 6, "video_bias": True, "audio_bias": True}
        cls.projection = build(lambda: projection.build(spec))
        cls.engine = text_engine.LTXTextEngine(cls.model, tokenizer, cls.projection)

    def test_prompt_is_bos_and_text_left_padded_to_1024(self):
        ids = [t for t, _ in self.engine.tokens("a red fox runs")[0]]
        self.assertEqual(len(ids), text_engine.MIN_LENGTH)
        self.assertEqual(ids[-5:], [2, 4, 5, 6, 7])
        self.assertEqual(set(ids[:-5]), {0})
        self.assertEqual(self.engine.tokenize("a red fox runs"), [4, 5, 6, 7])

    def test_condition_is_the_projection_of_the_real_tokens(self):
        out = self.engine(["a red fox runs"])
        self.assertEqual(len(out), 1)
        self.assertEqual(tuple(out[0].shape), (5, 18))

        ids = torch.tensor([[t for t, _ in self.engine.tokens("a red fox runs")[0]]])
        mask = (torch.arange(ids.shape[1]) >= ids.shape[1] - 5).long()[None]
        with torch.no_grad():
            _, hidden = self.model(ids, attention_mask=mask, intermediate_output="all", final_layer_norm_intermediate=False, dtype=torch.float32)
            expected = self.projection(hidden[:, :, -5:])[0]
        torch.testing.assert_close(out[0], expected, rtol=1e-4, atol=1e-5)

    def test_same_prompt_twice_is_encoded_once(self):
        calls = []
        original = self.engine.text_encoder.encode_token_weights
        self.engine.text_encoder.encode_token_weights = lambda tokens: calls.append(1) or original(tokens)
        try:
            out = self.engine(["a fox", "a fox"])
        finally:
            del self.engine.text_encoder.encode_token_weights
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(out), 2)


@unittest.skipUnless(FORGE and (GEMMA3_FILES / "tokenizer.model").is_file(), "needs the Gemma 3 tokenizer files (LTX_HF_DIR)")
class Gemma3TokenizerTests(unittest.TestCase):
    def test_sentencepiece_conversion_matches_the_official_tokenizer_json(self):
        converted = text_engine.gemma3_tokenizer(torch.frombuffer(bytearray((GEMMA3_FILES / "tokenizer.model").read_bytes()), dtype=torch.uint8))
        official = Tokenizer.from_file(str(GEMMA3_FILES / "tokenizer.json"))
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertEqual(converted.encode(prompt, add_special_tokens=False).ids, official.encode(prompt, add_special_tokens=False).ids)

    def test_tokenizer_json_from_embedded_bytes(self):
        data = torch.frombuffer(bytearray((GEMMA3_FILES / "tokenizer.json").read_bytes()), dtype=torch.uint8)
        tokenizer = text_engine.gemma4_tokenizer(data)
        self.assertEqual(tokenizer.encode(PROMPTS[0], add_special_tokens=False).ids[:3],
                         Tokenizer.from_file(str(GEMMA3_FILES / "tokenizer.json")).encode(PROMPTS[0], add_special_tokens=False).ids[:3])


if __name__ == "__main__":
    unittest.main()
