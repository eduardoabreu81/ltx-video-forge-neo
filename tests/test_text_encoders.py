"""Gemma 3 12B, Gemma 4 12B and the LTX text projection on CPU.

Layouts are checked against the published headers (tests/fixtures/ltx_headers.json.gz); the forward passes against
transformers on tiny models with random weights: Gemma 3 live (Forge Neo ships transformers 4.57), Gemma 4 from a
stored reference (tools/make_gemma4_reference.py, transformers 5.19).
"""

import unittest
from dataclasses import dataclass

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_ltx.native import gemma3, gemma4, projection

PAD = 4


def build(factory, device="meta"):
    with using_forge_operations(device=torch.device(device), dtype=torch.bfloat16 if device == "meta" else torch.float32):
        return factory()


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_gemma3_matches_the_published_files(self):
        model = build(gemma3.Gemma3_12B)
        for name in ("gemma_3_12B_it.safetensors", "gemma_3_12B_it_fp8_scaled.safetensors", "gemma_3_12B_it_fp4_mixed.safetensors"):
            with self.subTest(name):
                self.assertEqual(layout_problems(model, self.headers[name]["tensors"], gemma3.VISION_PREFIXES), [])

    def test_gemma4_matches_the_published_files(self):
        model = build(gemma4.Gemma4_12B)
        for name in ("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"):
            with self.subTest(name):
                skip = gemma4.VISION_PREFIXES + (projection.PREFIX,)
                self.assertEqual(layout_problems(model, self.headers[name]["tensors"], skip), [])

    def test_gemma4_global_layers_share_k_and_v(self):
        model = build(gemma4.Gemma4_12B)
        globals_ = [i for i, layer in enumerate(model.model.layers) if not layer.sliding_attention]
        self.assertEqual(globals_, [5, 11, 17, 23, 29, 35, 41, 47])
        self.assertIsNone(model.model.layers[5].self_attn.v_proj)

    def test_projection_per_version(self):
        expected = {
            "ltx-2-19b-distilled.safetensors": "single_linear",
            "ltx-2-19b-distilled-fp8.safetensors": "single_linear",
            "ltx-2.3-22b-distilled-1.1.safetensors": "dual_linear",
            "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors": "dual_linear",
        }
        for name, kind in expected.items():
            with self.subTest(name):
                tensors = self.headers[name]["tensors"]
                spec = projection.detect({k: v[1] for k, v in tensors.items()})
                self.assertEqual(spec["kind"], kind)
                self.assertEqual(spec["in_dim"], 49 * 3840)  # embedding + 48 layers of Gemma
                holder = torch.nn.Module()
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    holder.text_embedding_projection = projection.build(spec)
                only = {k: v for k, v in tensors.items() if k.startswith(projection.PREFIX)}
                self.assertEqual(layout_problems(holder, only), [])

    def test_gemma_files_tell_each_other_apart(self):
        # ComfyUI sd.py: Gemma 4 12B drops v_proj on its global layers, Gemma 3 12B has it everywhere
        gemma4_file = self.headers["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"]["tensors"]
        gemma3_file = self.headers["gemma_3_12B_it.safetensors"]["tensors"]
        self.assertNotIn("model.layers.5.self_attn.v_proj.weight", gemma4_file)
        self.assertIn("model.layers.5.self_attn.v_proj.weight", gemma3_file)


def _real(hidden):
    return hidden[:, :, PAD:]


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class Gemma3ForwardTests(unittest.TestCase):
    def test_matches_transformers(self):
        from transformers import Gemma3TextConfig, Gemma3TextModel

        torch.manual_seed(0)
        reference = Gemma3TextModel(Gemma3TextConfig(
            vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16, sliding_window=8,
            layer_types=["sliding_attention"] * 5 + ["full_attention"], rope_theta=1000000.0,
            rope_local_base_freq=10000.0, rope_scaling={"rope_type": "linear", "factor": 8.0},
            attn_implementation="eager")).eval()
        with torch.no_grad():
            for param in reference.parameters():
                param.copy_(torch.randn_like(param) * (0.3 if param.ndim == 1 else 0.05))

        @dataclass
        class Tiny(gemma3.Gemma3_12BConfig):
            vocab_size: int = 300
            hidden_size: int = 64
            intermediate_size: int = 128
            num_hidden_layers: int = 6
            num_attention_heads: int = 4
            num_key_value_heads: int = 2
            head_dim = 16
            sliding_attention = [8, 8, 8, 8, 8, False]

        from backend.nn.llm.llama import Llama2_

        mine = build(lambda: Llama2_(Tiny()), device="cpu")
        mine.load_state_dict(reference.state_dict(), strict=True)

        ids = torch.randint(3, 300, (1, 20))
        ids[0, :PAD] = 0
        mask = torch.ones_like(ids)
        mask[0, :PAD] = 0
        with torch.no_grad():
            expected = torch.stack(reference(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states, dim=1)
            _, hidden = mine(ids, attention_mask=mask, intermediate_output="all", final_layer_norm_intermediate=False, dtype=torch.float32)
        self.assertEqual(hidden.shape, expected.shape)
        torch.testing.assert_close(_real(hidden), _real(expected), rtol=1e-4, atol=1e-4)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class Gemma4ForwardTests(unittest.TestCase):
    def test_matches_transformers(self):
        stored = torch.load(FIXTURES / "gemma4_tiny_reference.pt", weights_only=True)
        c = stored["config"]

        @dataclass
        class Tiny(gemma4.Gemma4_12BConfig):
            vocab_size: int = c["vocab_size"]
            hidden_size: int = c["hidden_size"]
            intermediate_size: int = c["intermediate_size"]
            num_hidden_layers: int = c["num_hidden_layers"]
            num_attention_heads: int = c["num_attention_heads"]
            num_key_value_heads: int = c["num_key_value_heads"]
            num_global_key_value_heads: int = c["num_global_key_value_heads"]
            head_dim: int = c["head_dim"]
            global_head_dim: int = c["global_head_dim"]
            sliding_attention = [c["sliding_window"]] * 5 + [False]

        mine = build(lambda: gemma4.Gemma4Transformer(Tiny()), device="cpu")
        mine.load_state_dict(stored["state_dict"], strict=True)
        with torch.no_grad():
            _, hidden = mine(stored["input_ids"], attention_mask=stored["attention_mask"], intermediate_output="all",
                             final_layer_norm_intermediate=False, dtype=torch.float32)
        expected = stored["hidden_states"]
        self.assertEqual(hidden.shape, expected.shape)
        torch.testing.assert_close(_real(hidden), _real(expected), rtol=1e-4, atol=1e-4)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class ProjectionTests(unittest.TestCase):
    def test_keeps_the_trailing_real_tokens(self):
        spec = {"kind": "dual_linear", "in_dim": 3 * 8, "video_dim": 6, "audio_dim": 4, "video_bias": True, "audio_bias": True}
        layer = build(lambda: projection.build(spec), device="cpu")
        hidden = torch.randn(1, 3, 10, 8)
        mask = torch.tensor([[0] * 4 + [1] * 6])
        out = projection.project(layer, hidden, mask)
        self.assertEqual(tuple(out.shape), (1, 6, 10))
        torch.testing.assert_close(out, layer(hidden[:, :, 4:]))


if __name__ == "__main__":
    unittest.main()
