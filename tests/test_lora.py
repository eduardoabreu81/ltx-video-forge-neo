"""LoRAs on the LTX DiT: the published distilled LoRAs against the DiT's own layer names through Forge Neo's key map,
and on-the-fly LoRAs as low-rank terms on the INT8 layers (native/lora.py, from the H3 extension) with stand-ins for
Forge Neo's pieces."""

import types
import unittest

from forge_env import FIXTURES, FORGE, headers

try:
    import comfy_kitchen  # noqa: F401
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None


class Adapter:
    """Forge's LoRAAdapter: weights = (up, down, alpha, mid, dora_scale, reshape)."""
    name = "lora"

    def __init__(self, up, down, alpha=None, mid=None, dora_scale=None):
        self.weights = (up, down, alpha, mid, dora_scale, None)


class OnlineLoRAPatch:
    """Forge's OnlineLoRAPatch: merges its LoRA into the dequantized weight at every call."""

    def __init__(self, adapter, strength=1.0, strength_model=1.0):
        self.patch = [[strength, adapter, strength_model, None, None]]

    def __call__(self, weight):
        strength, adapter, strength_model, _, _ = self.patch[0]
        up, down, alpha = adapter.weights[:3]
        scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
        return weight * strength_model + scale * up @ down


if torch is not None:
    class QuantLinear(nn.Linear):
        """Forge's mixed-precision Linear: the quantized kernel without weight_function, a merged weight with it."""
        layout_type = "TensorCoreINT8Layout"

        def __init__(self, n_in, n_out):
            super().__init__(n_in, n_out, bias=False)
            self.weight_function, self.bias_function = [], []
            self.int8_calls = self.merged_calls = 0

        def forward(self, x):
            if not self.weight_function:
                self.int8_calls += 1
                return F.linear(x, self.weight, self.bias)
            self.merged_calls += 1
            weight = self.weight
            for function in self.weight_function:
                weight = function(weight)
            return F.linear(x, weight, self.bias)


@unittest.skipIf(torch is None, "needs torch and comfy-kitchen")
class LowRankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(QuantLinear(32, 48), nn.ReLU(), QuantLinear(48, 16))
        self.x = torch.randn(5, 32)

    def lora(self, n_out, n_in, rank=4, **kwargs):
        return OnlineLoRAPatch(Adapter(torch.randn(n_out, rank), torch.randn(rank, n_in), **kwargs.pop("adapter", {})), **kwargs)

    def run_both(self):
        from forge_ltx.native import lora
        merged = self.model(self.x)
        with lora.low_rank(self.model) as moved:
            low_rank = self.model(self.x)
        return merged, low_rank, moved

    def test_plain_loras_keep_the_int8_path_with_the_same_result(self):
        first, second = self.model[0], self.model[2]
        first.weight_function = [self.lora(48, 32, adapter={"alpha": 8.0}, strength=0.7), self.lora(48, 32)]
        second.weight_function = [self.lora(16, 48)]
        functions = list(first.weight_function)
        merged, low_rank, moved = self.run_both()
        self.assertEqual(moved, 2)
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4), float((merged - low_rank).abs().max()))
        self.assertEqual((first.int8_calls, first.merged_calls), (1, 1))
        # Forge's own state is back after the forward, and the layer is plain again outside it
        self.assertEqual(first.weight_function, functions)
        self.assertFalse(first.forward.active)
        self.model(self.x)
        self.assertEqual(first.merged_calls, 2)

    def test_other_patches_stay_on_forges_path(self):
        first, second = self.model[0], self.model[2]
        first.weight_function = [self.lora(48, 32, adapter={"dora_scale": torch.ones(48, 1)})]
        second.weight_function = [self.lora(16, 48, strength_model=0.5)]
        merged, low_rank, moved = self.run_both()
        self.assertEqual(moved, 0)
        self.assertTrue(torch.equal(merged, low_rank))
        self.assertEqual((first.int8_calls, second.int8_calls), (0, 0))

    def test_a_new_lora_set_or_none_is_picked_up(self):
        from forge_ltx.native import lora
        first = self.model[0]
        first.weight_function = [self.lora(48, 32)]
        self.run_both()
        first.weight_function = [self.lora(48, 32, strength=2.0)]
        merged, low_rank, _ = self.run_both()
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4))
        first.weight_function = []
        plain = self.model(self.x)
        with lora.low_rank(self.model) as moved:
            self.assertEqual(moved, 0)
            self.assertTrue(torch.equal(self.model(self.x), plain))

    def test_layers_that_are_not_quantized_are_left_alone(self):
        from forge_ltx.native import lora
        layer = nn.Linear(32, 8, bias=False)
        layer.weight_function, layer.bias_function = [self.lora(8, 32)], []
        with lora.low_rank(nn.Sequential(layer)) as moved:
            self.assertEqual(moved, 0)


    def test_the_dit_forward_runs_its_loras_as_low_rank_terms(self):
        from backend.operations import using_forge_operations

        from forge_ltx.native.forge_dit import ForgeLTXAV
        from forge_ltx.native.streams import Generation, StreamShapes
        stored = torch.load(FIXTURES / "ltx_dit_tiny_reference.pt", weights_only=False)["2.3"]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = ForgeLTXAV(**stored["config"]).eval()
        model.load_state_dict(stored["state_dict"], strict=True)
        attn = model.transformer_blocks[0].attn1
        n_out, n_in = attn.to_q.weight.shape
        layer = QuantLinear(n_in, n_out)
        layer.weight.data.copy_(attn.to_q.weight)
        layer.bias = attn.to_q.bias
        layer.weight_function = [self.lora(n_out, n_in)]
        shapes = StreamShapes(video=(1, 8, 3, 2, 3), audio=(1, 8, 7, 16))
        model.generation = Generation(shapes=shapes, seed=1, fps=24.0, frames=17)
        packed = shapes.pack(torch.randn(shapes.video), torch.randn(shapes.audio))
        with torch.inference_mode():
            attn.to_q.weight.copy_(layer.weight_function[0](layer.weight))
            merged = model(packed, torch.tensor([0.7]), stored["text"].clone())
            attn.to_q = layer
            low_rank = model(packed, torch.tensor([0.7]), stored["text"].clone())
        self.assertTrue(torch.allclose(merged, low_rank, atol=1e-4), float((merged - low_rank).abs().max()))
        self.assertEqual((layer.int8_calls, layer.merged_calls), (1, 0))


# the distilled LoRAs Lightricks publishes, by the version of the dev checkpoint they adapt
DISTILLED_LORAS = {"ltx-2-19b-distilled-lora-384.safetensors": "ltx-2-19b-dev-fp4.safetensors",
                   "ltx-2.3-22b-distilled-lora-384-1.1.safetensors": "ltx-2.3-22b-dev.safetensors",
                   "ltx-2.5-22b-distilled-lora-450-bf16.safetensors": "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (LTX_FORGE_DIR)")
class KeyMapTests(unittest.TestCase):
    """Forge Neo maps a LoRA by the model's own state dict (comfy lora.model_lora_keys_unet); the LTX LoRAs name the
    DiT's layers as ComfyUI does, so every one of their tensors has to land on a layer of the DiT."""

    def test_distilled_loras_cover_the_dit(self):
        from backend.operations import using_forge_operations
        from backend.patcher.lora import load_lora, model_lora_keys_unet

        from forge_ltx.native import detect, model
        from forge_ltx.native.transformer import LTXAVModel
        entries = headers()
        for name, dit in DISTILLED_LORAS.items():
            with self.subTest(name):
                entry = entries[dit]
                config = detect.dit_config({k: v[1] for k, v in entry["tensors"].items()}, entry["metadata"])
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    dit_model = LTXAVModel(**config)
                state = {f"diffusion_model.{k}": v for k, v in dit_model.state_dict().items()}
                patcher_model = types.SimpleNamespace(state_dict=lambda state=state: state, diffusion_model=types.SimpleNamespace(config=config),
                                                      config=types.SimpleNamespace(huggingface_repo=model.CONFIG_DIR))
                key_map = model_lora_keys_unet(patcher_model, {})
                lora_sd = {k: torch.empty(v[1], device="meta") for k, v in entries[name]["tensors"].items()}
                patches = load_lora(lora_sd, key_map)
                patches = patches[0] if isinstance(patches, tuple) else patches
                layers = {k.rsplit(".lora_", 1)[0] for k in lora_sd}
                # LTX-2.0's LoRA also adapts the text projection, which lives in the text encoder; ComfyUI renames that
                # layer in its LTX-2.0 encoder and leaves the LoRA tensor unloaded too
                unloaded = {layer for layer in layers if key_map.get(layer) not in patches}
                self.assertLessEqual(unloaded, {"text_embedding_projection.aggregate_embed"})
                self.assertEqual(len(patches), len(layers - unloaded))
                for target, adapter in patches.items():
                    up, down = adapter.weights[:2]
                    self.assertEqual((up.shape[0], down.shape[1]), tuple(state[target].shape[:2]), target)
