"""LTX-2.x prompt encoding, ported from ComfyUI comfy/text_encoders/lt.py (and the Gemma tokenizers it uses).

The prompt is not chat-templated: BOS, then the raw text, left-padded with 0 to 1024 tokens. Gemma runs with the
attention mask and returns the hidden state of every layer; the projection keeps the trailing real tokens
(projection.project) and maps them to the DiT's video and audio context.

Tokenizers come from the text encoder file itself: Gemma 4 embeds its tokenizer.json (tokenizer_json), Gemma 3 its
SentencePiece model (spiece_model). Forge Neo does not ship the sentencepiece package, so the SentencePiece model is
converted to a tokenizers BPE with transformers' own Gemma converter, reading the pieces straight from the model proto.
"""

import os
import tempfile
import types

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer
from tokenizers import Tokenizer

from . import projection as projection_module

BOS = 2
PAD = 0
MIN_LENGTH = 1024  # ComfyUI pads every LTX prompt to this many tokens, on the left


def _bytes(data) -> bytes:
    return bytes(data.tolist()) if isinstance(data, torch.Tensor) else bytes(data)


def gemma4_tokenizer(tokenizer_json) -> Tokenizer:
    return Tokenizer.from_str(_bytes(tokenizer_json).decode("utf-8"))


def gemma3_tokenizer(spiece_model) -> Tokenizer:
    from transformers.convert_slow_tokenizer import (
        GemmaConverter,
        generate_merges,
        import_protobuf,
    )

    class ProtoExtractor:
        # transformers' GemmaSentencePieceExtractor, with the pieces read from the proto instead of sentencepiece
        def __init__(self, path):
            proto = import_protobuf().ModelProto()
            with open(path, "rb") as f:
                proto.ParseFromString(f.read())
            self.pieces = [p.piece for p in proto.pieces]

        def extract(self, vocab_scores=None):
            vocab = {piece: i for i, piece in enumerate(self.pieces)}
            if "\t" not in vocab:
                vocab["\t"] = vocab.get("<0x09>")
            return vocab, generate_merges(vocab, vocab_scores)

    class Converter(GemmaConverter):
        SpmExtractor = ProtoExtractor

    fd, path = tempfile.mkstemp(suffix=".model")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_bytes(spiece_model))
        original = types.SimpleNamespace(vocab_file=path, add_bos_token=True, add_eos_token=False, legacy=False,
                                         pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>")
        return Converter(original).converted()
    finally:
        os.remove(path)


class _Ids:
    """What Forge Neo's SDTokenizer calls: text -> {"input_ids": [...]}, without special tokens."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> dict:
        return {"input_ids": self.tokenizer.encode(text, add_special_tokens=False).ids}


class LTXTextEngine:
    def __init__(self, text_encoder, tokenizer: Tokenizer, projection):
        self.text_encoder = SDClipModel(text_encoder, layer="all", special_tokens={"start": BOS, "pad": PAD}, layer_norm_hidden_state=False,
                                        enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(_Ids(tokenizer), pad_with_end=False, has_start_token=False, start_token=BOS, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=MIN_LENGTH, pad_left=True, pad_token=PAD,
                                     disable_weights=True)
        self.projection = projection

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, text: str) -> list[int]:
        # the prompt's own tokens, for Forge's token counter
        return self.tokenizer.tokenizer(text)["input_ids"]

    def tokens(self, text: str) -> list[list[tuple]]:
        return self.tokenizer.tokenize_with_weights(text)

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                hidden, _, extra = self.text_encoder.encode_token_weights(self.tokens(line))
                cache[line] = projection_module.project(self.projection, hidden, extra["attention_mask"])
            zs.extend(cache[line])  # (T, D) per prompt; the prompt parser stacks the batch
        return zs
