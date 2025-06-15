"""Tokenizer for Llama model with position stacking. ONLY single input is supported."""

import numpy as np
from transformers import Qwen2Tokenizer
from ..llama.tokenization_llama import ps_call


class Qwen2TokenizerWithPS(Qwen2Tokenizer):
     def __call__(cls, input_list):
       return ps_call(input_list, super().__call__)



