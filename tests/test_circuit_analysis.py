from IPython.display import display, HTML
import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
import yaml

import einops


import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.patching as patching

from torch import Tensor
from tqdm.notebook import tqdm
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from rich import print as rprint

from typing import List, Union
from utils.circuit_analysis import get_logit_diff, residual_stack_to_logit_diff, cache_to_logit_diff
import unittest


class TestTransformerLens(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_grad_enabled(False)
        cls.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        MODEL_NAME = "gpt2-small"
        cls.model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False,
        )
        cls.model.name = MODEL_NAME

        cls.pos_adjectives = [
            ' mesmerizing',
            ' heartwarming',
            ' captivating',
            ' enlightening',
            ' transcendent',
            ' riveting',
            ' spellbinding',
            ' masterful',
            ' exhilarating',
            ' uplifting',
            ' electrifying',
        ]
        cls.neg_adjectives = [
            ' forgettable',
            ' overwrought',
            ' pretentious',
            ' unimaginative',
            ' disengaging',
            ' incoherent',
            ' contrived',
            ' overrated',
            ' ponderous',
            ' formulaic',
            ' dissonant'
        ]
        cls.text = [
            f"The movie was {adj}. I thought it was" for adj in cls.pos_adjectives + cls.neg_adjectives
        ]

    def test_adjective_tokens(self):
        for adj in self.pos_adjectives + self.neg_adjectives:
            adj_tokens = self.model.to_str_tokens(adj, prepend_bos=False)
            self.assertEqual(len(adj_tokens), 2, f"Bad length {len(adj_tokens)} for {adj_tokens}")

    def test_text_token_length(self):
        seq_len = len(self.model.to_str_tokens(self.text[0]))
        for s in self.text:
            self.assertEqual(len(self.model.to_str_tokens(s)), seq_len, (
                f"Sequence length mismatch: {len(self.model.to_str_tokens(s))} != {seq_len}"
                f"for {self.model.to_str_tokens(s)}"
            ))

    def test_logit_diff_calculation(self):
        answers = [[" great", " terrible"]] * len(self.pos_adjectives) + [[" terrible", " great"]] * len(self.neg_adjectives)
        answer_tokens = self.model.to_tokens(answers, prepend_bos=False)
        clean_tokens = self.model.to_tokens(self.text)
        self.assertEqual(len(clean_tokens), len(answer_tokens))
        answer_tokens = einops.repeat(answer_tokens, "batch correct -> batch 1 correct")
        clean_logits, clean_cache = self.model.run_with_cache(self.text)
        clean_logit_diff = get_logit_diff(clean_logits, answer_tokens=answer_tokens)
        average_logit_diff = cache_to_logit_diff(clean_cache, answer_tokens, self.model, -1)
        self.assertLessEqual(torch.max(torch.abs(average_logit_diff - clean_logit_diff)), 1e-4)

if __name__ == "__main__":
    unittest.main()

