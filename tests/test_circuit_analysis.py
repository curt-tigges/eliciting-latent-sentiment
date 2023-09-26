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
from tqdm.auto import tqdm
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set
from rich import print as rprint

from typing import List, Union
from utils.circuit_analysis import (
    get_logit_diff, residual_stack_to_logit_diff, cache_to_logit_diff, 
    project_to_subspace, create_cache_for_dir_patching, get_prob_diff,
    get_final_non_pad_token,
)
import unittest


class TestCircuitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        MODEL_NAME = "gpt2-small"
        cls.model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False,
            device=cls.device,
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
        cls.answers = [[" great", " terrible"]] * len(cls.pos_adjectives) + [[" terrible", " great"]] * len(cls.neg_adjectives)
        cls.answer_tokens = cls.model.to_tokens(cls.answers, prepend_bos=False)
        cls.clean_tokens = cls.model.to_tokens(cls.text)
        cls.answer_tokens = einops.repeat(cls.answer_tokens, "batch correct -> batch 1 correct")
        cls.clean_logits, cls.clean_cache = cls.model.run_with_cache(cls.text)

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
        
        clean_logit_diff = get_logit_diff(self.clean_logits, answer_tokens=self.answer_tokens)
        average_logit_diff = cache_to_logit_diff(self.clean_cache, self.answer_tokens, self.model, -1)
        self.assertLessEqual(torch.max(torch.abs(average_logit_diff - clean_logit_diff)), 1e-4)

    def test_accumulated_resid(self):
        clean_accumulated_residual: Float[Tensor, "layer 1 d_model"]
        clean_accumulated_residual, labels = self.clean_cache.accumulated_resid(
            layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
        )
        clean_logit_lens_logit_diffs = residual_stack_to_logit_diff(
            clean_accumulated_residual, self.clean_cache, self.answer_tokens, self.model, biased=True
        )
        self.assertEqual(clean_logit_lens_logit_diffs.shape, (self.model.cfg.n_layers + 1, ))

    def test_project_to_fullspace(self):
        # Test example
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        subspace = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = project_to_subspace(vectors, subspace)
        self.assertTrue(torch.allclose(result, vectors, atol=1e-5))

    def test_project_to_line(self):
        # Test example
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        subspace = torch.tensor([[1.0, 0.0]]).T
        result = project_to_subspace(vectors, subspace)
        expected_result = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-5))

    def test_project_to_subspace(self):
        # Test example
        vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        subspace = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
        result = project_to_subspace(vectors, subspace)
        expected_result = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-5))

    def test_create_cache_for_dir_patching(self):
        # Mock data
        clean_data = {
            'result': torch.tensor([1.0, 2.0]),
            'resid_pre': torch.tensor([2.0, 3.0]),
            'other': torch.tensor([4.0, 5.0])
        }
        corrupted_data = {
            'result': torch.tensor([0.5, 1.5]),
            'resid_pre': torch.tensor([1.5, 2.5]),
            'other': torch.tensor([3.5, 4.5])
        }
        sentiment_dir = torch.tensor([1.0, 0.0])

        # Create caches
        clean_cache = ActivationCache(clean_data, self.model)
        corrupted_cache = ActivationCache(corrupted_data, self.model)
        
        result_cache = create_cache_for_dir_patching(
            clean_cache, corrupted_cache, sentiment_dir, self.model
        )
        
        # Check if the patched values are correct and others remain unchanged
        self.assertTrue(torch.allclose(
            result_cache['result'], torch.tensor([1.0, 1.5]), 
            atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            result_cache['resid_pre'], torch.tensor([2.0, 2.5]), 
            atol=1e-5
        ))
        self.assertTrue(torch.allclose(
            result_cache['other'], torch.tensor([3.5, 4.5]), 
            atol=1e-5
        ))


class TestGetProbDiff(unittest.TestCase):

    def test_basic_functionality(self):
        logits = torch.tensor([[[0.5, 1.0, 0.0]]])
        answer_tokens = torch.tensor([[0, 1]])
        result = get_prob_diff(logits, answer_tokens)
        expected = torch.tensor([-0.19928])  # Pre-computed value based on the given logits and tokens
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_per_prompt_true(self):
        logits = torch.tensor([[[0.5, 1.0, 0.0]]])
        answer_tokens = torch.tensor([[0, 1]])
        result = get_prob_diff(logits, answer_tokens, per_prompt=True)
        expected = torch.tensor([-0.19928])  # Same as above because there's only one batch
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_multiple_batches(self):
        logits = torch.tensor([[[0.5, 1.0, 0.0]], [[0.5, 0.0, 1.0]]])
        answer_tokens = torch.tensor([[0, 1], [1, 2]])
        result = get_prob_diff(logits, answer_tokens, per_prompt=True)
        expected_mean_diff = (torch.tensor([-0.19928, -0.3202]))
        self.assertTrue(torch.allclose(result, expected_mean_diff, atol=1e-4))


class TestFinalNonPadToken(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logits = torch.tensor([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]
        ])

    def test_simple_case(self):
        # Test case 1: Simple case
        attention_mask = torch.tensor([
            [1, 1, 0],
            [0, 1, 1]
        ])
        expected = torch.tensor([
            [0.4, 0.5, 0.6],
            [0.3, 0.2, 0.1]
        ])
        self.assertTrue(torch.allclose(
            get_final_non_pad_token(self.logits, attention_mask), expected
        ))

    def test_all_zeros(self):
        # Test case 2: All zeros in attention mask
        attention_mask = torch.tensor([
            [0, 0, 0],
            [1, 1, 1]
        ])
        # check raises assertion error
        with self.assertRaises(AssertionError):
            get_final_non_pad_token(self.logits, attention_mask)

    def test_all_ones(self):
        # Test case 3: All ones in attention mask
        attention_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 1]
        ])
        expected = torch.tensor([
            [0.7, 0.8, 0.9],
            [0.3, 0.2, 0.1]
        ])
        self.assertTrue(torch.allclose(
            get_final_non_pad_token(self.logits, attention_mask), expected
        ))
    
    def test_single_batch(self):
        # Test case 4: Single batch
        logits = torch.tensor([
            [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]]
        ])
        attention_mask = torch.tensor([
            [1, 0, 1]
        ])
        expected = torch.tensor([
            [0.7, 0.8]
        ])
        self.assertTrue(torch.allclose(
            get_final_non_pad_token(logits, attention_mask), expected
        ))




if __name__ == "__main__":
    unittest.main()

