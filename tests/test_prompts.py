from torch.utils.data import DataLoader
import torch
from transformer_lens import ActivationCache
import unittest
from utils.prompts import CleanCorruptedDataset, CleanCorruptedCacheResults


class TestCleanCorruptedDataset(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.clean_tokens = torch.tensor([[1.0], [2.0]])
        self.corrupted_tokens = torch.tensor([[1.5], [2.5]])
        self.answer_tokens = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
        self.all_prompts = ["prompt1", "prompt2"]
        self.dataset = CleanCorruptedDataset(self.clean_tokens, self.corrupted_tokens, self.answer_tokens, self.all_prompts)

    def test_initialization(self):
        # Test if the assertion works when shapes don't match
        with self.assertRaises(AssertionError):
            CleanCorruptedDataset(self.clean_tokens, torch.tensor([[1.5]]), self.answer_tokens, self.all_prompts)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        item = self.dataset[0]
        self.assertTrue(
            torch.isclose(item[0], self.clean_tokens[0]).all()
        )
        self.assertTrue(
            torch.isclose(item[1], self.corrupted_tokens[0]).all()
        )
        self.assertTrue(
            torch.isclose(item[2], self.answer_tokens[0]).all()
        )

    def test_get_dataloader(self):
        dataloader = self.dataset.get_dataloader(batch_size=1)
        self.assertIsInstance(dataloader, DataLoader)

    # ... Add more tests for run_with_cache, mocking the model and other dependencies ...


class TestCleanCorruptedCacheResults(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        dataset = CleanCorruptedDataset(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.5], [2.5]]), torch.tensor([[1.0, 1.5], [2.0, 2.5]]), ["prompt1", "prompt2"])
        corrupted_cache = ActivationCache({}, model=None)  # Assuming a mock model
        clean_cache = ActivationCache({}, model=None)  # Assuming a mock model
        self.results = CleanCorruptedCacheResults(dataset, corrupted_cache, clean_cache, 0.5, 0.6, 0.7, 0.8)

    def test_initialization(self):
        self.assertEqual(self.results.corrupted_logit_diff, 0.5)
        self.assertEqual(self.results.clean_logit_diff, 0.6)
        self.assertEqual(self.results.corrupted_prob_diff, 0.7)
        self.assertEqual(self.results.clean_prob_diff, 0.8)