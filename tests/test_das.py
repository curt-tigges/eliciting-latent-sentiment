import unittest
import torch
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
from utils.das import RotateLayer, InverseRotateLayer, hook_fn_base, act_patch_simple, TrainingConfig, train_das_subspace
from utils.prompts import PromptType


class TestDASFunctions(unittest.TestCase):

    def test_rotate_layer(self):
        torch.manual_seed(42)
        layer = RotateLayer(5, torch.device('cpu'))
        x = torch.randn(3, 5)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_inverse_rotate_layer(self):
        torch.manual_seed(42)
        layer = RotateLayer(5, torch.device('cpu'))
        x = torch.randn(3, 5)
        inv_layer = InverseRotateLayer(layer)
        y = inv_layer(layer(x))
        self.assertTrue(torch.allclose(y, x))

    def test_hook_fn_base(self):
        resid = torch.ones(3, 4, 5)
        layer = 0
        hook = HookPoint()
        hook.layer = lambda: layer
        hook.name = f'blocks.{layer}.resid_pre'
        new_value = torch.full((3, 5), 2)
        out = hook_fn_base(resid, hook, 0, 1, new_value)
        self.assertEqual(out[0, 1, 0], 2)
        self.assertEqual(out[0, 0, 0], 1)

    def test_act_patch_simple(self):
        layer = 10
        model = unittest.mock.MagicMock()
        model.run_with_hooks.return_value = torch.ones(3, 10)
        model.cfg.n_layers = layer
        orig_input = "test"
        new_value = torch.full((3, 5), 2)
        patching_metric = lambda x: torch.sum(x)
        out = act_patch_simple(model, orig_input, new_value, layer, 1, patching_metric)
        self.assertEqual(out.item(), 30)

    def test_training_config(self):
        config_dict = {
            "batch_size": 1,
            "seed": 42,
            "lr": 0.001,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
            "epochs": 10,
            "d_das": 2,
            "wandb_enabled": False,
            "model_name": "test_model",
            "clip_grad_norm": .999,
        }
        config = TrainingConfig(config_dict)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.to_dict(), config_dict)

    def test_global_grad_enabled(self):
        self.assertTrue(torch.is_grad_enabled())

    def test_train_das_direction(self):
        device = torch.device('cpu')
        model = HookedTransformer.from_pretrained(
            'attn-only-1l',
            device=device,
        ).train()
        model.name = 'test'
        direction = train_das_subspace(
            model, device,
            PromptType.SIMPLE, 'ADJ', 0,
            PromptType.SIMPLE, 'ADJ', 0,
            wandb_enabled=False,
            epochs=1,
        )
        self.assertTrue(isinstance(direction, torch.Tensor))


if __name__ == "__main__":
    unittest.main()

# %%
