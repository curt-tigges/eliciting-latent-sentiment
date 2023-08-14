#%%
import einops
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
from jaxtyping import Float, Int, Bool
from typing import Dict, Iterable, List, Tuple, Union
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_dataset, tokenize_and_concatenate, get_act_name, test_prompt
from transformer_lens.hook_points import HookPoint
from tqdm.notebook import tqdm
import pandas as pd
from utils.store import load_array, save_html, save_array, is_file, get_model_name, clean_label, save_text
#%%
torch.set_grad_enabled(False)
device = "cuda"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
model.name = MODEL_NAME
#%%
sentiment_dir = load_array("km_2c_line_embed_and_mlp0", model)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(device=device, dtype=torch.float32)
sentiment_dir /= sentiment_dir.norm()
#%%
BATCH_SIZE = 64
owt_data = load_dataset("stas/openwebtext-10k", split="train")
dataset = tokenize_and_concatenate(owt_data, model.tokenizer)
data_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
)
#%%
def compute_mean_loss(model: HookedTransformer, data_loader: DataLoader) -> float:
    total_loss = 0
    for _, batch_value in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_tokens = batch_value['tokens'].to(device)
        loss = model(batch_tokens, return_type="loss", prepend_bos=False, loss_per_token=False)
        total_loss += loss
    average_loss = total_loss / len(data_loader)
    return average_loss
#%%
average_loss = compute_mean_loss(model, data_loader)
print(f"Average loss: {average_loss:.4f}")
#%%
#%%
def resample_hook(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint, 
    direction: Float[Tensor, "d_model"],
):
    assert 'resid' in hook.name
    assert input.shape == (BATCH_SIZE, model.cfg.n_ctx, model.cfg.d_model)
    assert direction.shape == (model.cfg.d_model,)
    assert direction.norm().item() == 1.0
    # shuffle input tensor along the batch dimension
    shuffled = input[torch.randperm(BATCH_SIZE)]
    orig_proj: Float[Tensor, "batch pos"] = einops.einsum(
        input, direction, 'batch pos d_model, d_model -> batch pos'
    )
    new_proj: Float[Tensor, "batch pos"] = einops.einsum(
        shuffled, direction, 'batch pos d_model, d_model -> batch pos'
    )
    return (
        input + (new_proj - orig_proj).unsqueeze(-1) * direction
    )
#%%
def get_resample_ablated_loss(
    direction: Float[Tensor, "d_model"],
    model: HookedTransformer,
):
    model.reset_hooks()
    hook = partial(resample_hook, direction=direction)
    model.add_hook(
        lambda name: 'resid_post' in name,
        hook,
        dir="fwd",
    )
    loss = compute_mean_loss(model, data_loader)
    model.reset_hooks()
    return loss
#%%
ablated_loss = get_resample_ablated_loss(sentiment_dir, model)
print(f"Ablated loss: {ablated_loss:.4f}")
#%%
for seed in range(1, 5):
    torch.manual_seed(seed)
    random_dir = torch.randn(model.cfg.d_model).to(device)
    random_dir /= random_dir.norm()
    random_ablated_loss = get_resample_ablated_loss(random_dir, model)
    print(f"Seed {seed} loss: {random_ablated_loss:.4f}")

# %%
