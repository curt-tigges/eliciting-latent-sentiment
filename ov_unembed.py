#%%
from jaxtyping import Float, Int, Bool
from transformer_lens import HookedTransformer
import torch
from torch import Tensor
from utils.prompts import neg_adj
from utils.store import save_array, load_array
#%%
# set grad enabled to false globally
torch.set_grad_enabled(False)
device = torch.device('cpu')
MODEL_NAME = 'gpt2-small'
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
model.name = MODEL_NAME
# %%
layer = 10
head = 4
# %%
# model.W_O: Float[Tensor, "layer head d_head d_model"]
# model.W_V: Float[Tensor, "layer head d_model d_head"]
# model.W_U: Float[Tensor, "d_model d_vocab"]
ov_unembed: Float[Tensor, "d_model d_vocab"] = (
    model.W_V[layer, head, :, :] @ model.W_O[layer, head, :, :] @ model.W_U
)
# %%
neg_tokens: Int[Tensor, "batch"] = model.to_tokens(
    neg_adj, prepend_bos=False
).squeeze(1)
neg_tokens.shape
#%%
# one-hot encode neg_tokens
neg_tokens: Int[Tensor, "batch d_vocab"] = torch.nn.functional.one_hot(
    neg_tokens, num_classes=model.cfg.d_vocab
).to(device=device, dtype=torch.float32)
neg_tokens.shape
# %%
neg_ov_directions: Float[Tensor, "batch d_model"] = neg_tokens @ ov_unembed.T
neg_ov_directions.shape
# %%
mean_ov_direction: Float[Tensor, "d_model"] = neg_ov_directions.mean(dim=0)
# %%
save_array(
    mean_ov_direction.cpu().detach().numpy(), 
    f"mean_ov_direction_{layer}_{head}", model
)
# %%
