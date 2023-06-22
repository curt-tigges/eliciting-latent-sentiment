#%%
from functools import partial
from typing import Iterable, Optional
from jaxtyping import Int, Float
import numpy as np
import torch
from torch import Tensor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from concept_erasure import ConceptEraser
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from prompt_utils import get_dataset
import plotly.express as px
#%%
torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
#%%
km_line = np.load('data/km_line_embed_and_mlp0.npy')
km_line = torch.from_numpy(km_line).to(device)
#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
example_string = model.to_string(clean_tokens[0])
adjective_token = example_prompt.index(" perfect")
verb_token = example_prompt.index(' loved')
example_prompt_indexed = [f'{i}: {s}' for i, s in enumerate(example_prompt)]
example_answer = model.to_str_tokens(answer_tokens[0])[0]
print(example_prompt_indexed)
print(example_answer)
print(adjective_token, verb_token)
#%%
def embed_and_mlp0(
    tokens: Int[Tensor, "batch 1"],
    transformer: HookedTransformer = model
):
    block0 = transformer.blocks[0]
    resid_mid = transformer.embed(tokens)
    mlp_out = block0.mlp((resid_mid))
    resid_post = resid_mid + mlp_out
    return block0.ln2(resid_post)
# ============================================================================ #
# Fit LEACE

#%% # Setup training data
X_t: Float[Tensor, "batch pos d_model"] = embed_and_mlp0(clean_tokens, model)
X_t: Float[Tensor, "batch d_model"] = X_t[:, adjective_token, :].to(
    torch.float64
)
Y_t: Float[Tensor, "batch"] = (
    torch.arange(len(X_t), device=device) % 2 == 0
).to(torch.int64)
X = X_t.cpu().detach().numpy()
Y = Y_t.cpu().detach().numpy()
X_t.shape, Y_t.shape, X_t.dtype, Y_t.dtype
#%% # Before concept erasure
real_lr = LogisticRegression(max_iter=1000).fit(X, Y)
beta = torch.from_numpy(real_lr.coef_[0, :]).to(device)
print('L infinity norm', beta.norm(p=torch.inf))
assert beta.norm(p=torch.inf) > 0.1
#%% # compute cosine similarity of beta and km_line
print(
    'Cosine similarity', 
    torch.dot(beta / beta.norm(), km_line / km_line.norm())
)
#%% # fit eraser
eraser = ConceptEraser.fit(X_t, Y_t)
X_ = eraser(X_t)
#%% # LR learns nothing after
null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(
    X_.cpu().detach().numpy(), Y
)
beta = torch.from_numpy(null_lr.coef_[0])
print(beta.norm(p=torch.inf))
assert beta.norm(p=torch.inf) < 1e-4
# %%
px.line(beta)
#%%
# ============================================================================ #
# Hooks
top_k = 20
# ============================================================================ #
# Baseline prompt

model.reset_hooks()
print('Baseline prompt')
utils.test_prompt(
    example_string, example_answer, model, 
    prepend_space_to_answer=False,
    prepend_bos=False,
    top_k=top_k,
)

# %%
def leace_hook_base(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint,
    tokens: Iterable[int] = (adjective_token, verb_token),
    double: bool = False,
    layer: Optional[int] = None,
):
    if layer is not None and hook.layer() != layer:
        return input
    for token in tokens:
        input[:, token, :] += (
            eraser(input[:, token, :]) - input[:, token, :]
            ) * (2 if double else 1)
    assert 'hook_resid_post' in hook.name
    return input

#%%
def name_filter(name: str):
    return 'hook_resid_post' in name
#%%
# ============================================================================ #
# Define a hook for each experiment
experiments = dict(
    layer_zero_adj_verb_hook = partial(
        leace_hook_base,
        tokens=(adjective_token, verb_token),
        layer=0,
        double=False,
    ),
    layer_zero_adj_verb_double_hook = partial(
        leace_hook_base,
        tokens=(adjective_token, verb_token),
        layer=0,
        double=True,
    ),
    layer_zero_all_pos_hook = partial(
        leace_hook_base,
        tokens=np.arange(len(example_prompt)),
        layer=0,
        double=False,
    ),
    layer_zero_all_pos_double_hook = partial(
        leace_hook_base,
        tokens=np.arange(len(example_prompt)),
        layer=0,
        double=True,
    ),

    all_layer_adj_verb_hook = partial(
        leace_hook_base,
        tokens=(adjective_token, verb_token),
        double=False,
    ),
    all_layer_adj_verb_double_hook = partial(
        leace_hook_base,
        tokens=(adjective_token, verb_token),
        double=True,
    ),
    all_layer_all_pos_hook = partial(
        leace_hook_base,
        tokens=np.arange(len(example_prompt)),
        double=False,
    ),
    all_layer_all_pos_double_hook = partial(
        leace_hook_base,
        tokens=np.arange(len(example_prompt)),
        double=True,
    ),
)
#%%
for experiment_name, experiment_hook in experiments.items():
    model.reset_hooks()
    model.add_hook(
        name_filter,
        experiment_hook
    )
    print(experiment_name)
    utils.test_prompt(
        example_string, example_answer, model, 
        prepend_space_to_answer=False,
        prepend_bos=False,
        top_k=top_k,
    )


# %%
