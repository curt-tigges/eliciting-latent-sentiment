#%%
from functools import partial
from typing import Iterable, Optional
from jaxtyping import Int, Float
import numpy as np
import torch
from torch import Tensor
import einops
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
km_line = torch.from_numpy(km_line).to(device, torch.float32)
km_line_unit = km_line / km_line.norm()
#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
adjective_token = 6
verb_token = 9
example_prompt_indexed = [f'{i}: {s}' for i, s in enumerate(example_prompt)]
print(example_prompt_indexed)
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
#%%
average_km_component: Float[Tensor, ""] = einops.einsum(
    X_t, km_line_unit, "b d, d -> b"
).mean()
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
induction_prompt = "Here, this is an induction prompt for you. \n Here, this is an induction for"
memorisation_prompt = "The famous quote from Neil Armstrong is: One small step for man, one giant leap for"
induction_answer = " you"
memorisation_answer = " mankind"
example_index = 1
example_string = model.to_string(clean_tokens[example_index])
example_answer = model.to_str_tokens(answer_tokens[example_index])[0]
# ============================================================================ #
# Baseline prompts

model.reset_hooks()
print('Baseline sentiment prompt')
utils.test_prompt(
    example_string, example_answer, model, 
    prepend_space_to_answer=False,
    prepend_bos=False,
    top_k=top_k,
)
#%%

model.reset_hooks()
print('Baseline induction prompt')
utils.test_prompt(
    induction_prompt, induction_answer, model, 
    prepend_space_to_answer=False,
    prepend_bos=False,
    top_k=top_k,
)
#%%

model.reset_hooks()
print('Baseline memorisation prompt')
utils.test_prompt(
    memorisation_prompt, memorisation_answer, model, 
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
# %%
def linear_hook_base(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint,
    tokens: Iterable[int] = (adjective_token, verb_token),
    layer: Optional[int] = None,
):
    assert 'hook_resid_post' in hook.name
    if layer is not None and hook.layer() != layer:
        return input
    proj = einops.einsum(input, km_line_unit, "b p d, d -> b p")
    avg_broadcast = einops.repeat(
        average_km_component, " -> b p", 
        b=input.shape[0], p=input.shape[1]
    )
    proj_diff: Float[Tensor, "batch pos 1"] = (
        avg_broadcast - proj
    )[:, tokens].unsqueeze(dim=-1)
    input[:, tokens, :] += proj_diff * km_line_unit
    return input

#%%
def name_filter(name: str):
    return 'hook_resid_post' in name
#%%
# ============================================================================ #
# Define a hook for each experiment
experiments = dict(
    # layer_zero_adj_verb_leace = partial(
    #     leace_hook_base,
    #     tokens=(adjective_token, verb_token),
    #     layer=0,
    #     double=False,
    # ),
    # layer_zero_adj_verb_double_leace = partial(
    #     leace_hook_base,
    #     tokens=(adjective_token, verb_token),
    #     layer=0,
    #     double=True,
    # ),
    # layer_zero_all_pos_leace = partial(
    #     leace_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    #     layer=0,
    #     double=False,
    # ),
    # layer_zero_all_pos_double_leace = partial(
    #     leace_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    #     layer=0,
    #     double=True,
    # ),

    # all_layer_adj_verb_leace = partial(
    #     leace_hook_base,
    #     tokens=(adjective_token, verb_token),
    #     double=False,
    # ),
    # all_layer_adj_verb_double_leace = partial(
    #     leace_hook_base,
    #     tokens=(adjective_token, verb_token),
    #     double=True,
    # ),
    # all_layer_all_pos_leace = partial(
    #     leace_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    #     double=False,
    # ),
    # all_layer_all_pos_double_leace = partial(
    #     leace_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    #     double=True,
    # ),


    layer_0_linear_adj_verb = partial(
        linear_hook_base,
        layer=0,
        tokens=(adjective_token, verb_token),
    ),
    all_layer_linear_adj_verb = partial(
        linear_hook_base,
        tokens=(adjective_token, verb_token),
    ),
    # layer_0_linear = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    # ),
    # all_layer_linear = partial(
    #     linear_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    # ),
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
utils.test_prompt(
    induction_prompt, induction_answer, model,
    prepend_space_to_answer=False,
    prepend_bos=False,
    top_k=top_k,
)
# %%
utils.test_prompt(
    memorisation_prompt, memorisation_answer, model,
    prepend_space_to_answer=False,
    prepend_bos=False,
    top_k=top_k,
)
#%%