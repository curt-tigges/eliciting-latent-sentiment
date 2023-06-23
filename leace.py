#%%
from functools import partial
from typing import Iterable, Optional
from jaxtyping import Int, Float
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import einops
from sklearn.linear_model import LogisticRegression
from concept_erasure import ConceptEraser
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from prompt_utils import get_dataset
import plotly.express as px
from cache_utils import (
    residual_sentiment_sim_by_head, residual_sentiment_sim_by_pos
)
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
model.cfg.use_attn_results = True
#%%
heads = model.cfg.n_heads
layers = model.cfg.n_layers
#%%
km_line = np.load('data/km_line_embed_and_mlp0.npy')
km_line = torch.from_numpy(km_line).to(device, torch.float32)
km_line_unit = km_line / km_line.norm()
#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(model, device)
#%%
batch_size = clean_tokens.shape[0]
sentiment_repeated = einops.repeat(
    km_line, "d_model -> batch d_model", batch=batch_size
)
even_batch_repeated = einops.repeat(
    torch.arange(batch_size, device=device) % 2 == 0, 
    "batch -> batch d_model", 
    d_model=len(km_line)
)
sentiment_directions: Float[Tensor, "batch d_model"] = torch.where(
    even_batch_repeated,
    sentiment_repeated,
    -sentiment_repeated,
).to(device)
#%%
example_prompt = model.to_str_tokens(clean_tokens[0])
adjective_token = 6
verb_token = 9
example_prompt_indexed = [f'{i}: {s}' for i, s in enumerate(example_prompt)]
seq_len = len(example_prompt)
print(example_prompt_indexed)
#%%
all_adjectives = [
    model.to_str_tokens(clean_tokens[i])[adjective_token] 
    for i in range(len(clean_tokens))
]
all_adjectives[:5]

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
#%%
def mean_ablate_km_component(
    input: Float[Tensor, "batch pos d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
):
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
# ============================================================================ #
# Fit LEACE

#%% # Setup training data
clean_embeddings: Float[Tensor, "batch pos d_model"] = embed_and_mlp0(
    clean_tokens, model
).to(torch.float32)
clean_projections: Float[Tensor, "batch"] = einops.einsum(
    clean_embeddings, km_line_unit, "b s d, d -> b s"
)
X_t: Float[Tensor, "batch d_model"] = clean_embeddings[:, adjective_token, :]
X_c: Float[Tensor, "batch"] = clean_projections[:, adjective_token]
Y_t: Float[Tensor, "batch"] = (
    torch.arange(len(X_t), device=device) % 2 == 0
).to(torch.int64)
X = X_t.cpu().detach().numpy()
Y = Y_t.cpu().detach().numpy()
X_t.shape, Y_t.shape, X_t.dtype, Y_t.dtype
#%%

pre_comp_df = pd.DataFrame({
    'token': all_adjectives,
    'km_components': X_c.cpu().detach().numpy(),
})
px.histogram(
    data_frame=pre_comp_df,
    x='km_components',
    hover_data=['token'], 
    marginal="rug",
    nbins=100,
    title="Histogram of km_components before intervention",
)
#%%
average_km_component: Float[Tensor, ""] = X_c.mean()
#%% # Before concept erasure
real_lr = LogisticRegression(max_iter=1000).fit(X, Y)
beta = torch.from_numpy(real_lr.coef_[0, :]).to(device, torch.float32)
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
X_ablated = mean_ablate_km_component(X_t.unsqueeze(1), (0)).squeeze(1)
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
X_c_ablated: Float[Tensor, "batch"] = einops.einsum(
    X_ablated, km_line_unit, "b d, d -> b"
)
post_comp_df = pd.DataFrame({
    'token': all_adjectives,
    'km_components': X_c_ablated.cpu().detach().numpy(),
})
px.histogram(
    data_frame=post_comp_df,
    x='km_components',
    hover_data=['token'], 
    marginal="rug",
    nbins=100,
    title="Histogram of km_components after ablation",
)
#%%
# ============================================================================ #
# Hooks
top_k = 20
induction_prompt = "Here, this is an induction prompt for you. \n Here, this is an induction for"
memorisation_prompt = "The famous quote from Neil Armstrong is: One small step for man, one giant leap for"
induction_answer = " you"
memorisation_answer = " mankind"
example_index = 0
example_string = model.to_string(clean_tokens[example_index])
example_answer = model.to_str_tokens(answer_tokens[example_index])[0]
#%%
def stack_name_filter(name: str) -> bool:
    return name.endswith('result') or name.endswith('z') or name.endswith('_scale')
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
_, base_cache = model.run_with_cache(
    clean_tokens,
    names_filter=stack_name_filter,
    prepend_bos=False,
    return_type=None
)
#%%
per_pos_sentiment = residual_sentiment_sim_by_pos(
    base_cache, sentiment_directions, len(example_prompt)
)
del base_cache
fig = px.imshow(
    per_pos_sentiment.squeeze().cpu().detach().numpy(),
    labels={'x': 'Position', 'y': 'Component'},
    title=f'Per position sentiment for baseline',
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    x=example_prompt,
    y=[f'L{l}H{h}' for l in range(layers) for h in range(heads)],
    height = heads * layers * 20,
)
fig.show()
#%%

# model.reset_hooks()
# print('Baseline induction prompt')
# utils.test_prompt(
#     induction_prompt, induction_answer, model, 
#     prepend_space_to_answer=False,
#     prepend_bos=False,
#     top_k=top_k,
# )
#%%

# model.reset_hooks()
# print('Baseline memorisation prompt')
# utils.test_prompt(
#     memorisation_prompt, memorisation_answer, model, 
#     prepend_space_to_answer=False,
#     prepend_bos=False,
#     top_k=top_k,
# )
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
    return mean_ablate_km_component(input, tokens=tokens)
    

#%%
def hook_name_filter(name: str):
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


    # layer_0_linear_adj_verb = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=(adjective_token, verb_token),
    # ),
    # all_layer_linear_adj_verb = partial(
    #     linear_hook_base,
    #     tokens=(adjective_token, verb_token),
    # ),
    # layer_0_linear = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    # ),
    all_layer_linear = partial(
        linear_hook_base,
        tokens=np.arange(len(example_prompt)),
    ),
)
#%%
for experiment_name, experiment_hook in experiments.items():
    model.reset_hooks()
    model.add_hook(
        hook_name_filter,
        experiment_hook
    )
    print(experiment_name)
    utils.test_prompt(
        example_string, example_answer, model, 
        prepend_space_to_answer=False,
        prepend_bos=False,
        top_k=top_k,
    )
    _, test_cache = model.run_with_cache(
        clean_tokens, 
        prepend_bos=False,
        return_type=None
    )
    per_pos_sentiment = residual_sentiment_sim_by_pos(
        test_cache, sentiment_directions, len(example_prompt)
    )
    fig = px.imshow(
        per_pos_sentiment.squeeze().cpu().detach().numpy(),
        labels={'x': 'Position', 'y': 'Component'},
        title=f'Per position sentiment for {experiment_name}',
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        x=example_prompt,
        y=[f'L{l}H{h}' for l in range(layers) for h in range(heads)],
        height = heads * layers * 20,
    )
    fig.show()


# # %%
# utils.test_prompt(
#     induction_prompt, induction_answer, model,
#     prepend_space_to_answer=False,
#     prepend_bos=False,
#     top_k=top_k,
# )
# # %%
# utils.test_prompt(
#     memorisation_prompt, memorisation_answer, model,
#     prepend_space_to_answer=False,
#     prepend_bos=False,
#     top_k=top_k,
# )
#%%