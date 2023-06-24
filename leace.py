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
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from prompt_utils import get_dataset, get_logit_diff, logit_diff_denoising
import plotly.express as px
from cache_utils import (
    residual_sentiment_sim_by_head, residual_sentiment_sim_by_pos
)
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%%
torch.set_grad_enabled(False)
device = torch.device("cpu")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.cfg.use_attn_result = True
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
clean_logits, clean_cache = model.run_with_cache(
    clean_tokens,
    names_filter=lambda name: name.endswith('resid_post'), 
    prepend_bos=False,
)
clean_cache.to(device)

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
    multiplier: float = 1.0,
):
    proj = einops.einsum(input, km_line_unit, "b p d, d -> b p")
    avg_broadcast = einops.repeat(
        average_km_component, " -> b p", 
        b=input.shape[0], p=input.shape[1]
    )
    proj_diff: Float[Tensor, "batch pos 1"] = (
        avg_broadcast - proj
    )[:, tokens].unsqueeze(dim=-1)
    input[:, tokens, :] += multiplier * proj_diff * km_line_unit
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
print('Average km component', average_km_component)
#%% # Before concept erasure
real_lr = LogisticRegression(max_iter=1000).fit(X, Y)
print('Accuracy', real_lr.score(X, Y))
real_beta = torch.from_numpy(real_lr.coef_[0, :]).to(device, torch.float32)
print('L infinity norm', real_beta.norm(p=torch.inf))
assert real_beta.norm(p=torch.inf) > 0.1
#%% # compute cosine similarity of beta and km_line
print(
    'Cosine similarity', 
    torch.dot(real_beta / real_beta.norm(), km_line / km_line.norm())
)
#%% # fit eraser
eraser = ConceptEraser.fit(X_t, Y_t)
X_ = eraser(X_t)
X_ablated = mean_ablate_km_component(
    X_t.unsqueeze(1), (0)
).squeeze(1)
#%% # LR learns nothing after LEACE (in-sample)
null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(
    X_.cpu().detach().numpy(), Y
)
print(null_lr.score(X_, Y_t))
null_beta = torch.from_numpy(null_lr.coef_[0])
print(null_beta.norm(p=torch.inf))
assert null_beta.norm(p=torch.inf) < 1e-4
#%% # LR learns nothing after ablation (in-sample) 
null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(
    X_ablated.cpu().detach().numpy(), Y
)
print(null_lr.score(X_ablated, Y_t))
# null_beta = torch.from_numpy(null_lr.coef_[0])
# print(null_beta.norm(p=torch.inf))
# assert null_beta.norm(p=torch.inf) < 1e-4

# %% # test LEACE / LR out-of-sample
Y_rep = einops.repeat(Y_t, "b -> (b s)", s=seq_len)
for layer in range(layers):
    layer_cache: Float[Tensor, "batch pos d_model"] = clean_cache[
        utils.get_act_name('resid_post', layer)
    ]
    layer_flat: Float[Tensor, "batch_pos d_model"] = einops.rearrange(
        layer_cache, "b p d -> (b p) d"
    )
    layer_lr_pre = LogisticRegression(max_iter=1000, tol=0.0).fit(
        layer_flat.cpu().detach().numpy(), Y_rep
    )
    layer_score_pre = layer_lr_pre.score(layer_flat, Y_rep)
    assert layer_score_pre > 0.8

    layer_erased: Float[Tensor, "batch_pos d_model"] = eraser(layer_flat)
    layer_lr_post = LogisticRegression(max_iter=1000, tol=0.0).fit(
        layer_erased.cpu().detach().numpy(), Y_rep
    )
    layer_score_post = layer_lr_post.score(layer_erased, Y_rep)
    assert layer_score_post < 0.6
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
example_index = 1
example_string = model.to_string(clean_tokens[example_index])
example_answer = model.to_str_tokens(answer_tokens[example_index])[0]
#%%
def stack_name_filter(name: str) -> bool:
    return name.endswith('result') or name.endswith('z') or name.endswith('_scale')
#%%
def extract_sentiment_layer_pos(
    cache: ActivationCache,
    centre_residuals: bool = True,
) -> Float[Tensor, "layer pos"]:
    dots: Float[Tensor, "layer pos"] = torch.empty((layers, seq_len))
    for layer in range(layers):
        act: Float[Tensor, "batch pos d_model"] = cache[
            utils.get_act_name('resid_post', layer)
        ]
        if centre_residuals:
            act -= einops.reduce(
                act, "b p d -> 1 p d", reduction="mean"
            ) # centre residual stream vectors
        dots[layer] = torch.einsum(
            "b p d, b d -> p", act, sentiment_directions
        ) / batch_size
    return dots


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
clean_logit_diff = get_logit_diff(clean_logits, answer_tokens)
print('clean logit diff', clean_logit_diff)
#%%
def ablation_metric(
    logits: Float[Tensor, "batch pos vocab"],
) -> float:
    zero = torch.tensor(0.0, device=device)
    logit_diffs: Float[Tensor, "batch"] = get_logit_diff(
        logits, answer_tokens, per_prompt=True
    ).squeeze(1)
    return (
        torch.max(logit_diffs, zero).mean() / 
        clean_logit_diff
    ).cpu().detach().numpy()
#%%
#%%
results_dict = {'base': ablation_metric(clean_logits)}
#%%
# base_sentiment = extract_sentiment_layer_pos(
#     base_cache,
# )
# #%%
# fig = px.imshow(
#     base_sentiment.cpu().detach().numpy(),
#     x=example_prompt_indexed,
#     labels={'x': 'Position', 'y': 'Layer'},
#     title='Baseline sentiment',
#     color_continuous_scale="RdBu",
#     color_continuous_midpoint=0,
# )
# fig.show()
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
    multiplier: float = 1.0,
):
    assert 'hook_resid_post' in hook.name
    if layer is not None and hook.layer() != layer:
        return input
    return mean_ablate_km_component(
        input, tokens=tokens, multiplier=multiplier
    )
    

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
    # all_layer_linear = partial(
    #     linear_hook_base,
    #     tokens=np.arange(len(example_prompt)),
    # ),

    # layer_0_linear_0_2 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=0.2,
    # ),
    # layer_0_linear_0_4 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=0.4,
    # ),
    # layer_0_linear_0_6 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=0.6,
    # ),
    # layer_0_linear_0_8 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=0.8,
    # ),
    # layer_0_linear_1_0 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=1.0,
    # ),
    # layer_0_linear_1_2 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=1.2,
    # ),
    # layer_0_linear_1_4 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=1.4,
    # ),
    # layer_0_linear_1_6 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=1.6,
    # ),
    # layer_0_linear_1_8 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=1.8,
    # ),
    # layer_0_linear_2_0 = partial(
    #     linear_hook_base,
    #     layer=0,
    #     tokens=np.arange(len(example_prompt)),
    #     multiplier=2.0,
    ),
)
#%%
for experiment_name, experiment_hook in experiments.items():
    model.reset_hooks()
    model.add_hook(
        hook_name_filter,
        experiment_hook
    )
    test_logits, test_cache = model.run_with_cache(
        clean_tokens, 
        prepend_bos=False,
        names_filter=lambda name: name.endswith('resid_post'),
    )
    test_cache.to(device)
    test_logit_diffs = get_logit_diff(
        test_logits, answer_tokens, per_prompt=True
    )
    test_metric = ablation_metric(test_logits)
    results_dict[experiment_name] = test_metric
    print(experiment_name, test_metric)
    if experiment_name == 'all_layer_linear':
        utils.test_prompt(
            example_string, example_answer, model, 
            prepend_space_to_answer=False,
            prepend_bos=False,
            top_k=top_k,
        )
        # fig = px.histogram(
        #     x=test_logit_diffs.squeeze().cpu().detach().numpy(),
        #     title=f'Logit difference distribution for {experiment_name}',
        #     nbins=len(clean_tokens),
        #     color=answer_tokens[:, 0, 0] == answer_tokens[0, 0, 0],
        #     labels={'x': 'Logit difference', 'color': 'Positive sentiment?'},
        # )
        # fig.show()
    # test_sentiment = extract_sentiment_layer_pos(test_cache)
    # fig = px.imshow(
    #     test_sentiment.cpu().detach().numpy(),
    #     labels={'x': 'Position', 'y': 'Layer'},
    #     x=example_prompt_indexed,
    #     title=f'Sentiment on {experiment_name}',
    #     color_continuous_scale="RdBu",
    #     color_continuous_midpoint=0,
    # )
    # fig.show()

#%%
px.bar(
    pd.Series(results_dict).reset_index().rename(
        columns={'index': 'experiment', 0: 'metric'}
    ),
    x='experiment',
    y='metric',
    title='Logit difference metric by experiment'
)
#%%

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