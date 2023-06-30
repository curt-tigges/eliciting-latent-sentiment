#%%
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple
from jaxtyping import Int, Float
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import einops
from sklearn.linear_model import LogisticRegression
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from utils.prompts import get_dataset
from utils.circuit_analysis import get_log_probs
import plotly.express as px
from utils.cache import (
    residual_sentiment_sim_by_head, residual_sentiment_sim_by_pos
)
from utils.store import load_array, save_html
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%%
torch.set_grad_enabled(False)
device = torch.device("cuda")
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
model.cfg.use_attn_result = True
#%%
heads = model.cfg.n_heads
layers = model.cfg.n_layers
#%%
km_line = load_array('km_2c_line_embed_and_mlp0', model)
km_line = torch.from_numpy(km_line).to(device, torch.float32)
km_line_unit = km_line / km_line.norm()
# #%%
# pc0 = load_array('pca_0_embed_and_mlp0', model)
# pc0 = torch.from_numpy(pc0).to(device, torch.float32)
# pc1 = load_array('pca_1_embed_and_mlp0', model)
# pc1 = torch.from_numpy(pc1).to(device, torch.float32)
#%%
# neg_log_prob_grad = load_array('derivative_log_prob', model)
# neg_log_prob_grad = torch.from_numpy(neg_log_prob_grad).to(device, torch.float32)
# grad_unit = neg_log_prob_grad / neg_log_prob_grad.norm()
#%%
rotation_direction = load_array('rotation_direction0', model)
rotation_direction = torch.from_numpy(rotation_direction).to(device, torch.float32)
torch.testing.assert_close(
    rotation_direction.norm(), torch.tensor(1.0, device=device),
    rtol=0, atol=.001
)
#%%
# ov_direction = load_array('mean_ov_direction_10_4', model)
# ov_direction = torch.from_numpy(ov_direction).to(device, torch.float32)
# ov_direction = ov_direction / ov_direction.norm()
#%%
ccs_direction = load_array('ccs', model).squeeze(0)
ccs_direction = torch.from_numpy(ccs_direction).to(device, torch.float32)
ccs_direction = ccs_direction / ccs_direction.norm()
#%%
all_prompts, answer_tokens, clean_tokens, corrupted_tokens = get_dataset(
    model, device
)
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
def compute_mean_projection(
    direction: Float[Tensor, "d_model"], 
    tokens: Iterable[int] = (adjective_token, verb_token),
) -> Float[Tensor, ""]:
    assert torch.isclose(direction.norm(), torch.tensor(1.0), rtol=0, atol=.001)
    clean_projections: Float[Tensor, "batch"] = einops.einsum(
        clean_embeddings, direction, "b s d, d -> b s"
    )
    return clean_projections[:, tokens].mean()
#%%
def mean_ablate_direction(
    input: Float[Tensor, "batch pos d_model"],
    direction: Float[Tensor, "d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    multiplier: float = 1.0,
):
    assert torch.isclose(direction.norm(), torch.tensor(1.0), rtol=0, atol=.001)
    proj = einops.einsum(input, direction, "b p d, d -> b p")
    avg = compute_mean_projection(direction, tokens)
    avg_broadcast = einops.repeat(
        avg, " -> b p", 
        b=input.shape[0], p=input.shape[1]
    )
    proj_diff: Float[Tensor, "batch pos 1"] = (
        avg_broadcast - proj
    )[:, tokens].unsqueeze(dim=-1)
    input[:, tokens, :] += multiplier * proj_diff * direction
    return input
#%%
def mean_ablate_km_component(
    input: Float[Tensor, "batch pos d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    multiplier: float = 1.0,
):
    return mean_ablate_direction(
        input, km_line_unit, tokens, multiplier
    )

#%%
def mean_ablate_pcs(
    input: Float[Tensor, "batch pos d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    multiplier: float = 1.0,
    n_components: int = 2,
):
    for i in range(n_components):
        pc = globals()[f'pca_{i}_embed_and_mlp0']
        input = mean_ablate_direction(input, pc, tokens, multiplier)
    return input
#%%
#%%
# def mean_ablate_gradient_direction(
#     input: Float[Tensor, "batch pos d_model"],
#     tokens: Iterable[int] = (adjective_token, verb_token),
#     multiplier: float = 1.0,
# ):
#     return mean_ablate_direction(
#         input, grad_unit, tokens, multiplier
#     )

#%%
def mean_ablate_rotation_direction(
    input: Float[Tensor, "batch pos d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    multiplier: float = 1.0,
):
    return mean_ablate_direction(
        input, rotation_direction, tokens, multiplier
    )
#%%
# def mean_ablate_ov_direction(
#     input: Float[Tensor, "batch pos d_model"],
#     tokens: Iterable[int] = (adjective_token, verb_token),
#     multiplier: float = 1.0,
# ):
#     return mean_ablate_direction(
#         input, ov_direction, tokens, multiplier
#     )
#%%
def mean_ablate_ccs_direction(
    input: Float[Tensor, "batch pos d_model"],
    tokens: Iterable[int] = (adjective_token, verb_token),
    multiplier: float = 1.0,
):
    return mean_ablate_direction(
        input, ccs_direction, tokens, multiplier
    )
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
X_ablated = mean_ablate_km_component(
    X_t.unsqueeze(1), (0)
).squeeze(1)
#%% # LR after ablation (in-sample) 
null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(
    X_ablated.cpu().detach().numpy(), Y
)
print('post-ablation accuracy', null_lr.score(X_ablated.cpu(), Y_t.cpu()))

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
def ablation_metric(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Int[Tensor, "batch n_pairs 2"],
) -> Tuple[float]:
    pos_mask = torch.arange(batch_size, device=device) % 2 == 0
    pos_log_probs = get_log_probs(
        logits[pos_mask, ...], answer_tokens[pos_mask, :, 0]
    ).cpu().detach().numpy()
    neg_log_probs = get_log_probs(
        logits[~pos_mask, ...], answer_tokens[~pos_mask, :, 0]
    ).cpu().detach().numpy()
    return pos_log_probs, neg_log_probs
#%%
pos_results_dict = dict()
neg_results_dict = dict()
pos_results_dict['base'], neg_results_dict['base'] = ablation_metric(
    clean_logits, answer_tokens
)
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
ABLATION = mean_ablate_rotation_direction


def linear_hook_base(
    input: Float[Tensor, "batch pos d_model"], 
    hook: HookPoint,
    tokens: Iterable[int] = (adjective_token, verb_token),
    layer: Optional[int] = None,
    multiplier: float = 1.0,
    ablation: Callable = ABLATION,
):
    assert 'hook_resid_post' in hook.name
    if layer is not None and hook.layer() != layer:
        return input
    return ablation(
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

    layer_0_linear_0_2 = partial(
        linear_hook_base,
        layer=0,
        tokens=np.arange(len(example_prompt)),
        multiplier=0.2,
    ),
    layer_0_linear_0_4 = partial(
        linear_hook_base,
        layer=0,
        tokens=np.arange(len(example_prompt)),
        multiplier=0.4,
    ),
    layer_0_linear_0_6 = partial(
        linear_hook_base,
        layer=0,
        tokens=np.arange(len(example_prompt)),
        multiplier=0.6,
    ),
    layer_0_linear_0_8 = partial(
        linear_hook_base,
        layer=0,
        tokens=np.arange(len(example_prompt)),
        multiplier=0.8,
    ),
    layer_0_linear_1_0 = partial(
        linear_hook_base,
        layer=0,
        tokens=np.arange(len(example_prompt)),
        multiplier=1.0,
    ),
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
    # ),
)
#%%
for experiment_name, experiment_hook in experiments.items():
    model.reset_hooks()
    model.add_hook(
        hook_name_filter,
        experiment_hook
    )
    test_logits = model(
        clean_tokens, 
        prepend_bos=False,
    )
    test_metric = ablation_metric(test_logits, answer_tokens)
    (
        pos_results_dict[experiment_name], 
        neg_results_dict[experiment_name]
    ) = test_metric
    print(experiment_name, test_metric)
    
#%%
for experiment_name, experiment_hook in experiments.items():
    if experiment_name == 'layer_0_linear_1_0':
        model.reset_hooks()
        model.add_hook(
            hook_name_filter,
            experiment_hook
        )
        utils.test_prompt(
            example_string, example_answer, model, 
            prepend_space_to_answer=False,
            prepend_bos=False,
            top_k=top_k,
        )

#%%
results_df = pd.Series(pos_results_dict).rename('log_prob').reset_index()
results_df['pos_neg'] = 'positive'
results_df = pd.concat(
    [results_df, pd.Series(neg_results_dict).rename('log_prob').reset_index()]
)
results_df['pos_neg'] = results_df['pos_neg'].fillna('negative')
results_df.rename(columns={'index': 'experiment'}, inplace=True)
results_df
#%%
fig = px.bar(
    results_df,
    x='experiment',
    y='log_prob',
    color='pos_neg',
    title=f'Log prob metric by experiment ({ABLATION.__name__})',
    barmode='group',
)
save_html(fig, f'log_prob_metric_by_experiment_{ABLATION.__name__}', model)
fig.show()
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
