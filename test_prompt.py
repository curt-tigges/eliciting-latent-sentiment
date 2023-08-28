#%%
from functools import partial
import torch
from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import test_prompt, get_act_name
from utils.store import load_array
import tqdm
import itertools
#%%
device = "cpu"
MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
)
model.name = MODEL_NAME
#%%
test_prompt("The traveller{ADV} walked to their destination. The traveller felt very".format(ADV=" excitedly"), "happy", model)
#%%
test_prompt("The traveller{ADV} walked to their destination. The traveller felt very".format(ADV=" nervously"), "sad", model)
#%%
sentiment_dir = load_array("km_2c_line_embed_and_mlp0", model)
sentiment_dir: Float[Tensor, "d_model"] = torch.tensor(sentiment_dir).to(device=device, dtype=torch.float32)
sentiment_dir /= sentiment_dir.norm()
#%%
def steering_hook(
    input: Float[Tensor, "batch pos d_model"], hook: HookPoint, coef: float, direction: Float[Tensor, "d_model"]
):
    assert 'resid_post' in hook.name
    input += coef * direction
    return input
#%%
def steer_and_test_prompt(
    coef: float,
    direction: Float[Tensor, "d_model"],
    prompt: str,
    answer: str,
    model: HookedTransformer,
    prepend_space_to_answer: bool = True,
):
    model.reset_hooks()
    hook = partial(steering_hook, coef=coef, direction=direction)
    model.add_hook(
        get_act_name("resid_post", 0),
        hook,
        dir="fwd",
    )
    test_prompt(prompt, answer, model, prepend_space_to_answer=prepend_space_to_answer)
    model.reset_hooks()
#%%
def steer_and_generate(
    coef: float,
    direction: Float[Tensor, "d_model"],
    prompt: str,
    model: HookedTransformer,
    **kwargs,
):
    model.reset_hooks()
    hook = partial(steering_hook, coef=coef, direction=direction)
    model.add_hook(
        get_act_name("resid_post", 0),
        hook,
        dir="fwd",
    )
    input = model.to_tokens(prompt)
    output = model.generate(input, **kwargs)
    model.reset_hooks()
    return model.to_string(output)
#%%
test_prompt("The movie was sumptuous. I thought it was", "great", model, prepend_space_to_answer=True)
# %%
# for coef in range(-50, -30, 2):
#     print(f"coef: {coef}")
#     steer_and_test_prompt(
#         coef,
#         sentiment_dir,
#         "I really enjoyed the movie, in fact I loved it. I thought the movie was just very",
#         "good",
#         model,
#     )
# -40 and +16 seem to be the best
# %%
# steer_and_test_prompt(
#     -40,
#     sentiment_dir,
#     "I really enjoyed the movie, in fact I loved it. I thought the movie was just very",
#     "good",
#     model,
# )
# %%
torch.manual_seed(0)
COEFS = [-10, -5, 0, 5, 10]
NUM_SAMPLES = 5
coef_dict = {c: [] for c in COEFS}
for coef, _ in tqdm(itertools.product(COEFS, NUM_SAMPLES)):
    coef_dict[coef].append(steer_and_generate(
        coef,
        sentiment_dir,
        "I really enjoyed the movie, in fact I loved it. I thought the movie was just very",
        model,
        max_new_tokens=20,
        do_sample=True,
        temperature=1.0,
        top_k=10,
    ))
# %%
coef_dict
# %%
