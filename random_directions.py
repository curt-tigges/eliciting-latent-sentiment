#%%
import torch
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
from utils.store import save_array
# %%
MODELS = [
    'gpt2-small',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-410m',
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
]
torch.random.manual_seed(42)
for model in tqdm(MODELS):
    model = HookedTransformer.from_pretrained(model)
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    for layer in range(n_layers + 1):
        random_direction = torch.randn(d_model)
        random_direction /= random_direction.norm()
        save_array(random_direction, f'random_direction_layer{layer:02d}', model)
#%%
