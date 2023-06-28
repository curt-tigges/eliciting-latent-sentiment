#%%
import argparse
import numpy as np
import einops
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from utils.prompts import get_dataset
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Float, Int, Bool
from dlk.utils import (
    get_all_hidden_states, save_generations, get_parser, 
    load_all_generations
)
from dlk.evaluate import split_train_test, fit_ccs, fit_lr
from transformers import AutoModelForCausalLM
import wandb
from utils.store import save_array, load_array
#%%
loader_batch_size = 1
device = torch.device('cuda')
MODEL_NAME = "gpt2-small" # FIXME: "EleutherAI/pythia-1.4b"
HF_NAME = "gpt2"
model_type = "decoder"
#%%
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
model.device = model.cfg.device
#%%
pos_answers = [" Positive"] #, " amazing", " good"]
neg_answers = [" Negative"] #, " terrible", " bad"]
all_prompts, answer_tokens, clean_tokens, _ = get_dataset(
    model, device, n_pairs=1, prompt_type="classification", 
    pos_answers=pos_answers, neg_answers=neg_answers,
) # FIXME: change to classification_4
answer_tokens: Int[Tensor, "batch 2"] = answer_tokens.squeeze(1)
clean_tokens.shape
#%%
possible_answers = answer_tokens[0]
possible_answers_repeated: Int[Tensor, "batch 2"] = einops.repeat(
    possible_answers, "answers -> batch answers", batch=clean_tokens.shape[0]
)
#%%
# concatenate clean_tokens and answer_tokens along new dimension
pos_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
    (clean_tokens, possible_answers_repeated[:, :1]), dim=1
)
neg_tokens: Float[Tensor, "batch q_and_a"] = torch.cat(
    (clean_tokens, possible_answers_repeated[:, -1:]), dim=1
)
gt_labels: Int[Tensor, "batch"] = (
    pos_tokens[:, -1] == answer_tokens[:, 0]
).to(torch.int64)
truncated: Bool[Tensor, "batch"] = torch.zeros(
    gt_labels.shape[0], device=device, dtype=torch.bool
)
pos_prompts = [
    [prompt, answer] 
    for prompt in all_prompts 
    for answer in pos_answers
]
neg_prompts = [
    [prompt, answer]
    for prompt in all_prompts
    for answer in neg_answers
]
assert len(pos_prompts) == len(pos_tokens)
#%%
print(clean_tokens[0])
print(pos_tokens[0])
print(gt_labels[:5])
#%%
# create a Dataloader
class PromptDataset(Dataset):
    def __init__(self, neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, truncated):
        self.neg_tokens = neg_tokens.detach()
        self.pos_tokens = pos_tokens.detach()
        self.neg_prompts = neg_prompts
        self.pos_prompts = pos_prompts
        self.gt_labels = gt_labels.detach().cpu()
        self.truncated = truncated.detach()

    def __len__(self):
        return len(self.neg_tokens)

    def __getitem__(self, idx):
        return (
            self.neg_tokens[idx],
            self.pos_tokens[idx],
            self.neg_prompts[idx],
            self.pos_prompts[idx],
            self.gt_labels[idx],
            self.truncated[idx],
        )
    
dataset = PromptDataset(
    neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, truncated
)
dataloader = DataLoader(
    dataset, batch_size=loader_batch_size, shuffle=False
)
len(dataloader)
#%%
model = AutoModelForCausalLM.from_pretrained(HF_NAME, cache_dir=None).to(device)
#%%
parser = get_parser()
argv = [
    "--model_name", MODEL_NAME,
    "--device", str(device),
    "--layer", "-1",
    # "--all_layers",
    "--num_examples", "1000",
]
gen_args = parser.parse_args(argv)
parser.add_argument("--nepochs", type=int, default=1000)
parser.add_argument("--ntries", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--ccs_log_freq", type=int, default=10)
parser.add_argument("--ccs_batch_size", type=int, default=-1)
parser.add_argument("--ccs_device", type=str, default="cuda")
parser.add_argument('--hidden_size', type=int, default=None)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mean_normalize", action=argparse.BooleanOptionalAction)
parser.add_argument("--var_normalize", action=argparse.BooleanOptionalAction)
parser.add_argument('--eval_path', type=str, default='results.json')
parser.add_argument('--plot_dir', type=str, default='plots')
parser.add_argument('--wandb_enabled', action='store_true')
parser.add_argument("--lr_max_iter", type=int, default=100)
parser.add_argument("--lr_solver", type=str, default="lbfgs")
parser.add_argument("--lr_inv_reg", type=float, default=1.0)
extra_args = [
    "--wandb_enabled"
]
eval_args = parser.parse_args(argv + extra_args)
#%%
# Get the hidden states and labels
print("Generating hidden states")
neg_hs, pos_hs, y = get_all_hidden_states(
    model, dataloader, layer=gen_args.layer, all_layers=gen_args.all_layers, 
    token_idx=gen_args.token_idx, model_type=model_type, 
    use_decoder=gen_args.use_decoder,
)
#%%
# Save the hidden states and labels
# print("Saving hidden states")
# save_generations(neg_hs, eval_args, generation_type="negative_hidden_states")
# save_generations(pos_hs, eval_args, generation_type="positive_hidden_states")
# save_generations(y, eval_args, generation_type="labels")
#%%
if eval_args.wandb_enabled:
    wandb.init(config=eval_args)
(
    neg_hs_train, neg_hs_test,
    pos_hs_train, pos_hs_test,
    y_train, y_test
) = split_train_test(neg_hs, pos_hs, y)
lr_train_acc, lr_test_acc = fit_lr(
    neg_hs_train=neg_hs_train,
    pos_hs_train=pos_hs_train,
    neg_hs_test=neg_hs_test,
    pos_hs_test=pos_hs_test,
    y_train=y_train,
    y_test=y_test,
    args=eval_args,
)
ccs_train_acc, ccs_test_acc = fit_ccs(
    neg_hs_train=neg_hs_train,
    pos_hs_train=pos_hs_train,
    neg_hs_test=neg_hs_test,
    pos_hs_test=pos_hs_test,
    y_train=y_train,
    y_test=y_test,
    args=eval_args,
)
#%%
len(neg_hs_train), len(neg_hs_test)
#%%
ccs_train_acc, ccs_test_acc
# %%
ccs_line = load_array('ccs', MODEL_NAME).squeeze(0)
ccs_line.shape
# %%
km_line = load_array('km_2c_line_embed_and_mlp0', MODEL_NAME)
rotation_direction = load_array('rotation_direction0', MODEL_NAME)
# %%
# compute cosine similarity of ccs_line and km_line
ccs_line = ccs_line / np.linalg.norm(ccs_line)
km_line = km_line / np.linalg.norm(km_line)
ccs_km_sim = np.dot(ccs_line, km_line)
ccs_km_sim
# %%
# compute cosine similarity of ccs_line and rotation_direction
ccs_line = ccs_line / np.linalg.norm(ccs_line)
rotation_direction = rotation_direction / np.linalg.norm(rotation_direction)
ccs_km_sim = np.dot(ccs_line, rotation_direction)
ccs_km_sim
# %%
