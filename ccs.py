#%%
import einops
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from utils.prompts import get_dataset
from torch.utils.data import Dataset, DataLoader
from jaxtyping import Float, Int, Bool
from dlk.utils import get_all_hidden_states, save_generations, get_parser
#%%
loader_batch_size = 1
device = torch.device('cuda')
MODEL_NAME = "gpt2-small" # FIXME: "EleutherAI/pythia-1.4b"
model_type = "decoder"
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device=device,
)
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
    pos_tokens[:, 0] == answer_tokens[:, 0]
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
# create a Dataloader
class PromptDataset(Dataset):
    def __init__(self, neg_tokens, pos_tokens, neg_prompts, pos_prompts, gt_labels, truncated):
        self.neg_tokens = neg_tokens
        self.pos_tokens = pos_tokens
        self.neg_prompts = neg_prompts
        self.pos_prompts = pos_prompts
        self.gt_labels = gt_labels
        self.truncated = truncated

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
#%%
parser = get_parser()
args = parser.parse_args([
    "--model_name", MODEL_NAME,
    "--device", str(device),
    "--layer", "-1",
    # "--all_layers",
    "--num_examples", "1000",
])
#%%
# Get the hidden states and labels
print("Generating hidden states")
neg_hs, pos_hs, y = get_all_hidden_states(
    model, dataloader, layer=args.layer, all_layers=args.all_layers, 
    token_idx=args.token_idx, model_type=model_type, 
    use_decoder=args.use_decoder,
)
#%%
# Save the hidden states and labels
print("Saving hidden states")
save_generations(neg_hs, args, generation_type="negative_hidden_states")
save_generations(pos_hs, args, generation_type="positive_hidden_states")
save_generations(y, args, generation_type="labels")
#%%