#%%
import random
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt, get_attention_mask, LocallyOverridenDefaults
import plotly.express as px
from utils.prompts import CleanCorruptedCacheResults, get_dataset, PromptType, ReviewScaffold, CleanCorruptedDataset
#%%
model = HookedTransformer.from_pretrained("EleutherAI/pythia-1.4b")
#%%
batch_size = 256
device = torch.device("cuda")
prompt_type = PromptType.TREEBANK_TEST
scaffold = ReviewScaffold.CLASSIFICATION
names_filter = lambda _: False
clean_corrupt_data: CleanCorruptedDataset = get_dataset(
    model, device, prompt_type=prompt_type, scaffold=scaffold,
)
#%%
print(len(clean_corrupt_data))
# #%%
# get_attention_mask(model.tokenizer, clean_corrupt_data.clean_tokens, prepend_bos=False).sum(axis=1)
#%%
non_pad_tokens = clean_corrupt_data.get_num_non_pad_tokens()
px.histogram(non_pad_tokens.cpu(), nbins=100)
#%%
# with LocallyOverridenDefaults(model, padding_side="left"):
patching_dataset: CleanCorruptedCacheResults = clean_corrupt_data.restrict_by_padding(
    0, 25
).run_with_cache(
    model, 
    names_filter=names_filter,
    batch_size=batch_size,
    device=device,
    disable_tqdm=True,
    center=True,
)
print(len(patching_dataset.clean_logit_diffs))
print(patching_dataset)
# %%
len(patching_dataset.clean_logit_diffs), len(clean_corrupt_data.all_prompts)
#%%
patching_dataset.clean_logit_diffs[:10]
#%%
sample_index = random.randint(0, len(clean_corrupt_data.all_prompts))
clean_corrupt_data.all_prompts[sample_index]
#%%
# With padding
test_prompt(
    model.to_string(clean_corrupt_data.clean_tokens[sample_index]), 
    model.to_string(clean_corrupt_data.answer_tokens[sample_index, 0, 0]), 
    model,
    prepend_space_to_answer=False,
)
#%%
# Without padding
test_prompt(
    clean_corrupt_data.all_prompts[sample_index], 
    model.to_string(clean_corrupt_data.answer_tokens[sample_index, 0, 0]), 
    model,
    prepend_space_to_answer=False,
)
#%%
with LocallyOverridenDefaults(model, padding_side="right"):
    test_prompt(
        model.to_string(clean_corrupt_data.clean_tokens[sample_index]), 
        model.to_string(clean_corrupt_data.answer_tokens[sample_index, 0, 0]), 
        model,
        prepend_space_to_answer=False,
    )
#%%
# Artificial left padding
with LocallyOverridenDefaults(model, padding_side="left"):
    test_prompt(
        "".join([model.tokenizer.pad_token] * 2) + clean_corrupt_data.all_prompts[sample_index], 
        model.to_string(clean_corrupt_data.answer_tokens[sample_index, 0, 0]), 
        model,
        prepend_space_to_answer=False,
        prepend_bos=True,
    )
#%%
# Artificial right padding
with LocallyOverridenDefaults(model, padding_side="right"):
    test_prompt(
        clean_corrupt_data.all_prompts[sample_index] + "".join([model.tokenizer.pad_token] * 2), 
        model.to_string(clean_corrupt_data.answer_tokens[sample_index, 0, 0]), 
        model,
        prepend_space_to_answer=False,
        prepend_bos=True,
    )
#%%
# Attention mask for right padding
with LocallyOverridenDefaults(model, padding_side="right"):
    print(get_attention_mask(
        model.tokenizer,
        model.to_tokens(
            clean_corrupt_data.all_prompts[sample_index] + 
            "".join([model.tokenizer.pad_token] * 2),
            prepend_bos=False
        ), 
        prepend_bos=True,
    ))
#%%
test_prompt(
    clean_corrupt_data.all_prompts[0], 
    model.to_string(clean_corrupt_data.answer_tokens[0, 0, 0]), 
    model,
    prepend_space_to_answer=False,
)
# %%
test_prompt(
    clean_corrupt_data.all_prompts[1], 
    model.to_string(clean_corrupt_data.answer_tokens[0, 0, 1]), 
    model,
    prepend_space_to_answer=False,
)
# %%
