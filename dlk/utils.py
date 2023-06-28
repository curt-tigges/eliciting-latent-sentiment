import os
import argparse
from typing import Union, Tuple
import yaml

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType as TT

# make sure to install promptsource, transformers, and datasets!
from promptsource.templates import DatasetTemplates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from datasets import load_dataset


GENERATION_TYPES = [
    'negative_hidden_states',
    'positive_hidden_states',
    'labels'
]
REF_ROOT = 'reference_hidden_states'


############# Model loading and result saving #############

# Map each model name to its full Huggingface name
# This is just for convenience for common models. 
# You can run whatever model you'd like.
model_mapping = {
    # smaller models
    "gpt2-s": "gpt2",
    "gpt2-m": "gpt2-medium",
    "gpt-neo-s": "EleutherAI/gpt-neo-125M",
    "opt": "facebook/opt-125m",
    "stanford-a": "stanford-crfm/alias-gpt2-small-x21",
    "stanford-b": "stanford-crfm/battlestar-gpt2-small-x49",
    "pythia-s": "EleutherAI/pythia-160m",
    "deberta-s": "microsoft/deberta-large-mnli",

    # large models
    "opt-l": "facebook/opt-1.3b",
    "gpt2-l": "gpt2-large",
    "deberta-l": "microsoft/deberta-large-mnli",
    "roberta-l": "roberta-large-mnli",
    "uqa-l": "allenai/unifiedqa-t5-large",
    "gpt-neo-l": "EleutherAI/gpt-neo-1.3B",
    "pythia-l": "EleutherAI/pythia-1.4b",

    # XL models
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa-11b": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-xxl": "microsoft/deberta-xxlarge-v2-mnli",
}


def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="T5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Which split of the dataset to use")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Which prompt to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument('--seed', type=int, default=0, help="Seed to ensure determinism")
    # which hidden states we extract
    parser.add_argument("--use_decoder", action="store_true", help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument("--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states")
    parser.add_argument("--verbose", action="store_true")
    return parser


def load_model(model_name, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    with open('model_mapping.yaml', 'r') as f:
        model_mapping = yaml.load(f, Loader=yaml.FullLoader)
    if model_name in model_mapping:
        # use a nickname for our models
        full_model_name = model_mapping[model_name]
    else:
        # if you're trying a new model, make sure it's the full name
        full_model_name = model_name

    # use the right automodel, and get the corresponding model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "decoder"
    
        
    # specify model_max_length (the max token length) to be 512 to ensure that padding works 
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type


def args_to_filename(args: Union[dict, argparse.Namespace]):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    exclude_keys = [
        "save_dir", "cache_dir", "device", "verbose", "eval_path",
    ]
    sorted_keys = sorted([k for k in args.keys() if k not in exclude_keys])
    sorted_values = [
        args[k].replace('/', '_') 
        if isinstance(args[k], str) 
        else args[k]
        for k in sorted_keys
    ]
    return "__".join([
        '{}_{}'.format(k, v) 
        for k, v in zip(sorted_keys, sorted_values)
    ])


def generations_filename(args, generation_type):
    return generation_type + "__" + args_to_filename(args) + ".npy".format(generation_type)


def save_generations(
        generation: np.ndarray, args: argparse.Namespace, generation_type: str,
    ):
    """
    Input: 
        generation: 
            numpy array (e.g. hidden_states or labels) to save
        args: 
            arguments used to generate the hidden states. 
            This is used for the filename to save to.
        generation_type: 
            one of "negative_hidden_states" or "positive_hidden_states" or "labels"

    Saves the generations to an appropriate directory.
    """
    assert not np.isnan(generation).any()
    filename = generations_filename(args, generation_type)
    # create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save
    file_path = os.path.join(args.save_dir, filename)
    if args.verbose:
        print(f'Saving {generation.shape} array to {file_path}')
    np.save(file_path, generation)


def load_single_generation(args, generation_type="hidden_states"):
    filename = generations_filename(args, generation_type)
    loaded = np.load(os.path.join(args.save_dir, filename))
    assert not np.isnan(loaded).any()
    return loaded


def load_all_generations(args):
    # load all the saved generations: neg_hs, pos_hs, and labels
    neg_hs = load_single_generation(args, generation_type="negative_hidden_states")
    pos_hs = load_single_generation(args, generation_type="positive_hidden_states")
    labels = load_single_generation(args, generation_type="labels")
    return neg_hs, pos_hs, labels


def check_generations_exist(args):
    for gen_type in GENERATION_TYPES:
        filename = generations_filename(
            args, generation_type=gen_type
        )
        path = os.path.join(args.save_dir, filename)
        if not os.path.isfile(path):
            return False
    return True


############# Data #############
class ContrastDataset(Dataset):
    """
    Given a dataset and tokenizer (from huggingface), along with 
    a collection of prompts for that dataset from promptsource and 
    a corresponding prompt index, 
    returns a dataset that creates contrast pairs using that prompt
    
    Truncates examples larger than max_len, which can mess up 
    contrast pairs, so make sure to only give it examples that won't be truncated.
    """
    def __init__(
        self, raw_dataset, tokenizer, all_prompts, prompt_idx, 
        model_type="encoder_decoder", use_decoder=False, device="cuda",
        seed=None,
):
        np.random.seed(seed=seed)
        self.idxs_to_flip = np.random.randint(2, size=len(raw_dataset))
        # data and tokenizer
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        
        # for formatting the answers
        self.model_type = model_type
        self.use_decoder = use_decoder
        if self.use_decoder:
            assert self.model_type != "encoder"

        # prompt
        prompt_name_list = list(all_prompts.name_to_id_mapping.keys())
        self.prompt = all_prompts[prompt_name_list[prompt_idx]]

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, nl_prompt):
        """
        Tokenize a given natural language prompt
        (from after applying self.prompt to an example)
        
        For encoder-decoder models, we can either:
        (1) feed both the question and answer to the encoder, 
            creating contrast pairs using the encoder hidden states
            (which uses the standard tokenization, but also passes the empty string to the decoder), 
        or
        (2) feed the question the encoder and the answer to the decoder, 
        creating contrast pairs using the decoder hidden states
        
        If self.decoder is True we do (2), otherwise we do (1).
        """
        # get question and answer from prompt
        question, answer = nl_prompt
        
        # tokenize the question and answer 
        # (depending upon the model type and whether self.use_decoder is True)
        if self.model_type == "encoder_decoder":
            input_ids = self.get_encoder_decoder_input_ids(question, answer)
        elif self.model_type == "encoder":
            input_ids = self.get_encoder_input_ids(question, answer)
        else:
            input_ids = self.get_decoder_input_ids(question, answer)
        
        # get rid of the batch dimension since this will be added by the Dataloader
        if input_ids["input_ids"].shape[0] == 1:
            for k in input_ids:
                input_ids[k] = input_ids[k].squeeze(0)

        return input_ids


    def get_encoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models; standard formatting.
        """
        combined_input = question + " " + answer 
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids


    def get_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-only models.
        This is the same as get_encoder_input_ids except that 
        we add the EOS token at the end of the input (which apparently can matter)
        """
        combined_input = question + " " + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer(combined_input, truncation=True, padding="max_length", return_tensors="pt")
        return input_ids


    def get_encoder_decoder_input_ids(self, question, answer):
        """
        Format the input ids for encoder-decoder models.
        There are two cases for this, depending upon whether we want to use 
        the encoder hidden states or the decoder hidden states.
        """
        if self.use_decoder:
            # feed the same question to the encoder but different answers to the decoder to construct contrast pairs
            input_ids = self.tokenizer(question, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer(answer, truncation=True, padding="max_length", return_tensors="pt")
        else:
            # include both the question and the answer in the input for the encoder
            # feed the empty string to the decoder (i.e. just ignore it -- but it needs an input or it'll throw an error)
            input_ids = self.tokenizer(question, answer, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
        
        # move everything into input_ids so that it's easier to pass to the model
        input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
        input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]

        return input_ids


    def __getitem__(self, index):
        # get the original example
        data = self.raw_dataset[int(index)]

        # Hack truthful_qa so that the true prompt is not always first
        if 'mc1_targets' in data:
            assert data['mc1_targets']['labels'][:2] == [1, 0]
            choices = data['mc1_targets']['choices'][:2]
            true_answer = self.idxs_to_flip[index]
            data['label'] = true_answer
            data['choices'] = [choices[true_answer], choices[1 - true_answer]]
            data.pop('mc1_targets')
            data.pop('mc2_targets')

        # get the possible labels
        # (for simplicity assume the binary case for contrast pairs)
        label_list = self.prompt.get_answer_choices_list(data)
        assert len(label_list) == 2, print(
            "Make sure there are only two possible answers! "
            "Actual number of answers:", 
            label_list
        )

        # reconvert to dataset format but with fake/candidate labels to 
        # create the contrast pair
        true_answer = data["label"]
        neg_example = {k: v if k != 'label' else 0 for k, v in data.items() }
        pos_example = {k: v if k != 'label' else 1 for k, v in data.items() }

        # construct contrast pairs by answering the prompt with the two different possible labels
        # (for example, label 0 might be mapped to "no" and label 1 might be mapped to "yes")
        neg_prompt, pos_prompt = self.prompt.apply(neg_example), self.prompt.apply(pos_example)

        # tokenize
        neg_ids, pos_ids = self.encode(neg_prompt), self.encode(pos_prompt)

        # verify these are different (e.g. tokenization didn't cut off the difference between them)
        if self.use_decoder and self.model_type == "encoder_decoder":
            truncated = (
                neg_ids["decoder_input_ids"] == 
                pos_ids["decoder_input_ids"]
            ).all()
        else:
            truncated = (neg_ids["input_ids"] == pos_ids["input_ids"]).all()

        # return the tokenized inputs, the text prompts, and the true label
        return neg_ids, pos_ids, neg_prompt, pos_prompt, true_answer, truncated
    

def get_templates(dataset_name: str) -> DatasetTemplates:
    all_prompts = DatasetTemplates(dataset_name)
    if len(all_prompts) == 0:
        yaml_path = os.path.join('templates', dataset_name, 'templates.yaml')
        with open(yaml_path, 'r') as yaml_file:
            yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        all_prompts.templates = yaml_dict[all_prompts.TEMPLATES_KEY]
        all_prompts.sync_mapping()
    assert len(all_prompts) > 0
    return all_prompts

    
def get_dataloader(
    dataset_name, split, tokenizer, prompt_idx, batch_size=16, num_examples=1000,
    seed=0, model_type="encoder_decoder", use_decoder=False, device="cuda", pin_memory=True, 
    num_workers=1,
):
    """
    Creates a dataloader for a given dataset (and its split), tokenizer, and 
    prompt index

    Takes a random subset of (at most) num_examples samples from 
    the dataset that are not truncated by the tokenizer.
    """
    # load the raw dataset
    if '/' in dataset_name:
        dataset_name, config_name = dataset_name.split('/')
    else:
        config_name = None
    ds = load_dataset(dataset_name, name=config_name)
    if split == 'test' and (split not in ds or dataset_name == 'piqa'):
        # Some datasets are missing a test partition
        # The piqa dataset has a bug in the test partition
        split = 'validation'
    raw_dataset = ds[split]

    # load all the prompts for that dataset
    all_prompts = get_templates(dataset_name=dataset_name)

    # create the ConstrastDataset
    contrast_dataset = ContrastDataset(
        raw_dataset, tokenizer, all_prompts, prompt_idx, 
        model_type=model_type, use_decoder=use_decoder, 
        device=device, seed=seed,
    )

    # get a random permutation of the indices; we'll take the first num_examples of these that do not get truncated
    np.random.seed(seed=seed)
    random_idxs = np.random.permutation(len(contrast_dataset))

    # remove examples that would be truncated (since this messes up contrast pairs)
    keep_idxs = []
    for idx in random_idxs:
        truncated = contrast_dataset[int(idx)][-1]
        if not truncated:
            keep_idxs.append(idx)
            if len(keep_idxs) >= num_examples:
                break

    # create and return the corresponding dataloader
    subset_dataset = torch.utils.data.Subset(contrast_dataset, keep_idxs)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    return dataloader


############# Hidden States #############
def get_first_mask_loc(mask, shift=False):
    """
    return the location of the first pad token for the given ids, which corresponds to a mask value of 0
    if there are no pad tokens, then return the last location
    """
    # add a 0 to the end of the mask in case there are no pad tokens
    mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)

    if shift:
        mask = mask[..., 1:]

    # get the location of the first pad token; use the fact that torch.argmax() returns the first index in the case of ties
    first_mask_loc = torch.argmax((mask == 0).int(), dim=-1)

    return first_mask_loc


def get_individual_hidden_states(
        model, batch_ids, 
        layer=None, all_layers=True, token_idx=-1, 
        model_type="encoder_decoder", use_decoder=False
    ) -> TT['b', 'h', 'l']:
    """
    Given a model and a batch of tokenized examples, 
    returns the hidden states for either 
    a specified layer (if layer is a number) or 
    for all layers (if all_layers is True).
    
    If specify_encoder is True, then
    uses "encoder_hidden_states" instead of "hidden_states".
    This is necessary for getting the encoder hidden states for 
    encoder-decoder models, but it is not necessary for 
    encoder-only or decoder-only models.
    """
    batch_size, seq_len = batch_ids['input_ids'].shape
    if use_decoder:
        assert "decoder" in model_type
        
    # forward pass
    with torch.no_grad():
        batch_ids = batch_ids.to(model.device)
        output = model(**batch_ids, output_hidden_states=True)

    # get all the corresponding hidden states (which is a tuple of length num_layers)
    if use_decoder and "decoder_hidden_states" in output.keys():
        hs_tuple = output["decoder_hidden_states"]
    elif "encoder_hidden_states" in output.keys():
        hs_tuple = output["encoder_hidden_states"]
    else:
        hs_tuple = output["hidden_states"]
    num_layers = len(hs_tuple) if all_layers else 1
    assert hs_tuple[0].shape[:2] == (batch_size, seq_len)
    d_model = hs_tuple[0].shape[-1]

    # just get the corresponding layer hidden states
    if all_layers:
        # stack along the last axis so that it's easier to consistently index the first two axes
        hs = torch.stack([h.detach().cpu() for h in hs_tuple], axis=-1)  # (bs, seq_len, dim, num_layers)
    else:
        assert layer is not None
        hs = hs_tuple[layer].unsqueeze(-1).detach().cpu()  # (bs, seq_len, dim, 1)
    assert hs.shape == (batch_size, seq_len, d_model, num_layers)
    # we want to get the token corresponding to token_idx while ignoring the masked tokens
    if token_idx == 0:
        final_hs = hs[:, 0]  # (bs, dim, num_layers)
    else:
        # if token_idx == -1, then takes the hidden states corresponding to 
        # the last non-mask tokens
        # first we need to get the first mask location for each example in the batch
        assert token_idx < 0, print(
            "token_idx must be either 0 or negative, but got", token_idx
        )
        mask = batch_ids["decoder_attention_mask"] if (
            model_type == "encoder_decoder" and use_decoder
        ) else batch_ids["attention_mask"]
        first_mask_loc = get_first_mask_loc(mask).squeeze().detach().cpu()
        final_hs = hs[
            torch.arange(hs.size(0)), first_mask_loc+token_idx
        ]  # (bs, dim, num_layers)
    assert final_hs.shape == (batch_size, d_model, num_layers)
    return final_hs


def get_all_hidden_states(
        model, dataloader, layer=None, all_layers=True, token_idx=-1, 
        model_type="encoder_decoder", use_decoder=False
    ) -> Tuple[TT['b', 'h', 'l'], TT['b', 'h', 'l'], TT['b', 'h', 'l']]:
    """
    Given a model, a tokenizer, and a dataloader, returns the 
    hidden states (corresponding to a given position index) in 
    all layers for all examples in the dataloader,
    along with the average log probs corresponding to the answer tokens

    The dataloader should correspond to examples with a candidate label 
    already added to each example.
    E.g. this function should be used for 
    "Q: Is 2+2=5? A: True" or 
    "Q: Is 2+2=5? A: False", but NOT for 
    "Q: Is 2+2=5? A: ".
    """
    all_pos_hs, all_neg_hs = [], []
    all_gt_labels = []

    model.eval()
    for batch in tqdm(dataloader):
        neg_ids, pos_ids, _, _, gt_label, truncated = batch

        assert not truncated

        neg_hs = get_individual_hidden_states(
            model, neg_ids, layer=layer, all_layers=all_layers, token_idx=token_idx, 
            model_type=model_type, use_decoder=use_decoder
        )
        pos_hs = get_individual_hidden_states(
            model, pos_ids, layer=layer, all_layers=all_layers, token_idx=token_idx, 
            model_type=model_type, use_decoder=use_decoder
        )
        assert neg_hs.shape[0] == dataloader.batch_size
        assert pos_hs.shape[0] == dataloader.batch_size

        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(gt_label)
    
    all_neg_hs = np.concatenate(all_neg_hs, axis=0)
    all_pos_hs = np.concatenate(all_pos_hs, axis=0)
    all_gt_labels = np.concatenate(all_gt_labels, axis=0)

    assert all_gt_labels.std() > 0, f'All labels are the same: {all_gt_labels[0]}'

    return all_neg_hs, all_pos_hs, all_gt_labels

############# CCS #############
class MLPProbe(nn.Module):
    def __init__(self, d, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(d, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)
    

class LatentKnowledgeMethod(object):

    def __init__(
        self, 
        neg_hs_train: torch.Tensor, 
        pos_hs_train: torch.Tensor, 
        y_train: torch.Tensor,
        neg_hs_test: torch.Tensor, 
        pos_hs_test: torch.Tensor, 
        y_test: torch.Tensor,
        mean_normalize: bool = True,
        var_normalize: bool = True,
        device: str = 'cuda',
    ) -> None:
        '''
        x0: negative hidden states, shape [num_examples, num_hidden_states]
        x1: positive hidden states, shape [num_examples, num_hidden_states]
        '''
        self.device = device
        self.mean_normalize = mean_normalize
        self.var_normalize = var_normalize
        self.neg_hs_train = self.normalize(neg_hs_train)
        self.pos_hs_train = self.normalize(pos_hs_train)
        self.y_train = y_train
        self.neg_hs_test = self.normalize(neg_hs_test)
        self.pos_hs_test = self.normalize(pos_hs_test)
        self.y_test = y_test
        self.d = self.neg_hs_train.shape[-1]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
        if (sigma == 0).all():
            print(f'WARN: all std devs are zero!')
        if self.mean_normalize:
            x -= mu
        if self.var_normalize:
            x /= np.where(sigma > 0, sigma, 1)
        return x
    
    def get_tensor_data(self, x0=None, x1=None):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        if x0 is None:
            x0 = self.neg_hs_train
        if x1 is None:
            x1 = self.pos_hs_train
        x0 = torch.tensor(
            x0, dtype=torch.float, requires_grad=False, device=self.device
        )
        x1 = torch.tensor(
            x1, dtype=torch.float, requires_grad=False, device=self.device
        )
        return x0, x1
    
    def get_acc(
            self, x0_val: np.ndarray, x1_val: np.ndarray, y_val: np.ndarray,
            best=True,
        ) -> Tuple[float, np.ndarray]:
        """
        Computes accuracy for the current parameters on the given test inputs

        x0_val: negative hidden states, numpy array
        x1_val: positive hidden states, numpy array
        y_val: true labels, numpy array
        """
        probe = self.best_probe if best else self.probe
        x0, x1 = self.get_tensor_data(x0_val, x1_val)
        with torch.no_grad():
            p0, p1 = probe(x0), probe(x1)
        avg_confidence = (0.5 * (p0 + (1 - p1)))[:, 0]
        avg_conf_values = avg_confidence.detach().cpu().numpy()
        predictions = (avg_conf_values > 0.5).astype(int)
        correct_mask = predictions == y_val
        acc = correct_mask.mean()
        if acc < 0.5:
            # reverse the predictions
            avg_confidence = 1 - avg_confidence
            correct_mask = ~correct_mask
            acc = 1 - acc
        conf = np.where(y_val > 0, avg_conf_values, 1 - avg_conf_values)
        return acc, conf
    
    def get_train_acc(self, best: bool = True):
        return self.get_acc(
            self.neg_hs_train, self.pos_hs_train, self.y_train, best=best,
        )
    
    def get_test_acc(self, best: bool = True):
        return self.get_acc(
            self.neg_hs_test, self.pos_hs_test, self.y_test, best=best,
        )