import argparse
import copy
from einops import rearrange
import json
import numpy as np
import os
import time
import plotly.offline as off
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from typing import Union, Tuple
import wandb
from dlk.utils import (
    load_all_generations, MLPProbe, LatentKnowledgeMethod, args_to_filename
)


def clean_name(s: str):
    return s.lower().replace('-', '_').replace('/', '_')


def plot_feature_importance(
    vec: Union[np.ndarray, torch.Tensor], label: str, a: argparse.Namespace,
):
    if a.plot_dir is None:
        return
    model_name = clean_name(a.model_name)
    data_name = clean_name(a.dataset_name)
    label_clean = clean_name(label)
    if isinstance(vec, torch.Tensor):
        vec = vec.cpu().detach().numpy()
    fig = px.histogram(
        x=vec, 
        title=f'{label} feature importance distribution',
    )
    fig.update_layout({
        'title_x': 0.5,
    })
    if not os.path.exists(a.plot_dir):
        os.mkdir(a.plot_dir)
    off.plot(
        fig, 
        filename=os.path.join(
            a.plot_dir, 
            f'{model_name}_{data_name}_{label_clean}_feature_importance.html'
        ),
        auto_open=False,
    )
    

def save_eval(val, kind, reg, partition, args):
    """
    Saves the evaluations to the eval file.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()
    elif args.verbose:
        print(
            f'Evaluated {kind}={val} for model={args.model_name}, '
            f'reg={reg}, partition={partition}'
        )
    if args.eval_path is None:
        return
    key = (
        args_to_filename(args) + 
        '__kind_' + kind + 
        '__regression_' + reg + 
        '__partition_' + partition
    )
    if os.path.isfile(args.eval_path):
        with open(args.eval_path, 'r') as f:
            eval_d = json.load(f)
    else:
        eval_d = dict()
    eval_d[key] = val
    with open(args.eval_path, 'w') as f:
        f.write(json.dumps(eval_d))


def split_train_test(neg_hs, pos_hs, y):
    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    # Merge the layer and hidden dims
    neg_hs = rearrange(neg_hs, 'n h l -> n (h l)')
    pos_hs = rearrange(pos_hs, 'n h l -> n (h l)')
    if neg_hs.shape[1] == 1:  
        # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)
    # Very simple train/test split 
    # using the fact that the data is already shuffled
    neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
    pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
    y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]
    return (
        neg_hs_train, neg_hs_test,
        pos_hs_train, pos_hs_test,
        y_train, y_test
    )

def fit_lr(
    neg_hs_train: np.ndarray, pos_hs_train: np.ndarray, 
    neg_hs_test: np.ndarray, pos_hs_test: np.ndarray, 
    y_train: np.ndarray, y_test: np.ndarray, 
    args: argparse.Namespace,
):
    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train  
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(
        class_weight="balanced",
        random_state=args.seed,
        max_iter=args.lr_max_iter,
        solver=args.lr_solver,
        C=args.lr_inv_reg,
    )
    if args.verbose:
        n, p = x_train.shape
        print(
            f'Fitting LR with n={n}, p={p}, C={lr.C}, '
            f'max_iter={lr.max_iter}, random_state={lr.random_state}'
        )
    lr.fit(x_train, y_train)
    lr_train_acc = lr.score(x_train, y_train)
    lr_train_conf = lr.predict_proba(x_train)[
        np.arange(len(y_train)), y_train
    ]
    lr_test_acc = lr.score(x_test, y_test)
    lr_test_conf = lr.predict_proba(x_test)[
        np.arange(len(y_test)), y_test
    ]
    
    save_eval(
        lr_train_acc, kind='accuracy', reg='lr', partition='train', args=args
    )
    save_eval(
        lr_train_conf, kind='confidence', reg='lr', partition='train', args=args
    )
    save_eval(
        len(y_train), kind='n_samples', reg='lr', partition='train', args=args
    )
    save_eval(
        x_train.shape[1], kind='n_features', reg='lr', partition='train', 
        args=args
    )
    save_eval(
        lr_test_acc, kind='accuracy', reg='lr', partition='test', args=args
    )
    save_eval(
        lr_test_conf, kind='confidence', reg='lr', partition='train', args=args
    )
    save_eval(
        len(y_test), kind='n_samples', reg='lr', partition='test', args=args
    )
    save_eval(
        x_test.shape[1], kind='n_features', reg='lr', partition='test', 
        args=args
    )
    lr_fi = (x_train.std(0) * lr.coef_).squeeze()
    plot_feature_importance(lr_fi, 'LR', args)
    return lr_train_acc, lr_test_acc


class CCS(LatentKnowledgeMethod):
    def __init__(
        self, 
        neg_hs_train: torch.Tensor, 
        pos_hs_train: torch.Tensor, 
        y_train: torch.Tensor,
        neg_hs_test: torch.Tensor, 
        pos_hs_test: torch.Tensor, 
        y_test: torch.Tensor,
        nepochs: int = 1000, 
        ntries: int = 10, 
        seed: int = 0, 
        lr: int = 1e-3, 
        batch_size: int = -1, 
        verbose: bool = False, 
        device: str = "cuda", 
        hidden_size: int = 0, 
        weight_decay: float = 0.01, 
        mean_normalize: bool = True,
        var_normalize: bool = True,
        wandb_enabled: bool = False,
        log_freq: int = 1000,
    ):
        super().__init__(
            neg_hs_train=neg_hs_train, 
            pos_hs_train=pos_hs_train, 
            y_train=y_train,
            neg_hs_test=neg_hs_test, 
            pos_hs_test=pos_hs_test, 
            y_test=y_test,
            mean_normalize=mean_normalize, 
            var_normalize=var_normalize,
            device=device,
        )

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.wandb_enabled = wandb_enabled
        self.log_freq = log_freq
        
        # probe
        self.hidden_size = hidden_size
        self.linear = hidden_size is None or (hidden_size == 0)
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)
        
        self.step_num = 0

        
    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d, self.hidden_size)
        self.probe.to(self.device)    
    

    def get_loss(
            self, p0: torch.Tensor, p1: torch.Tensor
        ) -> Tuple[float, float, float]:
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        contrast_loss = (torch.min(p0, p1)**2).mean(0)
        return consistent_loss, contrast_loss, contrast_loss + consistent_loss
    
        
    def train(self) -> float:
        """
        Does a single training run of nepochs epochs
        Returns the final loss split by type
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training
        for _ in range(self.nepochs):
            epoch_total_losses = []
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                consistent_loss, contrast_loss, total_loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_total_losses.append(total_loss.detach().cpu().item())

                if self.wandb_enabled:
                    log_d = {
                        'batch_consistent_loss': consistent_loss,
                        'batch_contrast_loss': contrast_loss,
                        'batch_total_loss': total_loss,
                    }
                    wandb.log(log_d, step=self.step_num)
                if self.wandb_enabled and self.step_num % self.log_freq == 0:
                    wandb.log({
                        'train_accuracy': self.get_train_acc(best=False)[0],
                        'test_accuracy': self.get_test_acc(best=False)[0],
                    }, step=self.step_num)
                self.step_num += 1
                        
        return (
            sum(epoch_total_losses) / len(epoch_total_losses)
        )
    
    def repeated_train(self):
        torch.manual_seed(self.seed)
        self.step_num = 0
        best_loss = np.inf
        for _ in range(self.ntries):
            self.initialize_probe()
            total_loss = self.train()
            if total_loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = total_loss
                if self.wandb_enabled:
                    wandb.log({
                        'best_train_accuracy': self.get_train_acc(best=True)[0],
                        'best_test_accuracy': self.get_test_acc(best=True)[0],
                    }, step=self.step_num)
        return best_loss



def fit_ccs(
    neg_hs_train, pos_hs_train, y_train,
    neg_hs_test, pos_hs_test, y_test,
    args,
):
    ccs = CCS(
        neg_hs_train=neg_hs_train, 
        pos_hs_train=pos_hs_train, 
        y_train=y_train,
        neg_hs_test=neg_hs_test, 
        pos_hs_test=pos_hs_test, 
        y_test=y_test,
        nepochs=args.nepochs, ntries=args.ntries, 
        lr=args.lr, batch_size=args.ccs_batch_size, 
        verbose=args.verbose, device=args.ccs_device, 
        hidden_size=args.hidden_size,
        weight_decay=args.weight_decay, 
        var_normalize=args.var_normalize,
        wandb_enabled=args.wandb_enabled,
        log_freq=args.ccs_log_freq,
    )
    # train
    if args.verbose:
        n, p = neg_hs_train.shape
        print(f'Training CCS with n={n}, p={p}')
    t0_train = time.time()
    ccs.repeated_train()
    if args.verbose:
        print(f'Training CCS completed in {time.time() - t0_train:.1f}s')
    ccs_train_acc, ccs_train_conf = ccs.get_train_acc()
    save_eval(
        ccs_train_acc, kind='accuracy', reg='ccs', partition='train', args=args,
    )
    save_eval(
        ccs_train_conf, kind='confidence', reg='ccs', partition='train', args=args,
    )
    save_eval(
        len(y_train), kind='n_samples', reg='ccs', partition='train', args=args,
    )
    save_eval(
        neg_hs_train.shape[1], kind='n_features', reg='ccs', partition='train', 
        args=args,
    )
    ccs_test_acc, ccs_test_conf = ccs.get_test_acc()
    save_eval(
        ccs_test_acc, kind='accuracy', reg='ccs', partition='test', args=args,
    )
    save_eval(
        ccs_test_conf, kind='confidence', reg='ccs', partition='test', args=args,
    )
    save_eval(
        len(y_test), kind='n_samples', reg='ccs', partition='test', args=args,
    )
    save_eval(
        neg_hs_test.shape[1], kind='n_features', reg='ccs', partition='test', 
        args=args,
    )

    if ccs.linear:
        ccs_fi = (ccs.best_probe[0].weight * ccs.pos_hs_train.std()).squeeze()
        plot_feature_importance(ccs_fi, 'CCS', args)
    with open(f'learned_weights/{args.model_name}_ccs.npy', 'wb') as f:
        np.save(f, ccs.best_probe[0].weight.detach().cpu().numpy())
    return ccs_train_acc, ccs_test_acc


def run_eval(generation_args: argparse.Namespace, args: argparse.Namespace):
    if args.wandb_enabled:
        wandb.init(config=args)
    # load hidden states and labels
    neg_hs, pos_hs, y = load_all_generations(generation_args)
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
        args=args,
    )
    ccs_train_acc, ccs_test_acc = fit_ccs(
        neg_hs_train=neg_hs_train,
        pos_hs_train=pos_hs_train,
        neg_hs_test=neg_hs_test,
        pos_hs_test=pos_hs_test,
        y_train=y_train,
        y_test=y_test,
        args=args,
    )
    if args.wandb_enabled:
        wandb.finish()
    return (
        lr_train_acc, lr_test_acc,
        ccs_train_acc, ccs_test_acc,
    )
