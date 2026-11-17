import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, holds npu_support.py
import npu_support
import argparse
import random
import torch
import numpy as np
from time import time
import logging
import json

from torch.utils.data import DataLoader

try:
    from .datasets import EmbDataset
    from .models.rqvae import RQVAE
    from .trainer import Trainer
except ImportError:
    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from trainer import Trainer

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Games/Games.emb-llama-td.npy",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--geo_mode', choices=['none', 'global', 'residual'], default='none')
    parser.add_argument('--geo_weight', type=float, default=0.0)
    parser.add_argument('--geo_nproj', type=int, default=256)
    parser.add_argument('--geo_num_frequencies', type=int, default=17)
    parser.add_argument('--geo_tmax', type=float, default=3.0)
    parser.add_argument('--geo_eps', type=float, default=1e-6)
    parser.add_argument('--geo_projection_chunk', type=int, default=64)
    parser.add_argument('--geo_seed', type=int, default=2024)
    parser.add_argument('--resume', default=None, help='Trusted epoch-boundary checkpoint')
    parser.add_argument('--save_last', action='store_true', help='Save each completed epoch for exact recovery')
    parser.add_argument('--loader_seed', type=int, default=None, help='Optional dedicated DataLoader RNG')
    parser.add_argument('--config', default=None, help='JSON defaults; explicit CLI options take precedence')

    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        with open(preliminary.config, encoding='utf-8') as handle:
            defaults = json.load(handle)
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f'Unknown configuration fields: {sorted(unknown)}')
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv=None):
    """fix the random seed"""
    args = parse_args(argv)
    data = EmbDataset(args.data_path)
    if args.geo_mode != 'none' and args.geo_weight > 0:
        if args.batch_size < 2 or len(data) < 2 or len(data) % args.batch_size == 1:
            raise ValueError('GeoReg requires at least two items in every batch; '
                             'choose a batch_size without a singleton tail. No items are dropped.')
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  **{key: value for key, value in vars(args).items() if key.startswith('geo_')},
                  )
    print(model)
    generator = None if args.loader_seed is None else torch.Generator().manual_seed(args.loader_seed)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, generator=generator)
    trainer = Trainer(args,model, len(data_loader))
    if args.resume:
        trainer.load_checkpoint(args.resume, data_loader)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)


if __name__ == '__main__':
    main()

