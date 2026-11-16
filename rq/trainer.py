import logging
import random

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

try:
    from .utils import ensure_dir,set_color,get_local_time,delete_file
except ImportError:  # Original script entry point: python rqvae.py
    from utils import ensure_dir,set_color,get_local_time,delete_file
import os

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)
        self.global_step = 0
        self.start_epoch = 0
        self.train_data = None
        self.last_epoch_metrics = []

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        self.last_epoch_metrics = []
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            if getattr(self.model, 'geo_enabled', False):
                out, rq_loss, indices, aux = self.model(data, return_aux=True)
            else:
                out, rq_loss, indices = self.model(data)
                aux = None
            base_loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            loss = base_loss if aux is None else base_loss + self.model.geo_weight * aux['geo_loss']
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            metrics = {'step': self.global_step, 'epoch': epoch_idx,
                       'loss': loss.item(), 'reconstruction': loss_recon.item(),
                       'quantization': rq_loss.item(), 'base': base_loss.item(),
                       'geo': 0.0, 'weighted_geo': 0.0}
            if aux is not None:
                metrics.update(geo=aux['geo_loss'].item(),
                               weighted_geo=self.model.geo_weight * aux['geo_loss'].item(),
                               geo_levels=aux['geo_level_losses'].detach().cpu().tolist(),
                               residual_rms=aux['geo_residual_rms'].cpu().tolist(),
                               geo_scales=aux['geo_scales'].cpu().tolist())
            self.last_epoch_metrics.append(metrics)
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "resume_boundary": "epoch",
            "geo_config": getattr(self.model, 'geo_config', None),
            "geo_rng": self.model.georeg.get_rng_state() if hasattr(self.model, 'georeg') else None,
            "vq_initted": [v.initted for v in self.model.rq.vq_layers],
            "rng": {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
            "loader_generator": self._generator_state(self.train_data),
            "sampler_generator": self._generator_state(getattr(self.train_data, 'sampler', None)),
            "loader_sampler_generator_shared": (
                getattr(self.train_data, 'generator', None) is
                getattr(getattr(self.train_data, 'sampler', None), 'generator', None)),
            "loader_config": None if self.train_data is None else {
                'batch_size': self.train_data.batch_size,
                'drop_last': self.train_data.drop_last,
                'num_workers': self.train_data.num_workers,
                'persistent_workers': self.train_data.persistent_workers,
                'dataset_length': len(self.train_data.dataset),
                'sampler_type': type(self.train_data.sampler).__qualname__,
            },
            "budget_ledger": os.environ.get('GEOSID_GPU_BUDGET_LEDGER'),
        }
        temporary_path = ckpt_path + '.tmp'
        torch.save(state, temporary_path, pickle_protocol=4)
        os.replace(temporary_path, ckpt_path)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    @staticmethod
    def _generator_state(owner):
        generator = getattr(owner, 'generator', None)
        return generator.get_state() if generator is not None else None

    def load_checkpoint(self, path, train_data=None):
        """Restore trusted local checkpoints at epoch boundaries only.

        Legacy weights load strictly, but lack the state needed for exact resume.
        New checkpoints need the same data/configuration and generator topology.
        Both dedicated and global-default DataLoader RNG streams are supported.
        Exact worker recovery requires deterministic dataset transforms; workers'
        internal RNG state is not saved. Local research uses num_workers=0.
        No permutation/cursor is saved, so arbitrary-step exact resume is unsupported.
        """
        # Deserialize on CPU: AdamW's noncapturable/nonfused scalar `step`
        # normally lives there, and optimizer.load_state_dict deliberately does
        # not relocate it. The model copies weights to its existing device and
        # the optimizer relocates moment tensors according to parameter policy.
        # CPU deserialization also avoids transferring host RNG states to CUDA.
        state = torch.load(path, map_location='cpu', weights_only=False)
        boundary = state.get('resume_boundary')
        if boundary not in (None, 'epoch'):
            raise ValueError('Only completed epoch-boundary checkpoints support resume')
        exact = boundary == 'epoch'
        config = state.get('geo_config')
        if config is not None and config != getattr(self.model, 'geo_config', None):
            raise ValueError('Resume model GeoReg configuration differs from checkpoint')
        flags = state.get('vq_initted', [True] * len(self.model.rq.vq_layers))
        if (not isinstance(flags, (list, tuple)) or
                len(flags) != len(self.model.rq.vq_layers) or
                any(not isinstance(flag, bool) for flag in flags)):
            raise ValueError('Invalid VQ initialization flags')
        owners = [('loader_generator', train_data),
                  ('sampler_generator', getattr(train_data, 'sampler', None))]
        rng = state.get('rng')
        if exact:
            required = {'state_dict', 'optimizer', 'scheduler', 'epoch', 'global_step',
                        'rng', 'vq_initted', 'geo_config', 'geo_rng',
                        'loader_generator', 'sampler_generator', 'loader_config',
                        'loader_sampler_generator_shared'}
            missing = required - state.keys()
            if missing:
                raise ValueError(f'Incomplete epoch checkpoint; missing {sorted(missing)}')
            if train_data is None:
                raise ValueError('Exact resume requires the training DataLoader')
            if not isinstance(rng, dict) or not {'python', 'numpy', 'torch', 'cuda'} <= rng.keys():
                raise ValueError('Incomplete epoch checkpoint RNG state')
            if (not isinstance(state['global_step'], int) or state['global_step'] < 0 or
                    not isinstance(state['epoch'], int) or state['epoch'] < -1):
                raise ValueError('Invalid completed epoch or global step')
            if state['scheduler'] is None or config is None:
                raise ValueError('Incomplete epoch checkpoint scheduler or GeoReg configuration')
            if hasattr(self.model, 'georeg') and state['geo_rng'] is None:
                raise ValueError('Exact resume requires the independent GeoReg RNG state')
            if getattr(train_data, 'persistent_workers', False):
                raise ValueError('Exact resume does not support persistent worker RNG state')
            # Standard EmbDataset is deterministic. For arbitrary stochastic
            # worker datasets, no worker RNG checkpoint contract is provided.
            if getattr(train_data, 'num_workers', 0) > 0:
                self.logger.warning('Exact worker resume requires deterministic dataset transforms; '
                                    'worker-internal RNG state is not checkpointed')
            # LambdaLR does not serialize its scheduling function. Its defining
            # settings must match, in addition to restoring its numeric state.
            saved_args = state.get('args')
            for name in ('learner', 'lr_scheduler_type', 'epochs', 'warmup_epochs',
                         'batch_size', 'dropout_prob', 'bn', 'num_workers'):
                if (hasattr(saved_args, name) and hasattr(self.args, name) and
                        getattr(saved_args, name) != getattr(self.args, name)):
                    raise ValueError(f'Exact resume configuration differs: {name}')
            for key, owner in owners:
                saved = state[key]
                generator = getattr(owner, 'generator', None)
                if (saved is None) != (generator is None):
                    raise ValueError(f'Exact resume requires matching {key} (dedicated versus global RNG)')
                if saved is not None:
                    if not torch.is_tensor(saved):
                        raise ValueError(f'Invalid {key} state')
                    # Validate without mutating the active loader generator.
                    torch.Generator(device='cpu').set_state(saved.cpu())
            shared = (getattr(train_data, 'generator', None) is
                      getattr(getattr(train_data, 'sampler', None), 'generator', None))
            if shared != state['loader_sampler_generator_shared']:
                raise ValueError('Exact resume requires matching loader/sampler generator sharing')
            saved_loader = state.get('loader_config')
            if saved_loader is None:
                raise ValueError('Exact resume requires saved DataLoader configuration')
            for name in ('batch_size', 'drop_last', 'num_workers', 'persistent_workers'):
                if saved_loader[name] != getattr(train_data, name, None):
                    raise ValueError(f'Exact resume DataLoader configuration differs: {name}')
            if saved_loader['dataset_length'] != len(train_data.dataset):
                raise ValueError('Exact resume dataset length differs')
            if saved_loader['sampler_type'] != type(train_data.sampler).__qualname__:
                raise ValueError('Exact resume sampler type differs')

        # Validate configuration/completeness before mutating model or RNG state.
        self.model.load_state_dict(state['state_dict'], strict=True)
        if config is None and hasattr(self.model, 'geo_mode'):
            self.model.geo_mode, self.model.geo_weight = 'none', 0.0
        for quantizer, flag in zip(self.model.rq.vq_layers, flags):
            quantizer.initted = flag
        self.optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state.get('global_step', 0)
        self.start_epoch = state['epoch'] + 1
        self.best_loss = state.get('best_loss', np.inf)
        self.best_collision_rate = state.get('best_collision_rate', np.inf)
        self.train_data = train_data
        if state.get('geo_rng') is not None:
            self.model.georeg.set_rng_state(state['geo_rng'].cpu())
        for key, owner in owners:
            saved = state.get(key)
            if saved is not None:
                generator = getattr(owner, 'generator', None)
                if generator is None:
                    raise ValueError(f'Exact resume requires {key}')
                generator.set_state(saved.cpu())
        if rng is not None:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            if rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
        certified_exact = exact and getattr(train_data, 'num_workers', 0) == 0
        if not exact:
            self.logger.warning('Legacy checkpoint loaded; continuation is NOT exact training resume')
        elif not certified_exact:
            self.logger.warning('Epoch state restored, but exact multiworker continuation is not certified; '
                                'use num_workers=0 for the tested exact-resume contract')
        return {'exact_epoch_resume': certified_exact, 'start_epoch': self.start_epoch,
                'global_step': self.global_step}

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0
        self.train_data = data

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

            if getattr(self.args, 'save_last', False):
                self._save_checkpoint(epoch_idx, ckpt_file='last_epoch.pth')



        return self.best_loss, self.best_collision_rate




