"""
PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import re
import sys
import os
import copy
import yaml
from datetime import datetime

import lightning.pytorch as pl

import numpy as np
import torch
import torch.distributed as dist

from torch import optim
try:
    from torch.optim.swa_utils import AveragedModel
except ImportError:
    print('Failed to load SWA utils. If you want to use SWA, please use torch>=1.6')
try:
    from adamp import AdamP, SGDP
except ImportError:
    print('Failed to load adamp. Skip to load adamp')

from pcmepp.augmentation import Mixup
from pcmepp.criterions import get_criterion
from pcmepp.criterions.pcme import sample_gaussian_tensors
from pcmepp.models.encoders import get_image_encoder
from pcmepp.models.encoders import get_text_encoder

from pcmepp.evaluation import compute_matmul_sims, compute_matching_prob_sims, compute_csd_sims
from pcmepp.evaluation import eval_i2t_val, eval_t2i_val
from pcmepp.evaluation import eval_coco_test_loader, eval_coco_uncertainty

from pcmepp.dist import grad_all_gather


class PCMEPPModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        # Build Models
        self.img_enc = get_image_encoder(
            size_augment=opt.augment.img_size_augment, **opt.model)
        self.txt_enc = get_text_encoder(**opt.model)

        if opt.train.get('torch_compile'):
            now = datetime.now()
            if opt.train.torch_compile == 'max-autotune' or opt.train.torch_compile == 'reduce-overhead':
                torch.compile(self.img_enc, mode=opt.train.torch_compile)
                torch.compile(self.txt_enc, mode=opt.train.torch_compile)
            else:
                torch.compile(self.img_enc)
                torch.compile(self.txt_enc)
            print(f'torch.compile takes {datetime.now() - now}')

        self.criterion = get_criterion(**opt.criterion)

        # See `on_train_epoch_start` for mixup_fn
        self.mixup_fn = None
        self.validation_step_outputs = {}
        self.opt = opt

        self.swa_enabled = False
        self.eval_by_swa = False
        if self.opt.train.get('swa_start_epoch'):
            self.swa_enabled = True
            self.swa_iter_mode = False
            self.swa_epoch_mode = False

            swa_per_iters = self.opt.train.get('swa_per_iters')
            swa_per_epochs = self.opt.train.get('swa_per_epochs')
            is_swa_criterion = self.opt.model.get('is_probabilistic_model')

            self.swa_start_epoch = int(self.opt.train.swa_start_epoch)
            if swa_per_iters and swa_per_epochs:
                print(f'swa_per_iters and swa_per_epochs cannot be applied at the same time. Use {swa_per_iters=}')
            if swa_per_iters:
                self.swa_iter_mode = True
                self.swa_per_iters = int(swa_per_iters)
                print(f'Update SWA per {self.swa_per_iters} iterations from epoch {self.swa_start_epoch}')
            if swa_per_epochs:
                self.swa_epoch_mode = True
                self.swa_per_epochs = int(swa_per_epochs)
                print(f'Update SWA per {self.swa_per_epochs} epochs from epoch {self.swa_start_epoch}')
            self.is_swa_criterion = is_swa_criterion

        if self.swa_enabled:
            # Avoiding low GPU utilization
            self.swa_img_enc = AveragedModel(self.img_enc, device='cuda')
            self.swa_txt_enc = AveragedModel(self.txt_enc, device='cuda')
            if self.is_swa_criterion:
                self.swa_criterion = AveragedModel(self.criterion, device='cuda')

        self.save_hyperparameters()

    def on_train_start(self):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(yaml.safe_dump(self.opt))
            print(f'Train dataset size: {len(self.trainer.train_dataloader.dataset)}')
            print(f"Run Trainer with {int(os.environ.get('NSML_WORLD_SIZE', 1))} nodes!")

        if self.opt.train.get('img_backbone_reinit_layers'):
            self.img_enc.reinit(self.opt.train.img_backbone_reinit_layers)
        if self.opt.train.get('txt_backbone_reinit_layers'):
            self.txt_enc.reinit(self.opt.train.txt_backbone_reinit_layers)

    def forward_step(self, batch, mixup_fn=None, use_swa=False):
        if use_swa:
            img_enc = self.swa_img_enc
            txt_enc = self.swa_txt_enc
        else:
            img_enc = self.img_enc
            txt_enc = self.txt_enc

        if self.opt.model.precomp_enc_type == 'basic':
            images, img_lengths, captions, lengths, ids = batch
            matched = torch.eye(len(images)).to(images.device)
            img_emb = img_enc(images, img_lengths)
            if mixup_fn:
                raise NotImplementedError('MSDA is not implemented for pre-computed features')
        else:
            images, captions, lengths, img_ids, ids = batch
            matched = torch.eye(len(images)).to(images.device)

            if len(img_ids) != len(set(img_ids)):
                # NOTE: It could be rarely happend because each coco image is mapped to five captions.
                indices = {}
                seen = set()
                for idx, _id in enumerate(img_ids):
                    if _id in seen:
                        for _prev_idx in indices[_id]:
                            matched[idx][_prev_idx] = 1
                            matched[_prev_idx][idx] = 1
                    seen.add(_id)
                    indices.setdefault(_id, []).append(idx)

            if mixup_fn:
                batch_size = len(images)
                if batch_size % 2 != 0:
                    # NOTE: mixup only allows n % 2 == 0
                    images = images[:batch_size-1]
                    captions = captions[:batch_size-1]
                    lengths = lengths[:batch_size-1]
                    batch_size = len(images)
                # TODO return original images as well / mixed index
                # TODO sentence concatenation => maybe out of from this method
                images, matched, _ = mixup_fn(
                    images, torch.arange(batch_size).to(images.device))
            elif self.opt.augment.get('label_smoothing'):
                # TODO label smoothing scheduling
                matched = (1 - self.opt.augment.label_smoothing) * matched + \
                    self.opt.augment.label_smoothing * \
                    (torch.ones(len(images), len(images)).to(images.device) - matched)
            img_emb = self.img_enc(images)

        cap_emb = txt_enc(captions, torch.Tensor(lengths).to(captions.device))
        return img_emb, cap_emb, matched, ids

    def training_step(self, batch, batch_idx):
        # for TQDM progress bar
        sys.stdout.flush()
        img_emb, cap_emb, matched, _ = self.forward_step(batch, self.mixup_fn)
        if not self.opt.train.get('dist_train'):
            loss, loss_dict = self.criterion(img_emb, cap_emb, matched=matched)
        else:
            # code for the distributed training
            rank = dist.get_rank()

            cap_emb_mean_gat = grad_all_gather(cap_emb['mean'])
            if self.opt.model.is_probabilistic_model:
                cap_emb_std_gat = grad_all_gather(cap_emb['std'])
            else:
                cap_emb_std_gat = cap_emb_mean_gat

            if self.opt.train.get('all_gather_infonce'):
                img_emb_mean_gat = grad_all_gather(img_emb['mean'])
                img_emb_all = torch.cat([img_emb['mean']] + img_emb_mean_gat[:rank] + img_emb_mean_gat[rank + 1:])
                cap_emb_all = torch.cat([cap_emb['mean']] + cap_emb_mean_gat[:rank] + cap_emb_mean_gat[rank + 1:])

                loss_img, loss_dict_img = self.criterion(img_emb, {'mean': cap_emb_all}, distributed=True)
                loss_cap, loss_dict_cap = self.criterion(cap_emb, {'mean': img_emb_all}, distributed=True)
                loss = (loss_img + loss_cap) / 2
                loss_dict = {}
                for k, v in loss_dict_img.items():
                    loss_dict[k] = (v + loss_dict_cap[k]) / 2
            else:
                img_emb_mean_gat = grad_all_gather(img_emb['mean'])
                img_emb_std_gat = grad_all_gather(img_emb['std'])
                img_emb_all = torch.cat([img_emb['mean']] + img_emb_mean_gat[:rank] + img_emb_mean_gat[rank + 1:])
                cap_emb_all = torch.cat([cap_emb['mean']] + cap_emb_mean_gat[:rank] + cap_emb_mean_gat[rank + 1:])

                img_emb_std_all = torch.cat([img_emb['std']] + img_emb_std_gat[:rank] + img_emb_std_gat[rank + 1:])
                cap_emb_std_all = torch.cat([cap_emb['std']] + cap_emb_std_gat[:rank] + cap_emb_std_gat[rank + 1:])

                extended_matched = torch.cat(
                    [matched] +
                    [self.opt.augment.label_smoothing * torch.ones(len(img_emb['mean']), len(img_emb['mean'])).to(matched.device)] * (len(img_emb_mean_gat) - 1),
                    dim=1)
                loss_img, loss_dict_img = self.criterion(
                    img_emb,
                    {'mean': cap_emb_all, 'std': cap_emb_std_all},
                    matched=extended_matched)
                extended_matched = torch.cat(
                    [matched.T] +
                    [self.opt.augment.label_smoothing * torch.ones(len(img_emb['mean']), len(img_emb['mean'])).to(matched.device)] * (len(img_emb_mean_gat) - 1),
                    dim=1)
                loss_cap, loss_dict_cap = self.criterion(
                    cap_emb,
                    {'mean': img_emb_all, 'std': img_emb_std_all},
                    matched=extended_matched)

                loss = (loss_img + loss_cap) / 2
                loss_dict = {}
                for k, v in loss_dict_img.items():
                    loss_dict[k] = (v + loss_dict_cap[k]) / 2

        if self.opt.model.is_probabilistic_model:
            if self.opt.model.get('sigma_ln_init') is not None:
                loss_dict['ln_weight/img_weight'] = torch.mean(self.img_enc.image_std_ln.weight.data).item()
                loss_dict['ln_weight/txt_weight'] = torch.mean(self.txt_enc.text_std_ln.weight.data).item()
                loss_dict['ln_weight/img_bias'] = torch.mean(self.img_enc.image_std_ln.bias.data).item()
                loss_dict['ln_weight/txt_bias'] = torch.mean(self.txt_enc.text_std_ln.bias.data).item()
            loss_dict['std/img_gmean'] = torch.mean(img_emb['std'])
            loss_dict['std/txt_gmean'] = torch.mean(cap_emb['std'])
            loss_dict['std/img_sum'] = torch.sum(torch.exp(img_emb['std']))
            loss_dict['std/txt_sum'] = torch.sum(torch.exp(cap_emb['std']))

        # Warmup scheduling
        warmup_alpha = 1
        if self.current_epoch < self.opt.train.embedding_warmup_epochs and self.opt.train.get('linear_warmup', True):
            warmup_alpha = (self.current_epoch * self.trainer.num_training_batches + batch_idx + 1) / \
                (self.opt.train.embedding_warmup_epochs *
                 self.trainer.num_training_batches)
            loss *= warmup_alpha
            loss_dict['warmup_alpha/warmup_alpha'] = warmup_alpha
        if self.current_epoch == self.opt.train.embedding_warmup_epochs and self.opt.train.get('embedding_warmup_after_freeze'):
            # VSE infty style embedding warmup after freezing backbones
            warmup_alpha = (batch_idx + 1) / self.trainer.num_training_batches
            loss *= warmup_alpha
            loss_dict['warmup_alpha/warmup_alpha'] = warmup_alpha
        if torch.isnan(loss) or torch.isinf(loss):
            print('Killed due to NaN!!')
            exit(0)
        loss_dict['lr'] = self.trainer.optimizers[0].param_groups[0]['lr'] * warmup_alpha
        self.log_dict(loss_dict, on_step=True, on_epoch=False,
                      prog_bar=self.opt.train.get('debug'), logger=True, sync_dist=True, rank_zero_only=True)

        return loss

    def on_train_epoch_start(self):
        # Settings for mixed sample data augmentation
        if (self.opt.augment.get('mixup') or self.opt.augment.get('cutmix')) and \
                self.opt.augment.get('mixup_prob') and \
                (not self.opt.augment.get('mixup_off_epoch') or
                 (self.current_epoch < self.opt.augment.get('mixup_off_epoch'))):
            mixup_fn = Mixup(
                mixup_alpha=self.opt.augment.mixup,
                cutmix_alpha=self.opt.augment.cutmix,
                cutmix_minmax=self.opt.augment.cutmix_minmax,
                prob=self.opt.augment.mixup_prob,
                switch_prob=self.opt.augment.mixup_switch_prob,
                mode=self.opt.augment.mixup_mode,
                label_smoothing=self.opt.augment.mixup_smoothing,
                mix_batch_ratio=self.opt.augment.get('mix_batch_ratio', 1),
            )
            self.print('Setting MSDA')
        else:
            self.print('Setting no MSDA')
            mixup_fn = None
        self.mixup_fn = mixup_fn

        if 'vse_mean_warmup_epochs' in self.opt.train:
            if self.current_epoch >= self.opt.train.vse_mean_warmup_epochs:
                self.criterion.max_violation_on()
            else:
                self.criterion.max_violation_off()

        if 'mean_only_warmup_epochs' in self.opt.train:
            if self.current_epoch >= self.opt.train.mean_only_warmup_epochs:
                self.criterion.num_samples = self.opt.criterion.num_samples
                self.criterion.prob_distance = self.opt.criterion.prob_distance
                self.criterion.pdist_fn = self.opt.criterion.pdist_fn
            else:
                self.criterion.num_samples = 0
                self.criterion.prob_distance = 'squared_l2'
                self.criterion.pdist_fn = 'batchwise_cdist'

        if self.opt.model.get('no_freeze'):
            # Let's avoid `find_unused_parameters` for the fast pre-training
            return

        if self.opt.model.precomp_enc_type == 'backbone' and 'clip' not in self.opt.model.backbone_source:
            if self.current_epoch < self.opt.train.embedding_warmup_epochs:
                self.img_enc.freeze_backbone()
            else:
                self.img_enc.unfreeze_backbone(3)

            if self.current_epoch < self.opt.train.embedding_warmup_epochs:
                pass
            elif self.current_epoch < self.opt.train.embedding_warmup_epochs + self.opt.train.backbone_warmup_epochs:
                # only train the last block of resnet backbone
                self.img_enc.unfreeze_backbone(3)
            elif self.current_epoch < self.opt.train.embedding_warmup_epochs + self.opt.train.backbone_warmup_epochs * 2:
                self.img_enc.unfreeze_backbone(2)
            elif self.current_epoch < self.opt.train.embedding_warmup_epochs + self.opt.train.backbone_warmup_epochs * 3:
                self.img_enc.unfreeze_backbone(1)
            else:
                self.img_enc.unfreeze_backbone(0)
        elif self.opt.model.precomp_enc_type == 'backbone' and 'clip' in self.opt.model.backbone_source:
            embedding_warmup_epochs = self.opt.train.get('embedding_warmup_epochs', 0)
            if self.current_epoch < embedding_warmup_epochs and self.opt.train.get('img_backbone_freeze_epochs'):
                self.img_enc.freeze_backbone()
            else:
                self.img_enc.unfreeze_backbone()

            if self.opt.train.get('img_backbone_freeze_epochs'):
                if self.current_epoch < self.opt.train.img_backbone_freeze_epochs:
                    if self.opt.train.get('img_backbone_reinit_layers'):
                        self.img_enc.freeze_backbone()
                        self.img_enc.unfreeze_backbone(
                            fixed_blocks=12-self.opt.train.img_backbone_reinit_layers)
                    else:
                        self.img_enc.freeze_backbone()
                else:
                    self.img_enc.unfreeze_backbone()

            if self.opt.train.get('txt_backbone_freeze_epochs'):
                if self.current_epoch < self.opt.train.txt_backbone_freeze_epochs:
                    if self.opt.train.get('txt_backbone_reinit_layers'):
                        self.txt_enc.freeze_backbone()
                        self.txt_enc.unfreeze_backbone(
                            fixed_blocks=12-self.opt.train.txt_backbone_reinit_layers)
                    else:
                        self.txt_enc.freeze_backbone()
                else:
                    self.txt_enc.unfreeze_backbone()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.swa_enabled:
            if self.swa_iter_mode and self.current_epoch + 1 > self.swa_start_epoch:
                if batch_idx % self.swa_per_iters == 0:
                    self.swa_img_enc.update_parameters(self.img_enc)
                    self.swa_txt_enc.update_parameters(self.txt_enc)
                    if self.is_swa_criterion:
                        self.swa_criterion.update_parameters(self.criterion)

    def on_train_epoch_end(self):
        if self.swa_enabled:
            if self.swa_epoch_mode and self.current_epoch + 1 > self.swa_start_epoch:
                if (self.current_epoch + 1 - self.swa_start_epoch) % self.swa_per_epochs == 0:
                    self.swa_img_enc.update_parameters(self.img_enc)
                    self.swa_txt_enc.update_parameters(self.txt_enc)
                    if self.is_swa_criterion:
                        self.swa_criterion.update_parameters(self.criterion)

    def __get_params_with_layerwise_lr_decay(self, params, base_lr, layerwise_lr_decay, reinit_layers=0, regexp_pattern='resblocks\.(\d+)\.'):
        regex = re.compile(regexp_pattern)

        llrd_parameters = {}
        non_decayed_parameters = []
        for name, p in params:
            cur_block_cnt = regex.findall(name)
            if cur_block_cnt:
                cur_block_cnt = int(cur_block_cnt[0])
                # check for the probabilistic backbone
                if 'uncertainty' in name or 'prob_resblocks' in name:
                    llrd_parameters.setdefault(f'prob_backbone.{cur_block_cnt}', []).append(p)
                else:
                    llrd_parameters.setdefault(f'backbone.{cur_block_cnt}', []).append(p)
            else:
                non_decayed_parameters.append(p)
        grouped_parameters = []
        grouped_parameters.append(
            {'params': non_decayed_parameters, 'lr': base_lr})

        for name, params in llrd_parameters.items():
            block_cnt = int(name.split('.')[1])
            if 'prob_backbone' in name:
                decay_factor = layerwise_lr_decay ** (self.opt.model.n_unc_layers - block_cnt)
            else:
                decay_factor = layerwise_lr_decay ** (12 - block_cnt)
            grouped_parameters.append(
                {'params': params, 'lr': base_lr * decay_factor})
        return grouped_parameters

    def configure_optimizers(self):
        try:
            getter = getattr(optim, self.opt.optim.name)
        except AttributeError as err:
            if self.opt.optim.name.lower() == 'adamp':
                getter = AdamP
            elif self.opt.optim.name.lower() == 'sgdp':
                getter = SGDP
            else:
                raise ValueError(self.opt.optim.name) from err

        params = list(self.named_parameters())
        if self.opt.optim.get('layerwise_lr_decay') and self.opt.optim.layerwise_lr_decay < 1:
            # Handling layerwise lr decay
            # prob_resblocks
            grouped_parameters = [
                {'params': [
                    p for name, p in params if 'txt_enc' in name and 'txt_enc.backbone' not in name], 'lr': self.opt.optim.lr},
                {'params': [
                    p for name, p in params if 'img_enc' in name and 'img_enc.backbone' not in name], 'lr': self.opt.optim.lr},
                # do not apply weight decay for scalar values
                {'params': [
                    p for name, p in params if 'txt_enc' not in name and 'img_enc' not in name], 'lr': self.opt.optim.lr, 'weight_decay': 0},
            ]

            if self.opt.optim.get('no_img_unc_decay'):
                grouped_parameters.append(
                    {'params': [p for name, p in params if 'img_enc.backbone.uncertainty_transformer' in name],
                     'lr': self.opt.optim.lr})
                img_backbone_params = [
                    (name, p) for name, p in params if 'img_enc.backbone' in name and 'img_enc.backbone.uncertainty_transformer' not in name][::-1]
            else:
                img_backbone_params = [
                    (name, p) for name, p in params if 'img_enc.backbone' in name][::-1]

            img_backbone_decayed_params = self.__get_params_with_layerwise_lr_decay(
                img_backbone_params, self.opt.optim.lr * self.opt.optim.get('img_backbone_lr_decay', 1), self.opt.optim.layerwise_lr_decay, self.opt.train.get('img_backbone_reinit_layers'))
            grouped_parameters.extend(img_backbone_decayed_params)

            if self.opt.optim.get('no_txt_unc_decay'):
                grouped_parameters.append(
                    {'params': [p for name, p in params if 'txt_enc.backbone.transformer.prob_resblocks' in name],
                     'lr': self.opt.optim.lr})
                txt_backbone_params = [
                    (name, p) for name, p in params if 'txt_enc.backbone' in name and 'txt_enc.backbone.transformer.prob_resblocks' not in name][::-1]
            else:
                txt_backbone_params = [
                    (name, p) for name, p in params if 'txt_enc.backbone' in name][::-1]

            txt_backbone_decayed_params = self.__get_params_with_layerwise_lr_decay(
                txt_backbone_params, self.opt.optim.lr * self.opt.optim.get('txt_backbone_lr_decay', 1), self.opt.optim.layerwise_lr_decay, self.opt.train.get('txt_backbone_reinit_layers'))
            grouped_parameters.extend(txt_backbone_decayed_params)
        else:
            # otherwise, only applying backbone lr decay
            grouped_parameters = [
                {'params': [
                    p for name, p in params if 'txt_enc' in name and 'txt_enc.backbone' not in name], 'lr': self.opt.optim.lr},
                {'params': [
                    p for name, p in params if 'img_enc' in name and 'img_enc.backbone' not in name], 'lr': self.opt.optim.lr},
                # do not apply weight decay for scalar values
                {'params': [
                    p for name, p in params if 'txt_enc' not in name and 'img_enc' not in name], 'lr': self.opt.optim.lr, 'weight_decay': 0},
            ]
            if self.opt.optim.get('no_img_unc_decay'):
                grouped_parameters.append({
                    'params': [p for name, p in params if 'img_enc.backbone.uncertainty_transformer' in name],
                    'lr': self.opt.optim.lr})
                grouped_parameters.append({
                    'params': [p for name, p in params if 'img_enc.backbone' in name and 'img_enc.backbone.uncertainty_transformer' not in name],
                    'lr': self.opt.optim.lr * self.opt.optim.get('img_backbone_lr_decay', 1)})
            else:
                grouped_parameters.append({
                    'params': [p for name, p in params if 'img_enc.backbone' in name],
                    'lr': self.opt.optim.lr * self.opt.optim.get('img_backbone_lr_decay', 1)})

            if self.opt.optim.get('no_txt_unc_decay'):
                grouped_parameters.append({
                    'params': [p for name, p in params if 'txt_enc.backbone.transformer.prob_resblocks' in name],
                    'lr': self.opt.optim.lr})
                grouped_parameters.append({
                    'params': [p for name, p in params if 'txt_enc.backbone' in name and 'txt_enc.backbone.transformer.prob_resblocks' not in name],
                    'lr': self.opt.optim.lr * self.opt.optim.get('txt_backbone_lr_decay', 1)})
            else:
                grouped_parameters.append({'params': [p for name, p in params if 'txt_enc.backbone' in name],
                                           'lr': self.opt.optim.lr * self.opt.optim.get('txt_backbone_lr_decay', 1)})

        assert len(list(self.parameters())) == len([p for subparams in grouped_parameters for p in subparams['params']])

        optimizer = getter(grouped_parameters, lr=self.opt.optim.lr,
                           weight_decay=self.opt.optim.wd, **self.opt.optim.get('option', {}))

        if isinstance(self.opt.lr_scheduler, list):
            lr_schedulers = []
            milestones = []
            opts = copy.deepcopy(self.opt.lr_scheduler)
            for lr_scheduler_config in opts:
                name = lr_scheduler_config.pop('name')
                milestone = lr_scheduler_config.pop('milestone')
                getter = getattr(optim.lr_scheduler, name)
                scheduler = getter(optimizer, **lr_scheduler_config)
                lr_schedulers.append(scheduler)
                milestones.append(milestone)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=lr_schedulers, milestones=milestones[:-1])
        else:
            opt = copy.deepcopy(self.opt.lr_scheduler)
            name = opt.pop('name')
            getter = getattr(optim.lr_scheduler, name)
            scheduler = getter(optimizer, **opt)

        if self.opt.get('lr_scheduler_config'):
            lr_scheduler_config = copy.deepcopy(self.opt.lr_scheduler_config)
            lr_scheduler_config['scheduler'] = scheduler
            self.print(optimizer)
            self.print(lr_scheduler_config)
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config
            }
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }

    def on_validation_epoch_start(self):
        self.validation_step_outputs = {}

        for idx in range(2):
            # 0: COCO val loader
            # 1: COCO test loader
            shape = (len(self.trainer.val_dataloaders[idx].dataset), self.opt.model.embed_size)
            self.validation_step_outputs[idx] = {
                'img_embs': np.zeros(shape),
                'cap_embs': np.zeros(shape),
                'img_sigmas': np.zeros(shape),
                'cap_sigmas': np.zeros(shape),
            }

    def validation_step(self, batch, batch_idx, dataloader_idx):
        img_emb, cap_emb, matched, ids = self.forward_step(batch, use_swa=self.eval_by_swa)
        loss, loss_dict = self.criterion(img_emb, cap_emb, matched=matched)
        # keys = {
        #     0: 'coco_val',
        #     1: 'coco_test',
        # }
        loss_dict = {k: v for k, v in loss_dict.items() if 'loss' in k}

        if self.opt.model.is_probabilistic_model:
            self.validation_step_outputs[dataloader_idx]['img_sigmas'][ids] = img_emb['std'].cpu(
            ).numpy().copy()
            self.validation_step_outputs[dataloader_idx]['cap_sigmas'][ids, :] = cap_emb['std'].cpu().numpy().copy()
        self.validation_step_outputs[dataloader_idx]['img_embs'][ids] = img_emb['mean'].cpu(
        ).numpy().copy()
        self.validation_step_outputs[dataloader_idx]['cap_embs'][ids] = cap_emb['mean'].cpu().numpy().copy()

        self.log_dict(loss_dict, on_step=False, on_epoch=True,
                      prog_bar=self.opt.train.get('debug'), logger=True,
                      sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        report_dict = {}
        all_img_embs, all_img_sigmas = {}, {}
        all_cap_embs, all_cap_sigmas = {}, {}
        for idx in (0, 1):
            if self.opt.train.get('all_gather'):
                all_img_embs[idx] = sum(self.all_gather(torch.from_numpy(self.validation_step_outputs[idx]['img_embs'])))
                all_cap_embs[idx] = sum(self.all_gather(torch.from_numpy(self.validation_step_outputs[idx]['cap_embs'])))
                all_img_sigmas[idx] = sum(self.all_gather(torch.from_numpy(self.validation_step_outputs[idx]['img_sigmas'])))
                all_cap_sigmas[idx] = sum(self.all_gather(torch.from_numpy(self.validation_step_outputs[idx]['cap_sigmas'])))
            else:
                all_img_embs[idx] = torch.from_numpy(self.validation_step_outputs[idx]['img_embs'])
                all_cap_embs[idx] = torch.from_numpy(self.validation_step_outputs[idx]['cap_embs'])
                all_img_sigmas[idx] = torch.from_numpy(self.validation_step_outputs[idx]['img_sigmas'])
                all_cap_sigmas[idx] = torch.from_numpy(self.validation_step_outputs[idx]['cap_sigmas'])

        for idx, stage in enumerate(['val', 'test']):
            self.print(f'Computing {stage} sims...')
            # NOTE: embeddings are initialized by zero. sum will return the full embedding matrix
            img_embs = all_img_embs[idx]
            cap_embs = all_cap_embs[idx]

            now = datetime.now()
            if self.opt.model.is_probabilistic_model and self.opt.get('validation', {}).get('num_inference_samples'):
                img_sigmas = all_img_sigmas[idx]
                cap_sigmas = all_cap_sigmas[idx]

                sampled_image_features = sample_gaussian_tensors(
                    img_embs[::5, :], img_sigmas[::5, :], self.opt.validation.num_inference_samples).to('cuda')
                sampled_caption_features = sample_gaussian_tensors(
                    cap_embs, cap_sigmas, self.opt.validation.num_inference_samples).to('cuda')
                sims = compute_matching_prob_sims(
                    sampled_image_features, sampled_caption_features, self.opt.validation.num_inference_samples,
                    self.criterion.negative_scale, self.criterion.shift)
            elif self.opt.model.is_probabilistic_model and self.opt.get('validation', {}).get('eval_by_csd'):
                img_sigmas = all_img_sigmas[idx]
                cap_sigmas = all_cap_sigmas[idx]
                sims = compute_csd_sims(img_embs[::5, :], cap_embs, img_sigmas[::5, :], cap_sigmas)
            else:
                sims = compute_matmul_sims(img_embs[::5, :].cpu().numpy().copy(), cap_embs.cpu().numpy().copy())

            self.print(f'Computing sims {sims.shape=} takes {datetime.now() - now}')

            if idx == 0:
                npts = img_embs.shape[0] // 5
                (r1, r5, r10, medr, meanr) = eval_i2t_val(npts, sims)
                (r1i, r5i, r10i, medri, meanr) = eval_t2i_val(npts, sims)
                report_dict['val/r1'] = r1
                report_dict['val/r5'] = r5
                report_dict['val/r10'] = r10
                report_dict['val/r1i'] = r1i
                report_dict['val/r5i'] = r5i
                report_dict['val/r10i'] = r10i
                report_dict['val/rsum'] = r1 + r5 + r10 + r1i + r5i + r10i
            elif idx == 1:
                _report_dict = eval_coco_test_loader(sims, self.opt.dataloader.data_path)
                report_dict.update(_report_dict)
                if self.opt.model.is_probabilistic_model:
                    _report_dict, unc_vs_scores = eval_coco_uncertainty(sims, img_sigmas.cpu().numpy(), cap_sigmas.cpu().numpy(), self.opt.dataloader.data_path)
                    for metric, v in unc_vs_scores.items():
                        for mod, _v in v.items():
                            self.print(f'[Uncertainty Eval] [{metric}] [{mod}] {_v}')
                    report_dict.update(_report_dict)

        if not self.opt.train.get('save_every_k_steps'):
            if not self.eval_by_swa:
                report_dict['step'] = self.current_epoch
            else:
                report_dict['step'] = self.current_epoch + 1

        self.log_dict(report_dict, on_step=False, on_epoch=True,
                      prog_bar=False, logger=True,
                      sync_dist=True, rank_zero_only=True)
