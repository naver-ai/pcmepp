"""End-to-end training code for PCME++

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import os
import fire

import torch
from transformers import BertTokenizer

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from config import parse_config
from logger import PCMEPPLogger

from pcmepp.datasets import get_loaders, get_test_loader
from pcmepp.engine import PCMEPPModel


def main(config_path, load_from_checkpoint=None, **kwargs):
    """ The training and evaluation script for PCME++.
    This script supports the following features
    (1) loading configurations from the pre-defined configuration file.
    (2) overwriting the configuration file by commandline arguments.
    (3) re-starting the training from the checkpoint.
        (all other arguments will be ignored)

    You can add additional arguments on your configuration file.
    Usage:
        (1) option with value
            --<group_name>__<option_name> <value>
        (2) option with `true` flag
            --<group_name>__<option_name>

    For example, this command will update your lr to 0.0001
    ```
    python train.py ./configs/pcmepp.yaml --optim__lr 0.0001
    ```
    This command will activate early stopping
    ```
    python train.py ./configs/pcmepp.yaml --train__early_stoping
    ```

    Parameters
    ----------
    config_path          : str
                           The path to the configuration file.
                           Ignored if `loader_from_checkpoint` is given.
    load_from_checkpoint : str, optional
                           If given, re-start from the given checkpoint.
                           !Caution! It will ignore all other arguments.
    """
    # Load configuration
    if load_from_checkpoint:
        print(f'Resume from the previous weight {load_from_checkpoint=}')
        ckpt = torch.load(load_from_checkpoint)
        config = ckpt['hyper_parameters']['opt']
        for arg_key, arg_val in kwargs.items():
            keys = arg_key.split('__')
            n_keys = len(keys)

            _config = config
            for idx, _key in enumerate(keys):
                if n_keys - 1 == idx:
                    _config[_key] = arg_val
                else:
                    _config = _config[_key]
    else:
        config = parse_config(config_path,
                              strict_cast=False,
                              **kwargs)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab

    # Data loader
    train_loader, val_loader = get_loaders(
        **config.dataloader, tokenizer=tokenizer, opt=config, vocab_size=len(vocab))
    te_loader = get_test_loader('testall', 'coco', tokenizer,
                                config.dataloader.eval_batch_size, config.dataloader.workers, config, len(vocab))
    val_loader = [val_loader, te_loader]
    if config.train.get('skip_eval'):
        val_loader = None

    # Define model
    model = PCMEPPModel(config)

    # Model checkpoint options
    root_dir = config.train.get('expname', './results')

    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=root_dir,
        filename='model-{epoch:02d}-{eval_avg/eccv_map_at_r:.2f}-{eval_avg/coco_5k_r1:.2f}',
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor='val/rsum',
        mode='max',
    )
    model_summary_callback = ModelSummary(max_depth=2)
    progress_bar = TQDMProgressBar(refresh_rate=config.train.pbar_step)

    callbacks = [
        checkpoint_callback_epoch,
        model_summary_callback,
        progress_bar,
    ]

    if config.train.get('early_stopping'):
        callbacks.append(EarlyStopping(monitor='val/rsum', mode='max'))

    if config.train.get('strategy'):
        strategy = config.train.strategy
    else:
        strategy = DDPStrategy(
            # No way to avoid find_unused_parameters=True. https://github.com/pytorch/pytorch/issues/22049#issuecomment-505617666
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    trainer = pl.Trainer(
        strategy=strategy,
        callbacks=callbacks,
        logger=PCMEPPLogger(
            save_dir=os.path.join(root_dir, 'logs'),
            default_hp_metric=False
        ),
        precision=config.train.precision,
        gradient_clip_val=config.train.grad_clip,
        log_every_n_steps=config.train.log_step,
        max_epochs=config.train.train_epochs,
        num_nodes=int(config.train.get('world_size', 1)),
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        benchmark=True,
        default_root_dir=root_dir,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=load_from_checkpoint)

    if model.swa_enabled:
        model.print('evaluate by SWA')
        model.eval_by_swa = True
        trainer.validate(model, val_loader)


if __name__ == '__main__':
    fire.Fire(main)
