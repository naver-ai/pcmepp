"""COCO dataset loader
Only RawImageDataset is verified in the PCME++ paper.

Reference code: https://github.com/woodfrog/vse_infty/blob/master/lib/datasets/image_caption.py
"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2

import logging

from pcmepp.modules import clip

logger = logging.getLogger(__name__)


class RawImageDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name, data_split, tokenzier, opt, train, vocab_size=None, noise_ratio=None):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenzier
        self.vocab_size = vocab_size

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.json')
        if 'coco' in data_name:
            self.image_base = osp.join(data_path, 'images')
        else:
            self.image_base = osp.join(data_path, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Read Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.model.precomp_enc_type

        self.backbone_source = opt.model.backbone_source
        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if 'ViT' in self.backbone_source:
            # XXX Let fix base_target_size to 224 for convinience.
            self.base_target_size = 224
        elif hasattr(opt.model, 'input_scale_factor') and opt.model.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.model.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.model.input_scale_factor))

        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        self.length = len(self.captions)

        if not noise_ratio:
            self.idx_mapper = {idx: idx for idx in range(self.length)}
        else:
            # first, select indices to be shuffled
            noise_length = int(noise_ratio * self.length)
            noise_indices = np.arange(self.length)
            np.random.shuffle(noise_indices)
            selected = noise_indices < noise_length

            # second, shuffle indices for the selected indices
            all_indices = np.arange(self.length)
            shuffled_indices = all_indices[selected]
            np.random.shuffle(shuffled_indices)
            all_indices[selected] = shuffled_indices
            self.idx_mapper = {idx: shuffled_idx for idx, shuffled_idx in enumerate(all_indices)}

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # NOTE: this code is for noise_ratio.
        # If no noise ratio is given, idx_mapper is an identity
        cap_index = self.idx_mapper[index]

        img_index = index // self.im_div
        caption = self.captions[cap_index]
        image_id = self.images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        im_in = np.array(imread(image_path))

        if 'ViT' in self.backbone_source:
            target = process_caption_vit(
                clip.tokenize2(caption), self.train, self.vocab_size,
                self.opt.augment.get('txt_size_augment', 0.2),
                self.opt.augment.get('txt_size_augment_masking', 0.5),
                self.opt.augment.get('txt_size_augment_erasing', 0.1))
            is_clip_backbone = True
        else:
            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
            # Convert caption (string) to word ids (with Size Augmentation at training time).
            target = process_caption(self.tokenizer, caption_tokens, self.train)
            is_clip_backbone = False

        processed_image = self._process_image(im_in)
        image = torch.Tensor(processed_image)
        image = image.permute(2, 0, 1)
        return image, target, index, img_index, is_clip_backbone

    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """ Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train, **kwargs):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp_all')
        loc_image = osp.join(data_path, 'precomp_all')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        caption = self.captions[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        target = process_caption(self.tokenizer, caption_tokens, self.train)
        image = self.images[img_index]
        if self.train:  # Size augmentation for region feature
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]
        image = torch.Tensor(image)
        return image, target, index, img_index, 0

    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target


def process_caption_vit(tokens, train, vocab_size,
                        size_augment=0.2, masking=0.5, erasing=0.1):
    target = tokens[0]
    sot = target[0]
    eot = target[-1]
    tokens = target[1:-1]

    output_tokens = []

    # TODO configurable?
    for i, token in enumerate(tokens):
        if size_augment:
            prob = random.random()
            if prob < size_augment and train:  # mask/remove the tokens only during training
                prob /= size_augment

                if prob < masking:
                    # 50% randomly change token to mask token (10%)
                    output_tokens.append(clip._tokenizer.encoder['M'])
                elif prob < (masking + erasing):
                    # 10% randomly change token to random token (2%)
                    # output_tokens.append(random.choice(list(clip._tokenizer.decoder.keys())))
                    output_tokens.append(random.choice(list(range(vocab_size))))
                else:
                    # 40% randomly delete current token (8%)
                    pass
            else:
                output_tokens.append(token)
        else:
            output_tokens.append(token)

    target = [sot] + output_tokens + [eot]
    target = torch.Tensor(target)
    return target


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids, is_clip_backbone = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return all_images, img_lengths, targets, lengths, ids
    else:  # raw input image
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        if is_clip_backbone[0]:
            # XXX CLIP is set to max_len = 77
            max_len = 77
        else:
            max_len = max(lengths)
        targets = torch.zeros(len(captions), max_len).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return images, targets, lengths, img_ids, ids


def get_loader(data_path, data_name, data_split, tokenizer, opt, vocab_size=None, batch_size=100,
               shuffle=True, num_workers=2, train=True, noise_ratio=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.model.precomp_enc_type == 'basic':
        dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train, vocab_size, noise_ratio=noise_ratio)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt, vocab_size=None, **kwargs):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt, vocab_size,
                              batch_size, True, workers,
                              noise_ratio=opt.dataloader.get('noise_ratio'))
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt, vocab_size,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle, **kwargs):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt, vocab_size=None, **kwargs):
    test_loader = get_loader(opt.dataloader.data_path, data_name, split_name, tokenizer, opt, vocab_size,
                             batch_size, False, workers, train=False)
    return test_loader
