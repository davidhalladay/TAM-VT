import os
import copy
import torch
import json
import pickle
import glob
from torch.utils.data import Dataset
from pathlib import Path
from util.box_ops import masks_to_boxes
from einops import rearrange, asnumpy
import cv2
from torchvision import transforms
import numpy as np
import random

import sys

from scipy import stats
from PIL import Image

sys.path.insert(0, "code/VOST")
# import aot_plus.dataloaders
import aot_plus.dataloaders.video_transforms as tr
# from aot_plus.dataloaders.train_datasets import (
#     _merge_sample
# )

SKIP_VIDS = []


def _get_images(sample):
    return [sample['ref_img'], sample['prev_img']] + sample['curr_img']


def _get_labels(sample):
    return [sample['ref_label'], sample['prev_label']] + sample['curr_label']


def _merge_sample(sample1, sample2, min_obj_pixels=100, max_obj_n=10, ignore_in_merge=False):

    sample1_images = _get_images(sample1)
    sample2_images = _get_images(sample2)

    sample1_labels = _get_labels(sample1)
    sample2_labels = _get_labels(sample2)

    obj_idx = torch.arange(0, max_obj_n * 2 + 1).view(max_obj_n * 2 + 1, 1, 1)
    selected_idx = None
    selected_obj = None

    all_img = []
    all_mask = []
    for idx, (s1_img, s2_img, s1_label, s2_label) in enumerate(
            zip(sample1_images, sample2_images, sample1_labels,
                sample2_labels)):
        s2_fg = (s2_label > 0).float()
        s2_bg = 1 - s2_fg
        merged_img = s1_img * s2_bg + s2_img * s2_fg
        merged_mask = s1_label * s2_bg.long() + (
            (s2_label + max_obj_n) * s2_fg.long())
        merged_mask = (merged_mask == obj_idx).float()
        if idx == 0:
            after_merge_pixels = merged_mask.sum(dim=(1, 2), keepdim=True)
            selected_idx = after_merge_pixels > min_obj_pixels
            selected_idx[0] = True
            obj_num = selected_idx.sum().int().item() - 1
            selected_idx = selected_idx.expand(-1,
                                               s1_label.size()[1],
                                               s1_label.size()[2])
            if obj_num > max_obj_n:
                selected_obj = list(range(1, obj_num + 1))
                random.shuffle(selected_obj)
                selected_obj = [0] + selected_obj[:max_obj_n]

        merged_mask = merged_mask[selected_idx].view(obj_num + 1,
                                                     s1_label.size()[1],
                                                     s1_label.size()[2])
        if obj_num > max_obj_n:
            merged_mask = merged_mask[selected_obj]
        merged_mask[0] += 0.1
        merged_mask = torch.argmax(merged_mask, dim=0, keepdim=True).long()

        if ignore_in_merge:
            merged_mask = merged_mask + (s1_label == 255).long() * 255 * (merged_mask == 0).long()
            merged_mask = merged_mask + (s2_label == 255).long() * 255 * (merged_mask == 0).long()

        all_img.append(merged_img)
        all_mask.append(merged_mask)

    sample = {
        'ref_img': all_img[0],
        'prev_img': all_img[1],
        'curr_img': all_img[2:],
        'ref_label': all_mask[0],
        'prev_label': all_mask[1],
        'curr_label': all_mask[2:]
    }
    sample['meta'] = sample1['meta']
    sample['meta']['obj_num'] = min(obj_num, max_obj_n)
    return sample


class VOSTDatasetWrapper(Dataset):
    def __init__(
        self,
        root,
        is_train=False,
        args=None,
    ):
        assert args is not None
        self.args = args
        self.root = root
        self.is_train = is_train

        self.image_root = os.path.join(root, 'JPEGImages')
        self.label_root = os.path.join(root, 'Annotations')
        # valid_root = os.path.join(root, 'ValidAnns')
        split = ['train'] if is_train else ['val']
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets',
                                   spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        imglistdic = {}
        # valid_frames_dict = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(self.label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        # transform
        transforms_train = transforms.Compose([
            tr.RandomScale(min_scale=0.7, max_scale=1.3, short_edge=480),
            tr.BalancedRandomCrop((465, 465), max_step=5, max_obj_num=10, min_obj_pixel_num=100),
            tr.RandomHorizontalFlip(prob=0.5),
            tr.Resize(output_size=(465, 465), use_padding=True),
            # tr.ToTensor()
        ])
        # transforms_test = transforms.Compose([
        #     # tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
        #     #                      cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
        #     #                      cfg.MODEL_ALIGN_CORNERS),
        #     tr.MultiRestrictSize(),
        #     tr.MultiToTensor()
        # ])
        transforms_test = transforms.Compose([
            tr.RandomScale(min_scale=1.0, max_scale=1.0, short_edge=480),
            # tr.Resize(output_size=(465, 465), use_padding=True),
            # tr.ToTensor()
        ])

        self.to_tensor = tr.ToTensor()

        # init
        self.enable_prev_frame = False
        self.merge_prob = 0.0  # 0.2
        self.rgb = True
        self.ignore_thresh = 0.2
        self.valid_frames = None
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())

        if self.is_train:
            self.rand_gap = 3
            self.seq_len = self.args.model.vost.video_max_len
            self.rand_reverse = True
            self.repeat_time = 1
            self.transform = transforms_train
            self.dynamic_merge = True
            self.max_obj_n = 10
            self.ignore_in_merge = True
            self.dense_frames = False
        else:
            self.rand_gap = 1
            self.seq_len = -1
            self.rand_reverse = False
            self.repeat_time = 1
            self.transform = transforms_test
            self.dynamic_merge = False
            self.max_obj_n = -1
            self.ignore_in_merge = True
            self.dense_frames = False

        print('Video Num: {} X {}'.format(len(self.seqs), self.repeat_time))

        self.plot_pred = self.args.eval_flags.plot_pred

        # DEBUG
        if self.is_train and self.args.debug:
            _size_subsample = 20
            print(f"[TRAIN] WARNING WARNING WARNING WARNING WARNING:"
                  f" Subsampling train set for debugging"
                  f" from {len(self.seqs)} to size: {_size_subsample}")
            # self.annotations = self.annotations[:_size_subsample]
            self.seqs = self.seqs[:_size_subsample]
            # self.mask_list = self.mask_list[:_size_subsample]

        if not self.is_train:
            _size_subsample = None
            if self.args.debug:
                _size_subsample = self.args.train_flags.vost.eval_set_size.debug
            elif not self.args.eval:
                _size_subsample = self.args.train_flags.vost.eval_set_size.train
            else:
                pass

            if _size_subsample is not None:
                print(f"[EVAL] WARNING WARNING WARNING WARNING WARNING:"
                    f" Subsampling eval set for debugging"
                    f" from {len(self.seqs)} to size: {_size_subsample}")

                indices_to_subsample = [
                    *range(0, len(self.seqs), len(self.seqs) // _size_subsample)
                ][:_size_subsample]

                self.seqs = [self.seqs[e] for e in indices_to_subsample]
                # self.mask_list = [self.mask_list[e] for e in indices_to_subsample]
                print(f"Indices used to subsample eval set "
                      f"(len: {len(indices_to_subsample)}) : {indices_to_subsample}")
                print(f"Video ids subsampled: {self.seqs}")

    def get_ref_index_v2(self,
                         seqname,
                         lablist,
                         min_fg_pixels=200,
                         max_try=40,
                         total_gap=0,
                         ignore_thresh=0.2):
        valid_frames_seq = None
        if self.valid_frames is not None and len(self.valid_frames[seqname]) != 0:
            valid_frames_seq = self.valid_frames[seqname]

        search_range = len(lablist) - total_gap
        if search_range <= 1:
            return 0
        bad_indices = []

        better_than_bad_indices = []
        flag_ref_found = False

        for _ in range(max_try):
            ref_index = np.random.randint(search_range)
            if ref_index in bad_indices:
                continue
            frame_name = lablist[ref_index].split('.')[0] + '.jpg'
            if valid_frames_seq is not None and frame_name not in valid_frames_seq:
                bad_indices.append(ref_index)
                # print('Skipping frame %s for seq %s' % (lablist[ref_index], seqname))
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            xs_ignore, ys_ignore = np.nonzero(ref_label == 255)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and (len(xs_ignore) / len(xs)) <= ignore_thresh:
                flag_ref_found = True
                break
            bad_indices.append(ref_index)

            if len([e for e in np.unique(ref_label) if e != 0 and e != 255]) > 0:
                better_than_bad_indices.append(ref_index)

        if (not flag_ref_found
            or len([e for e in np.unique(ref_label) if e != 0 and e != 255]) == 0
        ):
            return better_than_bad_indices[-1]

        return ref_index

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def sample_gaps(self, seq_len, max_gap=99, max_try=10):
        for _ in range(max_try):
            curr_gaps = []
            total_gap = 0
            for _ in range(seq_len):
                gap = int(np.random.randint(self.rand_gap) + 1)
                # gap = 10
                total_gap += gap
                curr_gaps.append(gap)
            if total_gap <= max_gap:
                break
        return curr_gaps, total_gap

    def get_curr_gaps(self, seq_len, max_gap=99, max_try=10, labels=None, images=None, start_ind=0):
        curr_gaps, total_gap = self.sample_gaps(seq_len, max_gap, max_try)
        valid = False
        if start_ind + total_gap < len(images):
            label_name = images[start_ind + total_gap].split('.')[0] + '.png'
            if label_name in labels:
                valid = True
        count = 0
        while not valid and count < max_try:
            curr_gaps, total_gap = self.sample_gaps(seq_len, max_gap, max_try)
            valid = False
            count += 1
            if start_ind + total_gap < len(images):
                label_name = images[start_ind + total_gap].split('.')[0] + '.png'
                if label_name in labels:
                    valid = True

        if count == max_try:
            curr_gaps = [1] * min(seq_len, (len(images) - start_ind))
            if len(curr_gaps) < seq_len:
                curr_gaps += [0] * (seq_len - len(curr_gaps))
            total_gap = len(images) - start_ind

        return curr_gaps, total_gap

    def get_curr_indices(self, imglist, prev_index, gaps):
        total_len = len(imglist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index, is_ref=False):
        if is_ref:
            frame_name = lablist[index].split('.')[0]
        else:
            frame_name = imagelist[index].split('.')[0]

        image = cv2.imread(
            os.path.join(self.image_root, seqname, frame_name + '.jpg'))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label_name = frame_name + '.png'
        if label_name in lablist:
            label = Image.open(
                os.path.join(self.label_root, seqname, label_name))
            label = np.array(label, dtype=np.uint8)
        else:
            label = None

        return image, label

    def sample_sequence(self, idx, dense_seq=None):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        # import ipdb; ipdb.set_trace()
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 15
        try_step = 0
        while (is_consistent is False and try_step < max_try):
            try_step += 1

            assert not self.enable_prev_frame

            if dense_seq is None:
                dense_seq = False
                # if self.dense_frames and random.random() >= 0.5:
                #     dense_seq = True
            # get ref frame
            if self.is_train:
                ref_index = self.get_ref_index_v2(seqname, lablist, ignore_thresh=self.ignore_thresh, total_gap=self.seq_len)
            else:
                ref_index = 0
            # frame_name = lablist[ref_index].split('.')[0]
            # adjusted_index = imagelist.index(frame_name + '.jpg')
            adjusted_index = ref_index
            if self.is_train:
                curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1, labels=lablist, images=imagelist, start_ind=adjusted_index)
            else:
                curr_gaps, total_gap = self.get_curr_gaps(frame_num - 1, labels=lablist, images=imagelist, start_ind=adjusted_index)

            ref_image, ref_label = self.get_image_label(
                seqname, imagelist, lablist, ref_index, is_ref=True)
            ref_objs = list(np.unique(ref_label))

            # get curr frames
            curr_indices = self.get_curr_indices(imagelist, adjusted_index,
                                                    curr_gaps)
            curr_images, curr_labels, curr_objs = [], [], []
            labeled_frames = [1]
            for curr_index in curr_indices:
                curr_image, curr_label = self.get_image_label(
                    seqname, imagelist, lablist, curr_index)
                if curr_label is not None:
                    c_objs = list(np.unique(curr_label))
                    labeled_frames.append(1)
                else:
                    curr_label = np.full_like(ref_label, 255)
                    c_objs = []
                    labeled_frames.append(0)
                curr_images.append(curr_image)
                curr_labels.append(curr_label)
                curr_objs.extend(c_objs)

            objs = list(np.unique(curr_objs))
            prev_image, prev_label = curr_images[0], curr_labels[0]
            curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_ids = list(np.sort(ref_objs))
        if 255 not in obj_ids:
            obj_num = obj_ids[-1]
        else:
            obj_num = obj_ids[-2]

        sample = {
            'ref_img': ref_image,
            'prev_img': prev_image,
            'curr_img': curr_images,
            'ref_label': ref_label,
            'prev_label': prev_label,
            'curr_label': curr_labels
        }
        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num,
            'dense_seq': dense_seq
        }

        # if self.transform is not None:
        sample_unnorm = self.transform(sample)

        sample = self.to_tensor(sample_unnorm)
        return sample, sample_unnorm

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n, self.ignore_in_merge)

    def __getitem__(self, idx):
        """
        idx: 448
        """
        sample1, sample1_unnorm = self.sample_sequence(idx)

        if (self.is_train and self.dynamic_merge and not sample1['meta']['dense_seq']
            and (sample1['meta']['obj_num'] == 0 or random.random() < self.merge_prob)
        ):
            rand_idx = np.random.randint(len(self.seqs))
            while (rand_idx == (idx % len(self.seqs))):
                rand_idx = np.random.randint(len(self.seqs))

            sample2, _ = self.sample_sequence(rand_idx, False)

            sample = self.merge_sample(sample1, sample2)

            # check whether the merged sample isn't bad
            masks_merged = torch.stack(_get_labels(sample))
            if len([e.item() for e in masks_merged.unique() if e != 0 and e != 255]) == 0:
                sample = sample1  # revert back
        else:
            sample = sample1

        # bringing to TubeDETR's format
        frames = torch.stack(_get_images(sample))
        frames_unnorm = torch.stack(_get_images(sample1_unnorm))
        masks = torch.stack(_get_labels(sample))

        # import ipdb; ipdb.set_trace()
        id_objects_across_masks = masks.unique()
        id_object_to_consider = [e.item() for e in id_objects_across_masks if e != 0 and e != 255]

        id_object_to_consider_true = copy.deepcopy(id_object_to_consider)

        # SELECTING OBJECT IN TRAIN SET
        if self.is_train and self.args.train_flags.vost.multi_object.enable:
            raise ValueError("Written but unverified implementation")
            if len(id_object_to_consider) > self.args.train_flags.vost.multi_object.num_objects:
                id_object_to_consider = random.sample(
                    id_object_to_consider, k=self.args.train_flags.vost.multi_object.num_objects
                )
            elif len(id_object_to_consider) < self.args.train_flags.vost.multi_object.num_objects:
                diff = self.args.train_flags.vost.multi_object.num_objects - len(id_object_to_consider)
                id_object_to_consider = id_object_to_consider + [id_object_to_consider[-1]] * diff

            masks_one_hot = []
            for _id_object in id_object_to_consider:
                masks_one_hot.append((masks == _id_object))
            masks = torch.stack(masks_one_hot, axis=1)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            masks = masks.squeeze(2)

        elif self.is_train and not self.args.train_flags.vost.multi_object.enable:
            try:
                id_object_to_consider = random.sample(id_object_to_consider, k=1)[0]
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Data idx: {idx}")

            # mask out all other objects
            masks = (masks == id_object_to_consider)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            # masks = masks.unsqueeze(1)  # (T, N, H, W)  (N := num masks)

        else:
            masks_one_hot = []
            for _id_object in id_object_to_consider:
                masks_one_hot.append((masks == _id_object))
            masks = torch.stack(masks_one_hot, axis=1)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            masks = masks.squeeze(2)

        # Going forward
        h = frames.shape[2]
        w = frames.shape[3]

        targets_list = []
        for _i_img in range(len(frames)):
            target = {}
            target["image_id"] = _i_img
            target["boxes"] = masks_to_boxes(masks[_i_img])
            target["id_objects_present"] = torch.as_tensor(id_object_to_consider_true)
            target["id_object_considered"] = torch.as_tensor(id_object_to_consider)
            target["id_object_considered_true"] = torch.as_tensor(id_object_to_consider_true)

            target["size"] = torch.as_tensor([int(h), int(w)])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["masks"] = masks[_i_img]

            targets_list.append(target)

        images, targets = frames.permute(1, 0, 2, 3), targets_list

        # memory frames and masks
        indices_for_memory = [0]

        memory_images = []
        memory_masks = []
        for _ind_mem in indices_for_memory:
            memory_images.append(images[:, _ind_mem])
            memory_masks.append(targets[_ind_mem]['masks'].float())

        memory_images = torch.stack(memory_images, dim=1)  # (C, N, H, W)
        memory_masks = torch.stack(memory_masks, dim=1)  # (C, N, H, W)

        # for visualization
        reference = np.array(frames_unnorm[0])

        # reference = torch.as_tensor(rearrange(reference, "h w c -> () c h w"))
        reference = torch.as_tensor(rearrange(reference, "c h w -> () c h w"))
        reference = reference.float()

        ref_masks = targets_list[0]['masks'].float()

        references_raw_image_for_vis = [
            reference * ref_mask[None, None, ...]
            for ref_mask in ref_masks
        ]
        references_raw_image_for_vis = [
            rearrange(asnumpy(reference.byte()), "() c h w -> h w c")
            for reference in references_raw_image_for_vis
        ]
        references_orig = copy.deepcopy(references_raw_image_for_vis)

        # video level annotations
        tmp_target = {
            "video_id": self.seqs[idx],
            "frames_id": np.arange(len(frames))[max(indices_for_memory) + 1:],
            "memory_images": memory_images,
            "memory_masks": memory_masks,
            "task_name": "vost",
        }
        tmp_target['images_list_pims'] = frames_unnorm[max(indices_for_memory) + 1:]
        if self.plot_pred:
            tmp_target['reference_orig'] = references_orig

        return images[:, max(indices_for_memory) + 1:], targets[max(indices_for_memory) + 1:], tmp_target


def build(image_set, args):

    dataset = VOSTDatasetWrapper(
        args.data.vost.path,
        is_train=image_set == "train",
        args=args,
    )
    return dataset
