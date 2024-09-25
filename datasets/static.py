import os
import copy
import torch
import glob
from torch.utils.data import Dataset
from pathlib import Path
from util.box_ops import masks_to_boxes
from einops import rearrange, asnumpy
import numpy as np
import random

# from scipy import stats
from PIL import Image

import sys
sys.path.insert(0, "./sources")

import torchvision
import torchvision.transforms as TF
import aot_plus.dataloaders.image_transforms as IT

DEFAULT_DATA_RANDOMCROP = (465, 465)
MAX_OBJ_N = 16

SKIP_IDS = ['000000010900', '000000020207', '000000025673', '000000068738', '000000077542', '000000080300', '000000085390', '000000089829', '000000094103', '000000103936', '000000105014', '000000114153', '000000126914', '000000129582', '000000146219', '000000146878', '000000158160', '000000163615', '000000177148', '000000181047', '000000186451', '000000202070', '000000212363', '000000213525', '000000216014', '000000231996', '000000250533', '000000283502', '000000331419', '000000345618', '000000355291', '000000360902', '000000363942', '000000365305', '000000371814', '000000372319', '000000374490', '000000382287', '000000385627', '000000387604', '000000401425', '000000410708', '000000412036', '000000415174', '000000432036', '000000433288', '000000465414', '000000467956', '000000468214', '000000484280', '000000514440', '000000517070', '000000519299', '000000520100', '000000522365', '000000527670', '000000544089', '000000566757', '359', '408', '424', '460', '622']


def _get_images(sample):
    return [sample['ref_img'], sample['prev_img']] + sample['curr_img']


def _get_labels(sample):
    return [sample['ref_label'], sample['prev_label']] + sample['curr_label']


def _merge_sample(sample1, sample2, min_obj_pixels=100, max_obj_n=MAX_OBJ_N, ignore_in_merge=False):
    # import ipdb; ipdb.set_trace()
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


class StaticDatasetWrapper(Dataset):
    def __init__(
        self,
        root,
        is_train=False,
        args=None,
    ):
        assert args is not None
        self.args = args
        self.root = root

        self.output_size = DEFAULT_DATA_RANDOMCROP
        self.max_obj_n = MAX_OBJ_N

        self.img_list = list()
        self.mask_list = list()

        # load annotations
        dataset_list = list()
        lines = ['COCO', 'ECSSD', 'MSRA10K', 'PASCAL-S', 'PASCALVOC2012']
        for line in lines:
            dataset_name = line.strip()

            img_dir = os.path.join(root, 'JPEGImages', dataset_name)
            mask_dir = os.path.join(root, 'Annotations', dataset_name)

            img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) + \
                sorted(glob.glob(os.path.join(img_dir, '*.png')))
            mask_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

            if len(img_list) > 0:
                if len(img_list) == len(mask_list):
                    dataset_list.append(dataset_name)
                    self.img_list += img_list
                    self.mask_list += mask_list
                    print(f'\t{dataset_name}: {len(img_list)} imgs.')
                else:
                    print(
                        f'\tPreTrain dataset {dataset_name} has {len(img_list)} imgs and {len(mask_list)} annots. Not match! Skip.'
                    )
            else:
                print(
                    f'\tPreTrain dataset {dataset_name} doesn\'t exist. Skip.')

        print(f'{len(self.img_list)} imgs are used for PreTrain. They are from {dataset_list}.')

        # SKIP IMAGES WITH QUESTIONABLE MASKS
        img_list_refined = []
        mask_list_refined = []
        for _p_i, _p_m in zip(self.img_list, self.mask_list):
            _id = Path(_p_i).stem
            if _id not in SKIP_IDS:
                img_list_refined.append(_p_i)
                mask_list_refined.append(_p_m)

        self.img_list = img_list_refined
        self.mask_list = mask_list_refined
        print(f'{len(self.img_list)} imgs are used for PreTrain after filtering')

        # TRANSFORMS
        self.pre_random_horizontal_flip = IT.RandomHorizontalFlip(0.5)

        self.random_horizontal_flip = IT.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = IT.RandomAffine(degrees=20,
                                             translate=(0.1, 0.1),
                                             scale=(0.9, 1.1),
                                             shear=10,
                                             resample=torchvision.transforms.InterpolationMode.BICUBIC,
                                             fillcolor=(124, 116, 104))
                                            #  resample=Image.BICUBIC,
        base_ratio = float(self.output_size[1]) / self.output_size[0]
        self.random_resize_crop = IT.RandomResizedCrop(
            self.output_size, (0.8, 1),
            ratio=(base_ratio * 3. / 4., base_ratio * 4. / 3.),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            # interpolation=Image.BICUBIC
        self.to_tensor = TF.ToTensor()
        self.to_onehot = IT.ToOnehot(self.max_obj_n, shuffle=True)
        self.normalize = TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.is_train = is_train
        self.clip_n = self.video_max_len = self.args.model.static.video_max_len
        self.plot_pred = self.args.eval_flags.plot_pred

        # DEBUG
        if self.is_train and self.args.debug:
            _size_subsample = 20
            print(f"[TRAIN] WARNING WARNING WARNING WARNING WARNING:"
                  f" Subsampling train set for debugging"
                  f" from {len(self.img_list)} to size: {_size_subsample}")
            # self.annotations = self.annotations[:_size_subsample]
            self.img_list = self.img_list[:_size_subsample]
            self.mask_list = self.mask_list[:_size_subsample]

        if not self.is_train:
            _size_subsample = None
            if self.args.debug:
                _size_subsample = self.args.train_flags.static.eval_set_size.debug
            elif not self.args.eval:
                _size_subsample = self.args.train_flags.static.eval_set_size.train
            else:
                pass

            if _size_subsample is not None:
                print(f"[EVAL] WARNING WARNING WARNING WARNING WARNING:"
                    f" Subsampling eval set for debugging"
                    f" from {len(self.img_list)} to size: {_size_subsample}")

                indices_to_subsample = [
                    *range(0, len(self.img_list), len(self.img_list) // _size_subsample)
                ][:_size_subsample]

                self.img_list = [self.img_list[e] for e in indices_to_subsample]
                self.mask_list = [self.mask_list[e] for e in indices_to_subsample]
                print(f"Indices used to subsample eval set "
                      f"(len: {len(indices_to_subsample)}) : {indices_to_subsample}")
                print(f"Image ids subsampled: {[Path(e).stem for e in self.img_list]}")

    def __len__(self):
        return len(self.img_list)

    def sample_sequence(self, idx, gap=None):
        img_pil = self.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = self.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = []
        frames_unnorm = []
        masks = []

        img_pil, mask_pil = self.pre_random_horizontal_flip(img_pil, mask_pil)

        for i in range(self.clip_n):
            img, mask = img_pil, mask_pil

            if i > 0:
                img, mask = self.random_horizontal_flip(img, mask)
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            img, mask = self.random_resize_crop(img, mask)

            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_num = len(obj_list)
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            mask = torch.argmax(mask, dim=0, keepdim=True)

            frames_unnorm.append(img)
            frames.append(self.normalize(self.to_tensor(img)))
            masks.append(mask)

        sample = {
            'ref_img': frames[0],
            'prev_img': frames[1],
            'curr_img': frames[2:],
            'ref_label': masks[0],
            'prev_label': masks[1],
            'curr_label': masks[2:]
        }
        sample['meta'] = {
            'seq_name': self.img_list[idx],
            'frame_num': 1,
            'obj_num': obj_num
        }

        return sample, frames_unnorm

    def load_image_in_PIL(self, path, mode='RGB'):
        img = Image.open(path)
        img.load()  # Very important for loading large image
        return img.convert(mode)

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n)

    def __getitem__(self, idx):
        sample1, frames_unnorm = self.sample_sequence(idx)

        if self.is_train:
            condition = True
            while condition:
                rand_idx = np.random.randint(len(self.img_list))
                while (rand_idx == idx):
                    rand_idx = np.random.randint(len(self.img_list))

                sample2, _ = self.sample_sequence(rand_idx)
                sample = self.merge_sample(sample1, sample2)

                masks = torch.stack(_get_labels(sample))
                id_object_to_consider = [e.item() for e in masks.unique() if e != 0]
                if len(id_object_to_consider) == 1 and 0 in id_object_to_consider:
                    print(" > in recursive condition")
                elif len(id_object_to_consider) < 1:
                    print(" > in recursive condition")
                else:
                    condition = False
        else:
            sample = sample1

        # bringing to TubeDETR's format
        frames = torch.stack(_get_images(sample))
        masks = torch.stack(_get_labels(sample))

        id_objects_across_masks = masks.unique()
        id_object_to_consider = [e.item() for e in id_objects_across_masks if e != 0]

        id_object_to_consider_true = copy.deepcopy(id_object_to_consider)

        # SELECTING OBJECT IN TRAIN SET
        if self.is_train and self.args.train_flags.static.multi_object.enable:
            # raise ValueError("Written but unverified implementation")
            if len(id_object_to_consider) > self.args.train_flags.static.multi_object.num_objects:
                id_object_to_consider = random.sample(
                    id_object_to_consider, k=self.args.train_flags.static.multi_object.num_objects
                )
            elif len(id_object_to_consider) < self.args.train_flags.static.multi_object.num_objects:
                diff = self.args.train_flags.static.multi_object.num_objects - len(id_object_to_consider)
                id_object_to_consider = id_object_to_consider + [id_object_to_consider[-1]] * diff

            masks_one_hot = []
            for _id_object in id_object_to_consider:
                masks_one_hot.append((masks == _id_object))
            masks = torch.stack(masks_one_hot, axis=1)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            masks = masks.squeeze(2)

        elif self.is_train and not self.args.train_flags.static.multi_object.enable:
            id_object_to_consider = random.sample(id_object_to_consider, k=1)[0]

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

        # import ipdb; ipdb.set_trace()

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

        reference = torch.as_tensor(rearrange(reference, "h w c -> () c h w"))
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
            "video_id": Path(self.img_list[idx]).stem,
            "frames_id": np.arange(len(frames))[max(indices_for_memory) + 1:],
            "memory_images": memory_images,
            "memory_masks": memory_masks,
            "task_name": "static",
        }
        tmp_target['images_list_pims'] = frames[max(indices_for_memory) + 1:]
        if self.plot_pred:
            tmp_target['reference_orig'] = references_orig

        return images[:, max(indices_for_memory) + 1:], targets[max(indices_for_memory) + 1:], tmp_target

def build(image_set, args):
    data_dir = Path(args.data.static.path)
    dataset = StaticDatasetWrapper(
        data_dir,
        is_train=image_set == "train",
        args=args,
    )
    return dataset
