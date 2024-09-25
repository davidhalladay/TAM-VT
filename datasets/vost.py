import os
import copy
import torch
import json
import pickle
import glob
# import pims
from torch.utils.data import Dataset
from pathlib import Path
from .video_transforms import make_video_transforms, prepare
from util.vq2d_utils import get_clip_name_from_clip_uid, extract_window_with_context
from util.box_ops import masks_to_boxes
from einops import rearrange, asnumpy
import time
# import ffmpeg
import numpy as np
import random
import torchvision

from scipy import stats
from PIL import Image

SKIP_VIDS = []
IGNORE_THRESH = 0.2
MIN_FG_PIXELS = 200


class VOSTDatasetWrapper(Dataset):
    def __init__(
        self,
        vid_folder,
        mask_dir,
        ann_file,
        transforms,
        # transforms_reference_crop,
        fps_jittering_file=None,
        fps_jittering_range=[1, 2, 3, 4],
        is_train=False,
        is_inverse=False,
        args=None,
    ):
        """
        :param vid_folder: path to the folder containing a folder "video"
        :param ann_file: path to the json annotation file
        :param transforms: video data transforms to be applied on the videos and boxes
        :param is_train: whether training or not
        :param video_max_len: maximum number of frames to be extracted from a video
        :param video_max_len: maximum number of frames to be extracted from a video at training time
        :param fps: number of frames per second
        :param tmp_crop: whether to use temporal cropping preserving the annotated moment
        :param tmp_loc: whether to use temporal localization annotations
        :param stride: temporal stride k
        """
        assert args is not None
        self.args = args
        self.vid_folder = vid_folder
        self.mask_folder = mask_dir
        self.max_num_objects = args.model.vost.max_num_objects

        self.fps_jittering_file = fps_jittering_file
        self.fps_jittering_range = fps_jittering_range

        # fps jittering
        if self.fps_jittering_file is not None:
            fps_jittering_video_ids = []
            with open(self.fps_jittering_file, "r") as f_ann_file:
                for line in f_ann_file:
                    _id_video = line.rstrip('\n')
                    fps_jittering_video_ids.append(_id_video)
            self.fps_jittering_file = fps_jittering_video_ids
        
        if args.data.vost.get("is_inverse") is None:
            args.data.vost.is_inverse = False
        self.is_inverse = args.data.vost.is_inverse
        
        print("reverse mapping: ", self.is_inverse)


        self._transforms = transforms
        self.is_train = is_train

        # load dict_mask_unique_ids_train
        if self.is_train:
            p_dict_mask_ids = Path(self.args.data.vost.path_unique_mask_ids_train_set)

            with p_dict_mask_ids.open('rb') as fp:
                self.dict_mask_unique_ids_train = pickle.load(fp)

        # load annotations
        self.annotations_video_wise = []
        self.annotations_object_wise = []

        with open(ann_file, "r") as f_ann_file:
            for line in f_ann_file:
                _ann = {}
                _id_video = line.rstrip('\n')
                _ann['video_id'] = _id_video

                _ann['img_files'] = sorted(glob.glob(str(self.vid_folder / _id_video / '*.jpg')))
                _ann['mask_files'] = sorted(glob.glob(str(self.mask_folder / _id_video / '*.png')))
                _ann['num_frames'] = len(_ann['img_files'])

                _mask = np.array(Image.open(_ann['mask_files'][0]).convert("P"))
                _ann['shape'] = np.shape(_mask)

                # accumulate object ids from all frames
                if self.is_train:
                    _id_objects = []
                    for e in self.dict_mask_unique_ids_train[_id_video].values():
                        _id_objects += e
                    _id_objects = [*set(_id_objects)]

                    _ann['num_objects'] = len([e for e in _id_objects if e != 0 and e != 255])
                else:
                    _ann['num_objects'] = len([e for e in np.unique(_mask) if e != 0 and e != 255])
                    
                assert _ann['num_objects'] > 0

                self.annotations_video_wise.append(_ann)

                # adding object-specific annotations
                for _id_object in np.unique(_mask):
                    if _id_object == 0 or _id_object == 255:
                        continue
                    _ann_object_specific = copy.deepcopy(_ann)
                    _ann_object_specific['id_object'] = _id_object

                    # TODO: double check 
                    if self.is_train:
                        num_frames_for_object = len([e
                            for e in self.dict_mask_unique_ids_train[_id_video].values()
                            if _id_object in e
                        ])
                        if num_frames_for_object <= 2:
                            print(f"Disregarding the object with <=2 frames")
                            continue

                    self.annotations_object_wise.append(_ann_object_specific)

    

        # fetch clip-based annotations
        if (self.is_train
            and self.args.train_flags.vost.multi_object.iterate_over_all_clips
        ):
            len_clip = self.args.model.vost.video_max_len
            len_step = len_clip // 2 + 1

            self.annotations_clip_wise = []
            for elem in self.annotations_video_wise:
                len_video = elem['num_frames']

                for __i in range(0, len_video - len_clip, len_step):
                    candidate_frame_ids = [Path(e).stem for e in elem['img_files'][__i: min(__i + len_clip, len_video)]]
                    mask_ids_first_frame_clip = self.dict_mask_unique_ids_train[elem['video_id']][candidate_frame_ids[0]]
                    if len(mask_ids_first_frame_clip) == 1 and (0 in mask_ids_first_frame_clip or 255 in mask_ids_first_frame_clip):
                        continue
                    if len(mask_ids_first_frame_clip) == 2 and (0 in mask_ids_first_frame_clip and 255 in mask_ids_first_frame_clip):
                        continue

                    elem_clip = {
                        "video_id": elem["video_id"],
                        "img_files": elem['img_files'][__i: min(__i + len_clip, len_video)],
                        "mask_files": elem['mask_files'][__i: min(__i + len_clip, len_video)],
                        "num_frames": min(__i + len_clip, len_video) - __i,
                        "num_objects": len([e for e in mask_ids_first_frame_clip if e != 0 and e != 255]),
                        "shape": elem["shape"],
                    }
                    self.annotations_clip_wise.append(elem_clip)

        # if args.debug: import ipdb; ipdb.set_trace()
        if self.is_train:
            if self.args.train_flags.vost.multi_object.enable:
                if self.args.train_flags.vost.multi_object.iterate_over_all_clips:
                    print(" > [TRAIN] Keeping train dataset as multiple object annotations")
                    self.annotations = self.annotations_clip_wise
                else:
                    print(" > [TRAIN] Keeping train dataset as multiple object annotations")
                    self.annotations = self.annotations_video_wise
            else:
                print(" > [TRAIN] Keeping train dataset as single object annotations")
                self.annotations = self.annotations_object_wise
        else:
            print(" > [VAL] Keeping val dataset as multiple-object annotations"
                  " or all objects in one video sequence")
            self.annotations = self.annotations_video_wise

        # if args.debug: import ipdb; ipdb.set_trace()
        self.video_max_len = self.args.model.vost.video_max_len
        # self.fps = fps
        # self.tmp_crop = tmp_crop
        # self.tmp_loc = tmp_loc

        self.plot_pred = self.args.eval_flags.plot_pred

        # DEBUG
        if self.is_train and self.args.debug:
            _size_subsample = 20
            print(f"[TRAIN] WARNING WARNING WARNING WARNING WARNING:"
                  f" Subsampling train set for debugging"
                  f" from {len(self.annotations)} to size: {_size_subsample}")
            self.annotations = self.annotations[:_size_subsample]

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
                    f" from {len(self.annotations)} to size: {_size_subsample}")

                indices_to_subsample = [
                    *range(0, len(self.annotations), len(self.annotations) // _size_subsample)
                ][:_size_subsample]

                self.annotations = [self.annotations[e] for e in indices_to_subsample]
                print(f"Indices used to subsample eval set "
                      f"(len: {len(indices_to_subsample)}) : {indices_to_subsample}")
                print(f"Video ids subsampled: {[e['video_id'] for e in self.annotations]}")
                # print(f"Num unique clip_uids before: {len(set([e['clip_uid'] for e in self.annotations]))}")
                # print(f"Num unique clip_uids after subsampling: {len(set([e['clip_uid'] for e in self.annotations]))}")

        self._summarize()

        if args.eval and not self.is_train and args.eval_flags.TTA.enable:
            print(" > Duplicating eval dataset to make multi-scale inference")
            self._duplicate_annotations_with_multi_scale()

    def _summarize(self, ):
        len_imgs = []
        num_objs = []
        for _ann in self.annotations_object_wise:
            len_imgs.append(len(_ann['img_files']))
            num_objs.append(_ann['num_objects'])

        len_imgs = np.array(len_imgs)
        num_objs = np.array(num_objs)

        # Num objs in train set -- video-wise
        # (array([541,  21,   7]), array([ 1,  3,  5, 10]))

        bins = [0, 10, 50, 70, 100, 150, 200, 300]
        hist = np.histogram(len_imgs, bins=bins)[0]
        print(f"Histogram (num images):\n\tbins: {bins}\n\thist: {hist}")
        print(f"Num img files. "
              f"Mean: {np.mean(len_imgs)}, "
              f"Mode: {stats.mode(len_imgs, keepdims=False).mode}, "
              f"Min: {np.min(len_imgs)}, Max: {np.max(len_imgs)}"
            )

        bins = [0, 1, 3, 5, 7, 10, 20]
        hist = np.histogram(num_objs, bins=bins)[0]
        print(f"Histogram (num objects):\n\tbins: {bins}\n\thist: {hist}")
        print(f"Num objects. "
              f"Mean: {np.mean(num_objs)}, "
              f"Mode: {stats.mode(num_objs, keepdims=False).mode}, "
              f"Min: {np.min(num_objs)}, Max: {np.max(num_objs)}"
            )

    def _duplicate_annotations_with_multi_scale(self):
        # import ipdb; ipdb.set_trace()
        annotations_w_scale = []
        for _ann in self.annotations:
            for scale in self.args.eval_flags.TTA.scales:
                _ann_new = copy.deepcopy(_ann)
                _ann_new.update({"scale": scale})

                annotations_w_scale.append(_ann_new)

        self.annotations = annotations_w_scale

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CTHW video tensor
        targets: list of frame-level target, one per frame, dictionary with keys image_id, boxes, orig_sizes
        tmp_target: video-level target, dictionary with keys video_id, qtype, inter_idx, frames_id, caption
        """
        video = self.annotations[idx]
        video_id = video["video_id"]
        image_ids = [Path(e).stem for e in video["img_files"]]

        fps_jittering = 1 # no jittering
        if self.fps_jittering_file is not None:
            if video_id in self.fps_jittering_file:
                fps_jittering = random.sample(self.fps_jittering_range, k=1)[0]

        p_img_files = video["img_files"]

        if self.args.eval_flags.vost.fps10:
            print("This is 10fps eval")
            print("We just create dummy mask files")
            p_mask_files = video["mask_files"] * 2
        else:
            p_mask_files = video["mask_files"]

        frame_ids = [*range(len(p_img_files))]
        # if self.args.debug: import ipdb; ipdb.set_trace()

        # SELECTING OBJECT IN TRAIN SET
        if self.is_train and self.args.train_flags.vost.multi_object.enable:
            if self.args.train_flags.vost.multi_object.iterate_over_all_clips:
                frame_ids_to_select = copy.deepcopy(frame_ids)
            else:
                # get unique mask ids for the vid
                assert len(self.dict_mask_unique_ids_train[video_id]) == len(frame_ids)
                id_objects_across_masks = [
                    np.asarray(self.dict_mask_unique_ids_train[video_id][k])
                    for k in sorted(self.dict_mask_unique_ids_train[video_id].keys())
                ]

                # if self.args.debug: import ipdb; ipdb.set_trace()
                id_frames_to_consider = []
                for __i, __e in enumerate(id_objects_across_masks):
                    if len(__e) == 1 and (0 in __e or 255 in __e):
                        continue
                    elif len(__e) == 2 and (0 in __e and 255 in __e):
                        continue
                    else:
                        id_frames_to_consider.append(__i)

                # id_frames_to_consider = [__i for __i, __e in enumerate(id_objects_across_masks) if len(__e) > 1]
                id_frames_to_consider = [e for e in id_frames_to_consider if e < (len(frame_ids) - self.args.model.vost.video_max_len)]
                if len(id_frames_to_consider) > 0:
                    id_first_frame_sampled = random.sample(id_frames_to_consider, k=1)[0]
                else:
                    id_first_frame_sampled = 0

                frame_ids_to_select = [
                    *range(id_first_frame_sampled, min(id_first_frame_sampled + self.args.model.vost.video_max_len, max(frame_ids)))
                ]

            if "memory" in self.args.train_flags.vost and self.args.train_flags.vost.memory.preseve_0th_frame:
                if 0 not in frame_ids_to_select:
                    frame_ids_to_select = [0] + frame_ids_to_select

            # load only relevant frames
            frames = []
            masks = []
            for __e in frame_ids_to_select:
                frames.append(np.array(Image.open(p_img_files[__e]).convert('RGB')))
                try:
                    masks.append(np.array(Image.open(p_mask_files[__e]).convert('P'), dtype=np.uint8))
                except Exception as e:
                    print(f"Exception {e} is raised when loading {p_mask_files[__e]}")

            frames = np.stack(frames)
            masks = np.stack(masks)
            
            
            image_ids = [image_ids[__i] for __i in frame_ids_to_select]

            id_objects_across_masks = [np.unique(e) for e in masks]

            id_object_to_consider = np.unique(masks[0])
            id_object_to_consider = [e for e in id_object_to_consider if e != 0 and e != 255]

            id_object_to_consider_true = copy.deepcopy(id_object_to_consider)
            if len(id_object_to_consider) > self.args.train_flags.vost.multi_object.num_objects:
                id_object_to_consider = random.sample(
                    id_object_to_consider, k=self.args.train_flags.vost.multi_object.num_objects
                )
            elif len(id_object_to_consider) < self.args.train_flags.vost.multi_object.num_objects:
                diff = self.args.train_flags.vost.multi_object.num_objects - len(id_object_to_consider)
                id_object_to_consider = id_object_to_consider + [id_object_to_consider[-1]] * diff

            masks_one_hot = []
            for _id_object in id_object_to_consider:
                masks_one_hot.append(
                    (masks == _id_object).astype(np.uint8)
                )
            masks = np.stack(masks_one_hot, axis=1)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        elif self.is_train and not self.args.train_flags.vost.multi_object.enable:
            # get unique mask ids for the vid
            assert len(self.dict_mask_unique_ids_train[video_id]) == len(frame_ids)
            id_objects_across_masks = [
                np.asarray(self.dict_mask_unique_ids_train[video_id][k])
                for k in sorted(self.dict_mask_unique_ids_train[video_id].keys())
            ]

            # object id to consider
            id_object_to_consider = video["id_object"]
            id_object_to_consider_true = copy.deepcopy(id_object_to_consider)

            # if self.args.debug: import ipdb; ipdb.set_trace()

            # reverse
            if self.args.data.vost.use_rand_reverse and np.random.randint(2) == 1:
                p_img_files = p_img_files[::-1]
                p_mask_files = p_mask_files[::-1]
                id_objects_across_masks = id_objects_across_masks[::-1]

            # sample frame ids in which object id is present
            id_frames_to_consider = [__i for __i, __e in enumerate(id_objects_across_masks) if id_object_to_consider in __e]
            id_frames_to_consider = [e for e in id_frames_to_consider if e < (len(frame_ids) - self.args.model.vost.video_max_len)]
            # resample with fps_jittering
            id_frames_to_consider = id_frames_to_consider[::fps_jittering]

            # TODO: check whether this work

            if len(id_frames_to_consider) > 0:
                # prev
                id_first_frame_sampled = random.sample(id_frames_to_consider, k=1)[0]

                # ignore citeria added
                if self.args.data.vost.use_ignore_threshold:
                    max_try = 10
                    flag_ref_found = False
                    try_counter = 0
                    while not flag_ref_found and try_counter < max_try:
                        ref_label = Image.open(p_mask_files[id_first_frame_sampled])
                        ref_label = np.array(ref_label, dtype=np.uint8)
                        xs_ignore, ys_ignore = np.nonzero(ref_label == 255)
                        xs, ys = np.nonzero(ref_label)
                        if len(xs) > MIN_FG_PIXELS and (len(xs_ignore) / len(xs)) <= IGNORE_THRESH:
                            flag_ref_found = True
                        else:
                            id_first_frame_sampled = random.sample(id_frames_to_consider, k=1)[0]
            else:
                id_first_frame_sampled = 0

            frame_ids_to_select = [
                *range(id_first_frame_sampled, min(id_first_frame_sampled + self.args.model.vost.video_max_len, max(frame_ids)))
            ]
            
            # TODO: randomly subsample

            # TODO: check whether this work
            if len(frame_ids_to_select) < self.args.model.vost.video_max_len:
                # freeze last frame to make length equal
                frame_ids_to_select = frame_ids_to_select + [frame_ids_to_select[-1]] * (self.args.model.vost.video_max_len - len(frame_ids_to_select))

            # if self.args.debug: import ipdb; ipdb.set_trace()
            if "memory" in self.args.train_flags.vost and self.args.train_flags.vost.memory.preseve_0th_frame:
                if 0 not in frame_ids_to_select:
                    frame_ids_to_select = [0] + frame_ids_to_select

            # load only relevant frames
            frames = []
            masks = []
            for __e in frame_ids_to_select:
                frames.append(np.array(Image.open(p_img_files[__e]).convert('RGB')))
                try:
                    masks.append(np.array(Image.open(p_mask_files[__e]).convert('P'), dtype=np.uint8))
                except Exception as e:
                    print(f"Exception {e} is raised when loading {p_mask_files[__e]}")

            frames = np.stack(frames)
            masks = np.stack(masks)

            image_ids = [image_ids[__i] for __i in frame_ids_to_select]

            # mask out all other objects
            masks = (masks == id_object_to_consider).astype(np.uint8)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            masks = masks.unsqueeze(1)  # (T, N, H, W)  (N := num masks)
        else:
            # if self.args.debug: import ipdb; ipdb.set_trace()
            # frame_ids_to_select = [*range(len(frames))]
            frame_ids_to_select = copy.deepcopy(frame_ids)

            # load only relevant frames
            frames = []
            masks = []
            for __e in frame_ids_to_select:
                frames.append(np.array(Image.open(p_img_files[__e]).convert('RGB')))
                try:
                    masks.append(np.array(Image.open(p_mask_files[__e]).convert('P'), dtype=np.uint8))
                except Exception as e:
                    print(f"Exception {e} is raised when loading {p_mask_files[__e]}")

            frames = np.stack(frames)
            masks = np.stack(masks)

            image_ids = [image_ids[__i] for __i in frame_ids_to_select]

            id_objects_across_masks = [np.unique(e) for e in masks]

            id_object_to_consider = np.unique(masks[0])
            id_object_to_consider = [e for e in id_object_to_consider if e != 0 and e != 255]
            id_object_to_consider_true = copy.deepcopy(id_object_to_consider)

            masks_one_hot = []
            for _id_object in id_object_to_consider:
                masks_one_hot.append(
                    (masks == _id_object).astype(np.uint8)
                )
            masks = np.stack(masks_one_hot, axis=1)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        h = frames.shape[1]
        w = frames.shape[2]

        targets_list = []
        for _i_img in range(len(frames)):
            target = {}
            target["image_id"] = image_ids[_i_img]
            target["boxes"] = masks_to_boxes(masks[_i_img])
            # TODO: check this comment
            # target["id_objects_present"] = torch.as_tensor([
                # e for e in id_objects_across_masks[_i_img] if e != 0 and e != 255])
            # target["id_object_considered"] = torch.as_tensor(id_object_to_consider)
            # target["id_object_considered_true"] = torch.as_tensor(id_object_to_consider_true)

            target["size"] = torch.as_tensor([int(h), int(w)])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["masks"] = 1.-masks[_i_img] if self.is_inverse else masks[_i_img] # (N, H, W)
            targets_list.append(target)

        # video spatial transform
        if self._transforms is not None:
            if self.args.eval_flags.TTA.enable:
                assert self.args.eval
                images, targets = self._transforms(
                    frames, copy.deepcopy(targets_list), scale=video['scale'])
            else:
                images, targets = self._transforms(frames, copy.deepcopy(targets_list))
        else:
            images, targets = frames, targets_list

        # if self.args.debug: import ipdb; ipdb.set_trace()
        # memory frames and masks
        indices_for_memory = [0]
        # assert len(indices_for_memory) == 1
        if (self.is_train
            and self.args.train_flags.vost.memory.get('multiple_indices_for_memory', False)
        ):
            indices_for_memory = [0, 1]
        # if self.is_train and frame_ids_to_select[1] != 1:
        #     indices_for_memory = [0, 1]

        memory_images = []
        memory_masks = []
        for _ind_mem in indices_for_memory:
            memory_images.append(images[:, _ind_mem])
            memory_masks.append(targets[_ind_mem]['masks'].float())
        

        memory_images = torch.stack(memory_images, dim=1)  # (C, N, H, W)
        memory_masks = torch.stack(memory_masks, dim=1)  # (C, N, H, W)

        # for visualization
        reference = frames[0]

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
            "video_id": video_id,
            "frames_id": image_ids[max(indices_for_memory) + 1:],
            "memory_images": memory_images,
            "memory_masks": memory_masks,
            "scale": video.get('scale', 1.0),
            "task_name": "vost",
        }
        tmp_target['images_list_pims'] = frames[max(indices_for_memory) + 1:]
        if self.plot_pred:
            tmp_target['reference_orig'] = references_orig

        return images[:, max(indices_for_memory) + 1:], targets[max(indices_for_memory) + 1:], tmp_target


def build(image_set, args):
    data_dir = Path(args.data.vost.path)
    if args.eval_flags.vost.fps10:
        vid_dir = data_dir / 'JPEGImages_10fps'
    else:
        vid_dir = data_dir / 'JPEGImages'
    mask_dir = data_dir / 'Annotations'

    if args.test:
        data_test_dir = Path(args.data.vost.path_test)
        vid_dir = data_test_dir / 'JPEGImages'
        mask_dir = data_test_dir / 'Annotations'
        ann_file = data_test_dir / 'ImageSets' / 'test.txt'
    elif image_set == "val":
        if args.eval_flags.vost.ann_file != '':
            ann_file = Path(args.eval_flags.vost.ann_file)
        else:
            if args.eval_flags.vost.split == 'all':
                ann_file = data_dir / 'ImageSets' / 'val.txt'
            elif args.eval_flags.vost.split == 'LNG': # eval long video
                ann_file = Path('./datasets/vost/val_LNG.txt')
            elif args.eval_flags.vost.split == 'SHORT': # eval long video
                ann_file = Path('./datasets/vost/val_SHORT.txt')
            elif args.eval_flags.vost.split == 'MI': # eval video with multi-instance
                ann_file = Path('./datasets/vost/val_MI.txt')
            elif args.eval_flags.vost.split == 'SM': # eval video with small instance
                ann_file = Path('./datasets/vost/val_SM.txt')
            else:
                raise NotImplementedError
    else:
        ann_file_path = args.train_flags.vost.get('ann_file', '')
        if ann_file_path != '':
            ann_file = Path(args.train_flags.vost.ann_file)
        else:
            ann_file = data_dir / 'ImageSets' / 'train.txt'
    print(ann_file)
    assert ann_file.is_file()

    dataset = VOSTDatasetWrapper(
        vid_dir,
        mask_dir,
        ann_file,
        transforms=make_video_transforms(
            "val" if args.data.vost.use_test_transform else image_set,
            cautious=True,
            TTA=args.eval_flags.TTA.enable,
            resolution=args.resolution
        ),
        fps_jittering_file=args.data.vost.fps_jittering_file,
        fps_jittering_range=args.data.vost.fps_jittering_range,
        is_train=image_set == "train",
        args=args,
    )
    return dataset
