# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
# import math
# import sys
# import copy
# import os
# import warnings
from typing import Dict, Iterable, Optional
from collections import OrderedDict
from tqdm import tqdm
import gc
import torch
import numpy as np
# import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
# import seaborn as sns
import torch.nn.functional as F
# import numpy as np
from tqdm import tqdm
# import util.dist as dist
from pathlib import Path
from PIL import Image
from einops import rearrange, asnumpy
# from scipy.signal import find_peaks, medfilt
# from datasets.vq2d_eval import VQ2DEvaluator
# from datasets.vq2d_orig_eval import VQ2DOrigEvaluator
from datasets.vost_orig_eval import VOSTOrigEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to, NestedTensor
from util.optim import adjust_learning_rate, update_ema
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, masks_to_boxes
# from util.vq2d_utils import extract_window_with_context
import time


def Soft_aggregation(ps, K, device='cpu'):
    num_objects, seq, H, W = ps.shape
    em = torch.zeros(1, K, seq, H, W).to(device)
    em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def save_mask(mask, img_path):
    if np.max(mask) > 255:
        raise ValueError('Maximum id pixel value is 255')
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.putpalette(color_map().flatten().tolist())
    mask_img.save(img_path)
    
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    postprocessors: Dict[str, torch.nn.Module],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    if args.train_flags.vost.multi_object.enable:
        raise NotImplementedError()

    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # dict_multi_scale_inference_aggregator = {}

    # collected_features = dict()
    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, 1, header)
    ):

        samples = batch_dict["samples"].to(device)
        # if args.debug: import ipdb; ipdb.set_trace()

        durations = batch_dict["durations"]

        # collected_features[batch_dict["video_ids"][0]] = {'1':[], '2':[], '3':[]}

        targets = batch_dict["targets"]
        targets = targets_to(targets, device)

        # forward
        assert len(durations) == 1  # works only on batch size 1
        window_step_size = args.model.vost.memory.clip_length

        outputs_wrt_reference_crops = []
        for _i_mem in range(batch_dict["memory_masks"].tensors.shape[1]):
            # reference_crop = NestedTensor(
            #     reference_crops.tensors[_i_mem][None, ...],
            #     reference_crops.mask[_i_mem][None, ...]
            # )

            # memory initialization
            # kwargs = {}

            memory_encoded = OrderedDict()
            # memory = OrderedDict({0: reference_crop.to(device)})

            memory = OrderedDict({
                _k: {
                    "image": NestedTensor(batch_dict["memory_images"].tensors[_k: _k + 1], batch_dict["memory_images"].mask[_k: _k + 1]).to(device),
                    "mask": NestedTensor(batch_dict["memory_masks"].tensors[_k: _k + 1, _i_mem: _i_mem + 1], batch_dict["memory_masks"].mask[_k: _k + 1, _i_mem: _i_mem + 1]).to(device),
                    }
                for _k in range(batch_dict["memory_images"].tensors.shape[0])
            })

            # if len(collected_features) == 20:
            #     # save this dictionary into pkl file
            #     import pickle
            #     with open('collected_features.pkl', 'wb') as f:
            #         pickle.dump(collected_features, f)
            #     print('collected features saved')
            #     exit()

            outputs_frame_wise = {__i: {} for __i in range(durations[0])}
            for ind_start in range(0, durations[0], window_step_size):
                # measure time cusumed 
                # start_time = time.time()

                ind_end = min(durations[0], ind_start + window_step_size)
                samples_window = NestedTensor(
                    samples.tensors[ind_start: ind_end],
                    samples.mask[ind_start: ind_end]
                )
                if args.eval_flags.vost.eval_first_window_only and ind_start >= (args.model.vost.video_max_len - 1):
                    break

                if args.eval_flags.vost.eval_second_window_only and ind_start >= (2 * args.model.vost.video_max_len - 1):
                    break

                memory_cache_window, memory_encoded, memory = model(
                    "vost",
                    samples_window,
                    [ind_end - ind_start],
                    encode_and_save=True,
                    samples_fast=None,
                    memory_encoded=memory_encoded,
                    memory=memory,
                    # **kwargs,
                )

                # print('0', memory_cache_window['features'][0].shape)
                # TODO: collecting features
                # if ind_start < 10:
                #     collected_features[batch_dict["video_ids"][0]]['1'].append(memory_cache_window['features'][1].tensors[0].cpu().numpy())
                #     collected_features[batch_dict["video_ids"][0]]['2'].append(memory_cache_window['features'][2].tensors[0].cpu().numpy())
                #     collected_features[batch_dict["video_ids"][0]]['3'].append(memory_cache_window['features'][3].tensors[0].cpu().numpy())

                
                # if args.debug: import ipdb; ipdb.set_trace()
                outputs_window = model(
                    "vost",
                    samples_window,
                    [ind_end - ind_start],
                    encode_and_save=False,
                    memory_cache=memory_cache_window,
                    # **kwargs,
                )


                # propagate memory
                _pred_last = F.interpolate(
                    outputs_window['pred_masks'][-1][None], size=samples_window.tensors.shape[-2:],
                    mode="bilinear", align_corners=False
                ).sigmoid()

                # if args.debug: import ipdb; ipdb.set_trace()

                _mem_image_forward = samples_window.tensors[-1].unsqueeze(1)  # (C, N, H, W)
                _mem_image_forward = NestedTensor.from_tensor_list([_mem_image_forward.to(device)])

                _mem_mask_forward = _pred_last.detach()  # (C, N, H, W)
                _mem_mask_forward = NestedTensor.from_tensor_list([_mem_mask_forward.to(device)])

                if len(memory) >= args.model.vost.memory.bank_size and args.model.vost.memory.bank_size != 1:
                    _key_to_remove = [*memory.keys()][1]  # keep first frame fixed
                    assert _key_to_remove != 0

                    # remove entries from memory and encoded memory
                    del memory[_key_to_remove]
                    del memory_encoded[_key_to_remove]
                    

                    memory.update({max(memory.keys()) + 1: {
                        "image": _mem_image_forward, "mask": _mem_mask_forward
                    }})
                elif len(memory) >= args.model.vost.memory.bank_size and args.model.vost.memory.bank_size == 1:
                    pass
                else:
                    memory.update({max(memory.keys()) + 1: {
                        "image": _mem_image_forward, "mask": _mem_mask_forward
                    }})

                # if args.model.vost.memory.keep_first_frame_fixed:
                #     if len(kwargs["memory"]) == args.model.vost.memory.bank_size:
                #         # if args.debug: import ipdb; ipdb.set_trace()
                #         dict_0th = OrderedDict({0: kwargs["memory"][0]})
                #         dict_rest = OrderedDict({k - 1: v for k, v in kwargs["memory"].items() if k != 0})
                #         dict_rest.update({max(dict_rest.keys()) + 1: reference_forward})
                #         dict_rest.popitem(last=False)
                #         dict_0th.update(dict_rest)
                #         kwargs["memory"] = dict_0th
                #     else:
                #         kwargs["memory"].update({max(kwargs["memory"].keys()) + 1: reference_forward})
                # else:
                #     if len(kwargs["memory"]) == args.model.vost.memory.bank_size:
                #         kwargs["memory"].popitem(last=False)
                #         kwargs["memory"] = OrderedDict({k - 1: v for k, v in kwargs["memory"].items()})

                #     kwargs["memory"].update({max(kwargs["memory"].keys()) + 1: reference_forward})

                # UPDATE OUTPUTS
                for id_frame in range(ind_start, ind_end):
                    if len(outputs_frame_wise[id_frame]) == 0:
                        for k, v in outputs_window.items():
                            if k in ['aux_outputs', 'weights', 'ca_weights']:
                                continue
                            v_frame = v[id_frame - ind_start] if k in ['pred_boxes', 'pred_masks'] else v[0, id_frame - ind_start]
                            outputs_frame_wise[id_frame][k] = [v_frame]
                    else:
                        for k, v in outputs_window.items():
                            if k in ['aux_outputs', 'weights', 'ca_weights']:
                                continue
                            v_frame = v[id_frame - ind_start] if k in ['pred_boxes', 'pred_masks'] else v[0, id_frame - ind_start]
                            outputs_frame_wise[id_frame][k].append(v_frame)
                # end_time = time.time()
                # print(f"Time taken for window {ind_start}-{ind_end}: {end_time - start_time}")

            # frame-wise aggregation
            for id_frame in range(durations[0]):
                for k in outputs_frame_wise[id_frame].keys():
                    outputs_frame_wise[id_frame][k] = torch.stack(outputs_frame_wise[id_frame][k]).mean(0)

            outputs_frame_wise_aggregate = {}
            for id_frame in range(durations[0]):
                for k in outputs_frame_wise[id_frame].keys():
                    if k not in outputs_frame_wise_aggregate:
                        outputs_frame_wise_aggregate[k] = [outputs_frame_wise[id_frame][k]]
                    else:
                        outputs_frame_wise_aggregate[k].append(outputs_frame_wise[id_frame][k])

            for k, v in outputs_frame_wise_aggregate.items():
                if k in ['pred_boxes', 'pred_masks']:
                    outputs_frame_wise_aggregate[k] = torch.stack(v)
                elif k in ['pred_sted', 'pred_score_per_frame']:
                    outputs_frame_wise_aggregate[k] = torch.stack(v).unsqueeze(0)

            outputs_wrt_reference_crops.append(
                {k: v for k, v in outputs_frame_wise_aggregate.items() if k not in ['aux_outputs', 'weights', 'ca_weights']}
            )

        # if args.debug: import ipdb; ipdb.set_trace()

        # MASK EVAL

        # orig size
        orig_target_size = targets[0]['orig_size']
        orig_target_size = orig_target_size.cpu().numpy().tolist()

        pred_masks_for_eval = []
        gt_masks_for_eval = []

        # GT mask aggregate
        for _target in targets:
            _t_mask_upsampled = []
            for _i_reference_crop in range(batch_dict["memory_masks"].tensors.shape[1]):
                _t_mask_upsampled.append(
                    torch.nn.functional.interpolate(_target['masks'][_i_reference_crop][None, None, ...].float().cpu(), size=orig_target_size, mode="bilinear", align_corners=False)
                )
            gt_masks_for_eval.append(torch.stack(_t_mask_upsampled)[:, 0, 0])
        gt_masks_for_eval = torch.stack(gt_masks_for_eval)  # (seq_len, num_objs, H, W)

        # if args.debug: import ipdb; ipdb.set_trace()

        # PRED MASK AGGREGATE
        if args.train_flags.vost.multi_object.enable:
            num_objects = len(outputs_wrt_reference_crops)
            pred_masks_for_eval = [
                F.interpolate(outputs_wrt_reference_crops[o]['pred_masks'], size=orig_target_size, mode="bilinear", align_corners=False)
                for o in range(num_objects)
            ]
            pred_masks_for_eval = torch.stack(pred_masks_for_eval).cpu()  # (num_objs, seq_len, 2 (classes) H, W)
            ps = F.softmax(pred_masks_for_eval, dim=2)[:, :, 1]
            logit = Soft_aggregation(ps, num_objects + 1, device)  # 1, K, seq, H, W
            pred_argmax_obj_indices = logit[0].argmax(0)  # seq, H, W
            pred_masks_for_eval = torch.stack([pred_argmax_obj_indices == (o + 1) for o in range(num_objects)]).int()

        elif "multi_object" in args.eval_flags.vost and args.eval_flags.vost.multi_object.enable:
            num_objects = len(outputs_wrt_reference_crops)
            pred_masks_for_eval = [
                F.interpolate(outputs_wrt_reference_crops[o]['pred_masks'].cpu(), size=orig_target_size, mode="bilinear", align_corners=False)
                for o in range(num_objects)
            ]
            pred_masks_for_eval = torch.stack(pred_masks_for_eval).cpu()
            logit = Soft_aggregation(pred_masks_for_eval.squeeze(2).sigmoid(), num_objects + 1, 'cpu')
            pred_argmax_obj_indices = logit[0].argmax(0)  # seq, H, W
            pred_masks_for_eval = torch.stack([pred_argmax_obj_indices == (o + 1) for o in range(num_objects)]).int()

        else:
            for _i_reference_crop in range(batch_dict["memory_masks"].tensors.shape[1]):
                _p_mask_upsampled = []
                for _pred_mask in outputs_wrt_reference_crops[_i_reference_crop]['pred_masks']:
                    _pred_mask_upsamp = F.interpolate(
                        _pred_mask[None, ...], size=orig_target_size, mode="bilinear", align_corners=False
                    )
                    # _pred_mask_upsamp = (_pred_mask_upsamp.sigmoid() > 0.5).cpu().int()
                    _pred_mask_upsamp = _pred_mask_upsamp.cpu()
                    _p_mask_upsampled.append(_pred_mask_upsamp[0, 0])

                pred_masks_for_eval.append(torch.stack(_p_mask_upsampled))
            pred_masks_for_eval = torch.stack(pred_masks_for_eval)  # (num_objs, seq_len, H, W)

        # mask eval
        all_gt_masks = gt_masks_for_eval.permute(1, 0, 2, 3)[:, 1:].cpu().numpy()  # removing 1st frame
        all_res_masks = pred_masks_for_eval[:, 1:].cpu().numpy()  # removing 1st frame
        
        # if args.debug: import ipdb; ipdb.set_trace()

        if args.eval_flags.vost.eval_first_window_only:
            all_gt_masks = all_gt_masks[:, :all_res_masks.shape[1]]

        if args.eval_flags.vost.eval_second_window_only:
            all_gt_masks = all_gt_masks[:, :all_res_masks.shape[1]]

            all_res_masks = all_res_masks[:, args.model.vost.video_max_len - 1:]
            all_gt_masks = all_gt_masks[:, args.model.vost.video_max_len - 1:]

        assert len(evaluator_list) == 1 and isinstance(evaluator_list[0], VOSTOrigEvaluator)



        # if args.eval_flags.multi_scale_inference.enable:
        #     # import ipdb; ipdb.set_trace()
        #     video_id = batch_dict['video_ids'][0]

        #     if len(dict_multi_scale_inference_aggregator) == 0:
        #         dict_multi_scale_inference_aggregator[video_id] = {
        #             'all_res_masks': [all_res_masks],
        #             'all_gt_masks': [all_gt_masks]
        #         }
        #     elif video_id in dict_multi_scale_inference_aggregator:
        #         dict_multi_scale_inference_aggregator[video_id]['all_res_masks'].append(all_res_masks)
        #         dict_multi_scale_inference_aggregator[video_id]['all_gt_masks'].append(all_gt_masks)
        #     elif video_id not in dict_multi_scale_inference_aggregator:
        #         # import ipdb; ipdb.set_trace()
        #         for k, v in dict_multi_scale_inference_aggregator.items():
        #             evaluator_list[0].update(
        #                 v['all_gt_masks'][0],
        #                 (torch.from_numpy(np.stack(v['all_res_masks']).mean(0)).sigmoid() > 0.5).int().numpy()
        #             )
        #         dict_multi_scale_inference_aggregator = {}
        #         dict_multi_scale_inference_aggregator[video_id] = {
        #             'all_res_masks': [all_res_masks],
        #             'all_gt_masks': [all_gt_masks]
        #         }
        # else:


        ### r3
        if args.data.vost.is_inverse:
            all_gt_masks = 1 - all_gt_masks
            all_pred_masks = (torch.from_numpy(all_res_masks).sigmoid() < args.eval_flags.vost.confidence_threshold).int().numpy()
        else:
            all_pred_masks = (torch.from_numpy(all_res_masks).sigmoid() > args.eval_flags.vost.confidence_threshold).int().numpy()


        if not args.eval_flags.vost.vis_only_no_cache:
            evaluator_list[0].update(
                all_gt_masks,
                all_pred_masks
                # all_res_masks
            )

        # for evaluator in evaluator_list:
        #     if isinstance(evaluator, VOSTOrigEvaluator):

        if args.eval_flags.plot_pred:
            # if args.debug: import ipdb; ipdb.set_trace()
            if i_batch > 100:
                print("WARNING WARNING WARNING WARNING STOPPING TESTING ARBITRARILY")
                break
            assert len(batch_dict['frames_id']) == 1  # only works on batch size=1

            # reference-crop-specific
            for _i_reference_crop in range(batch_dict["memory_masks"].tensors.shape[1]):
                p_out_video = Path(args.output_dir) / "plot_pred" / \
                    f"{batch_dict['video_ids'][0]}" / f"object_id_{_i_reference_crop + 1}"
                p_out_video.mkdir(parents=True, exist_ok=True)

                p_out_reference = p_out_video / "reference_crop.jpg"
                reference_crop_orig = batch_dict['reference_orig'][0][_i_reference_crop]

                im_reference_crop_orig = Image.fromarray(reference_crop_orig)
                im_reference_crop_orig.save(p_out_reference)

                assert len(batch_dict['frames_id'][0]) == len(batch_dict['images_list_pims'][0])

                # write frames
                p_out_frames = p_out_video / "frames"
                p_out_frames.mkdir(exist_ok=True)
                
                # if args.debug: import ipdb; ipdb.set_trace()

                # for _frame_id, _image, _pred_box, _pred_mask, _pred_score in zip(
                for _frame_id, _image, _pred_mask in zip(
                    batch_dict['frames_id'][0],
                    batch_dict['images_list_pims'][0],
                    pred_masks_for_eval[_i_reference_crop],
                    # outputs_wrt_reference_crops[_i_reference_crop]['pred_boxes'],
                    # outputs_wrt_reference_crops[_i_reference_crop]['pred_masks'],
                    # outputs_wrt_reference_crops[_i_reference_crop]['pred_score_per_frame'].flatten().sigmoid(),
                ):
                    _frame_id_str = f"{_frame_id}"
                    _im = Image.fromarray(_image)
                    img_w, img_h = _im.size
                    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(torch.int).to(device)

                    ##### For CRF Experiment #####
                    # _im.save(p_out_frames / f"{_frame_id}.jpg")
                    # torch.save(_pred_mask, p_out_frames / f"{_frame_id}_pred_mask.pt")
                    ###############################

                    _target = None
                    for e in targets:
                        if e['image_id'] == _frame_id_str:
                            _target = e
                            break
                    # if args.debug: import ipdb; ipdb.set_trace()
                    if not args.eval_flags.vost.vis_only_pred_mask:
                        fig, ax = plt.subplots()
                        ax.axis("off")
                        ax.imshow(_im, aspect="auto")


                        # PLOT FRAME AND GT BOX
                        # get the id of the object within the frame
                        _id_object_considered = _target['id_object_considered'][_i_reference_crop].item()
                        if _target is not None and _id_object_considered in _target['id_objects_present']:
                            # gt_box_xyxy = box_cxcywh_to_xyxy(_target['boxes'][_i_reference_crop]) * scale_fct
                            # x1, y1, x2, y2 = gt_box_xyxy.cpu().int().numpy()
                            # w = x2 - x1
                            # h = y2 - y1
                            # rect = plt.Rectangle(
                            #     (x1, y1), w, h, linewidth=2, edgecolor="#00FF00", fill=False  # green
                            # )
                            # ax.add_patch(rect)

                            # SEGM
                            gt_mask = torch.nn.functional.interpolate(
                                _target['masks'][_i_reference_crop][None, None, ...].float(),
                                size=(img_h, img_w), mode="bilinear", align_corners=False
                            )
                            if args.data.vost.is_inverse:
                                gt_mask = 1.-gt_mask
                            gt_mask_coord = gt_mask[0, 0].nonzero().cpu().numpy()
                            plt.scatter(
                                gt_mask_coord[:, 1], gt_mask_coord[:, 0], color='green',
                                alpha=0.03,
                                s=3,
                            )

                        fig.set_dpi(100)
                        fig.set_size_inches(img_w / 100, img_h / 100)
                        fig.tight_layout(pad=0)

                        fig.savefig(
                            p_out_frames / f"{_frame_id}_GT.jpg",
                            format="jpg",
                        )
                        plt.close(fig)

                    fig, ax = plt.subplots()
                    ax.axis("off")
                     
                    if args.eval_flags.vost.vis_only_pred_mask:
                        background = np.zeros((img_h, img_w), dtype='float')
                        _im = Image.fromarray(background)
                        ax.imshow(_im, aspect="auto")
                        fg_color = 'white'
                        filename = p_out_frames / f"{_frame_id}.png"
                        filename_npy = p_out_frames / f"{_frame_id}.npy"
                        format_img='png'
                        alpha=1
                        s=3
                        edgecolor='none'
                        
                        mask_plot = (_pred_mask.sigmoid() > args.eval_flags.vost.confidence_threshold).cpu().numpy()
                        mask_plot = (mask_plot * 255.).astype(np.uint8)

                        mask_logit = (_pred_mask.sigmoid()).cpu().numpy()

                        # with open(filename_npy, 'wb') as f:
                        #     np.save(f, mask_logit)
                        
                        save_mask(mask_plot, filename)
                        
                    else:
                        ax.imshow(_im, aspect="auto")
                        fg_color = 'blue'
                        filename = p_out_frames / f"{_frame_id}_PRED.jpg"
                        format_img='jpg'
                        alpha=0.03
                        s=3
                        edgecolor=None

                        # PLOT PRED BOX
                        # pred_box_xyxy = box_cxcywh_to_xyxy(_pred_box) * scale_fct
                        # x1, y1, x2, y2 = pred_box_xyxy.cpu().int().numpy()
                        # w = x2 - x1
                        # h = y2 - y1
                        # rect = plt.Rectangle(
                        #     (x1, y1), w, h, linewidth=2, edgecolor="#0000FF", fill=False  # blue
                        # )
                        # ax.add_patch(rect)

                        # SEGM
                        if args.train_flags.vost.multi_object.enable:
                            _pred_mask_coord = _pred_mask.nonzero().cpu().numpy()
                        else:
                            # PREVIOUSLY
                            # _pred_mask_upsamp = F.interpolate(_pred_mask[None, ...], size=(img_h, img_w), mode="bilinear", align_corners=False)
                            # _pred_mask_upsamp = (_pred_mask_upsamp.sigmoid() > 0.5).cpu().int()
                            # _pred_mask_coord = _pred_mask_upsamp[0, 0].nonzero().cpu().numpy()
                            # _pred_mask_coord = _pred_mask.nonzero().cpu().numpy()
                            if args.data.vost.is_inverse:
                                _pred_mask_coord = (_pred_mask.sigmoid() < args.eval_flags.vost.confidence_threshold).nonzero().cpu().numpy()
                            else:
                                _pred_mask_coord = (_pred_mask.sigmoid() > args.eval_flags.vost.confidence_threshold).nonzero().cpu().numpy()

                        
                        plt.scatter(
                            _pred_mask_coord[:, 1], _pred_mask_coord[:, 0], color=fg_color,
                            alpha=alpha,
                            s=s,
                            edgecolor=edgecolor,
                        )

                        # place a text box in upper left in axes coords
                        """
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        ax.text(0.05, 0.95, f"f_hat: {_pred_score.item():.4f}",
                                transform=ax.transAxes, fontsize=10,
                                verticalalignment='top', bbox=props)
                        """

                        fig.set_dpi(100)
                        fig.set_size_inches(img_w / 100, img_h / 100)
                        fig.tight_layout(pad=0)

                        # save image with eventual box
                        fig.savefig(
                            filename,
                            format=format_img,
                        )
                        plt.close(fig)

    # send the last video for inference
    # if args.eval_flags.multi_scale_inference.enable:
    # for k, v in dict_multi_scale_inference_aggregator.items():
    #     evaluator_list[0].update(
    #         v['all_gt_masks'][0],
    #         (torch.from_numpy(np.stack(v['all_res_masks']).mean(0)).sigmoid() > 0.5).int().numpy()
    #     )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # vidstg_res = None
    # vq2d_res = None
    # hcstvg_res = None
    vost_res_orig = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, VOSTOrigEvaluator):
            vost_res_orig = evaluator.summarize()

    # accumulate predictions from all images
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # if vidstg_res is not None:
    #     stats["vidstg"] = vidstg_res

    # if vq2d_res is not None:
    #     stats["vq2d_res"] = vq2d_res

    if vost_res_orig is not None:
        stats["vost_res_orig"] = vost_res_orig

    # if hcstvg_res is not None:
    #     stats["hcstvg"] = hcstvg_res

    return stats
