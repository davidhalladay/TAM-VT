# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import copy
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Iterable, Optional
import math
import numpy as np
import torch
import torch.utils
import math
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

import util.dist as dist
import util.misc as utils
from datasets import build_dataset_vost

from datasets.vost_orig_eval import VOSTOrigEvaluator

from models import build_model_vost
from engine_vost_memory import train_one_epoch

from models.postprocessors import build_postprocessors
from torch.distributed.elastic.multiprocessing.errors import record

from engine_unified_eval_vost_memory import evaluate as evaluate_vost


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--config_path",
        default="./config/vost_multi_scale_memory.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def evaluate(
    task,
    model: torch.nn.Module,
    postprocessors: Dict[str, torch.nn.Module],
    data_loader,
    evaluator_list,
    device: torch.device,
    args
):
    if task == "vost":
        evaluate_vost(
            model=model,
            postprocessors=postprocessors,
            data_loader=data_loader,
            evaluator_list=evaluator_list,
            device=device,
            args=args,
        )
    else:
        raise ValueError(f"Task: {task} not recognized")


@record
def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)
    print("Config:")
    print(OmegaConf.to_yaml(args))
    print("#" * 80)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.deterministic_algorithms:
        print("[DEBUG] Setting deterministic algorithms!") 
        print("[DEBUG] Setting deterministic algorithms!") 
        print("[DEBUG] Setting deterministic algorithms!") 
        print("[DEBUG] Setting deterministic algorithms!") 
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)

    # Build the model
    model, criterion, weight_dict = build_model_vost(args)
    model.to(device)


    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    

    # if args.debug: import ipdb; ipdb.set_trace()

    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        },
    ]

    if args.train_flags.vost.train_mask_head_in_ft_mode:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if ("backbone" not in n and "text_encoder" not in n
                        and "bbox_attention" not in n and "mask_head" not in n
                        and p.requires_grad)
                ]
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if "mask_head" in n or "bbox_attention" in n and p.requires_grad
                ],
                "lr": args.lr * 10,
            },
        ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    task_datasets_train = {}

    for task in args.tasks.names:
        task_datasets_train[task] = build_dataset_vost(task, image_set="train", args=args)

    dataset_train_concat = ConcatDataset([task_datasets_train[task] for task in args.tasks.names])

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train_concat, shuffle=True, rank=args.rank,
                                           drop_last=True, num_replicas=args.world_size)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train_concat)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    dataloader_train = DataLoader(
        dataset_train_concat,
        batch_sampler=batch_sampler_train,
        collate_fn=partial(utils.video_collate_fn_concat, False, 0),
        num_workers=args.num_workers,
        # worker_init_fn=seed_worker if args.deterministic_algorithms else None,
        generator=g if args.deterministic_algorithms else None,
    )
    # val
    """
    changed:
        - task_dataloader_val: providing `batch_sampler` instead of just `sampler`
    """
    task_datasets_val = {}
    task_dataloader_val = {}
    # task_num_iters = {}

    for task in args.tasks.names:
        task_datasets_val[task] = build_dataset_vost(task, image_set="val", args=args)

        if args.distributed and not args.eval_full_set:
            sampler_val = DistributedSampler(task_datasets_val[task], shuffle=False, rank=args.rank)
        elif args.distributed and args.eval_full_set:
            sampler_val = torch.utils.data.SequentialSampler(task_datasets_val[task])
        else:
            sampler_val = torch.utils.data.SequentialSampler(task_datasets_val[task])

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            sampler=sampler_val,
            drop_last=False,
            collate_fn=partial(utils.video_collate_fn_unified(task), False, 0),
            # num_workers=args.num_workers,
            num_workers=0,
        )

    # Used for loading weights from another model and starting a training from scratch. Especially useful if
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model_ema" in checkpoint:
            if (
                args.num_queries < 100
                and "query_embed.weight" in checkpoint["model_ema"]
            ):  # initialize from the first object queries
                checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"][
                    "query_embed.weight"
                ][: args.num_queries]
            if "transformer.time_embed.te" in checkpoint["model_ema"]:
                del checkpoint["model_ema"]["transformer.time_embed.te"]
            if "transformer.time_embed_memory.te" in checkpoint["model_ema"]:
                del checkpoint["model_ema"]["transformer.time_embed_memory.te"]
            for task_name in args.tasks.names:
                key_time_embed = f"transformer.time_embed.{task_name}.te"
                if key_time_embed in checkpoint["model_ema"]:
                    print(f"Deleting: {key_time_embed} from checkpoint['model_ema']")
                    del checkpoint["model_ema"][key_time_embed]
            if "query_embed.weight" in checkpoint["model_ema"]:
                print("[LOAD] Duplicating query embed to text and visual")
                checkpoint["model_ema"]["query_embed.text.weight"] = copy.deepcopy(
                    checkpoint["model_ema"]["query_embed.weight"]
                )
                checkpoint["model_ema"]["query_embed.visual.weight"] = copy.deepcopy(
                    checkpoint["model_ema"]["query_embed.weight"]
                )
                del checkpoint["model_ema"]["query_embed.weight"]

            if args.model.name.backbone == "backbone_sep_mask":
                _tmp_dict = copy.deepcopy(checkpoint["model_ema"])
                for k, v in checkpoint["model_ema"].items():
                    if "backbone" in k:
                        _tmp_dict[k.replace('backbone', 'backbone_mask')] = v
                checkpoint["model_ema"] = _tmp_dict
                del _tmp_dict

            print("\nUnused params from the checkpoint:")
            for k, v in checkpoint["model_ema"].items():
                if k not in model_without_ddp.state_dict():
                    print(f"{k}: {v.shape}")

            print("\nModel params not present in the checkpoint:")
            for k, v in model_without_ddp.state_dict().items():
                if k not in checkpoint["model_ema"]:
                    print(f"{k}: {v.shape}")

            if args.model.vost.memory.rel_time_encoding_type in ["embedding_add", "embedding_add_shaw", "embedding_mul", "embedding_dyna_add", "embedding_dyna_mul"]:
                del checkpoint["model_ema"]["transformer.encoder.rel_time_theta.weight"]
            
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            if (
                args.num_queries < 100 and "query_embed.weight" in checkpoint["model"]
            ):  # initialize from the first object queries
                checkpoint["model"]["query_embed.weight"] = checkpoint["model"][
                    "query_embed.weight"
                ][: args.num_queries]
            if "transformer.time_embed.te" in checkpoint["model"]:
                del checkpoint["model"]["transformer.time_embed.te"]

            if args.model.vost.memory.rel_time_encoding_type in ["embedding_add", "embedding_add_shaw", "embedding_mul", "embedding_dyna_add", "embedding_dyna_mul"]:
                del checkpoint["model"]["transformer.encoder.rel_time_theta.weight"]

            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if "pretrained_resnet101_checkpoint.pth" in args.load:
            model_without_ddp.transformer._reset_temporal_parameters()
        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        print("resuming from", args.resume)
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")

        if "query_embed.weight" in checkpoint["model_ema"]:
            print("[RESUME] Duplicating query embed to text and visual in model_ema")
            checkpoint["model_ema"]["query_embed.text.weight"] = copy.deepcopy(
                checkpoint["model_ema"]["query_embed.weight"]
            )
            checkpoint["model_ema"]["query_embed.visual.weight"] = copy.deepcopy(
                checkpoint["model_ema"]["query_embed.weight"]
            )
            del checkpoint["model_ema"]["query_embed.weight"]

        if "query_embed.weight" in checkpoint["model"]:
            print("[RESUME] Duplicating query embed to text and visual in model")
            checkpoint["model"]["query_embed.text.weight"] = copy.deepcopy(
                checkpoint["model"]["query_embed.weight"]
            )
            checkpoint["model"]["query_embed.visual.weight"] = copy.deepcopy(
                checkpoint["model"]["query_embed.weight"]
            )
            del checkpoint["model"]["query_embed.weight"]
            
        if len(args.eval_flags.vost.interpolate_RTE) != 0:
            # expending RTE with dummy token for longer BS in testing
            # we only support RTE expansion for embedding_dyna_mul
            old_bs = eval(args.eval_flags.vost.interpolate_RTE.split(',')[0])
            new_bs = eval(args.eval_flags.vost.interpolate_RTE.split(',')[1])
            bs_in_ckpt = sum([ii+1 for ii in range(old_bs)])
            print("EXPANDING RTE WEIGHTS!")
            old_RTE_weights_ema = checkpoint["model_ema"]["transformer.encoder.rel_time_theta.weight"][:bs_in_ckpt, :]
            old_RTE_weights = checkpoint["model"]["transformer.encoder.rel_time_theta.weight"][:bs_in_ckpt, :]
            new_RTE_weights_ema = copy.deepcopy(old_RTE_weights_ema)
            new_RTE_weights = copy.deepcopy(old_RTE_weights)
            expand_num = new_bs - old_bs
            for i in range(old_bs+1, old_bs+expand_num+1):
                expand_ratio = (i-2) / float(i-1)
                last_sequence_ema = new_RTE_weights_ema[int(-1*(i-1)):, :]
                last_sequence = new_RTE_weights[int(-1*(i-1)):, :]
                tmp_sequence_ema = []
                tmp_sequence = []
                for jj in range(i):
                    floor_idx = math.floor(jj * expand_ratio) 
                    ceil_idx = math.ceil(jj * expand_ratio)
                    r = jj * expand_ratio - floor_idx
                    tmp_value_ema = r * last_sequence_ema[ceil_idx, :] + (1 - r) * last_sequence_ema[floor_idx, :]
                    tmp_value = r * last_sequence[ceil_idx, :] + (1 - r) * last_sequence[floor_idx, :]
                    tmp_sequence_ema.append([tmp_value_ema])
                    tmp_sequence.append([tmp_value])
                tmp_ema_sequence = torch.Tensor(tmp_sequence_ema, device=checkpoint["model_ema"]["transformer.encoder.rel_time_theta.weight"].device)
                tmp_sequence = torch.Tensor(tmp_sequence, device=checkpoint["model_ema"]["transformer.encoder.rel_time_theta.weight"].device)

                new_RTE_weights_ema = torch.cat((new_RTE_weights_ema, tmp_ema_sequence), dim=0)
                new_RTE_weights = torch.cat((new_RTE_weights, tmp_sequence), dim=0)

            checkpoint["model_ema"]["transformer.encoder.rel_time_theta.weight"] = copy.deepcopy(new_RTE_weights_ema)
            checkpoint["model"]["transformer.encoder.rel_time_theta.weight"] = copy.deepcopy(new_RTE_weights)

        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print(
                    "WARNING: ema model not found in checkpoint, resetting to current model"
                )
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    def build_evaluator_list(dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        if "vost" in dataset_name:
            evaluator_list.append(
                VOSTOrigEvaluator()
            )
        else:
            raise NotImplementedError()
        return evaluator_list

    writer = None

    # Runs only evaluation, by default on the validation set unless --test is passed.
    if args.eval:
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        for task in args.tasks.names:
            print(f"\nEvaluating {task}")
            evaluator_list = build_evaluator_list(task)
            postprocessors = build_postprocessors(args, task)
            curr_test_stats = evaluate(
                task=task,
                model=test_model,
                postprocessors=postprocessors,
                data_loader=task_dataloader_val[task],
                evaluator_list=evaluator_list,
                device=device,
                args=args,
            )

        return

    # Init task-specific count variables
    print("#" * 80)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=dataloader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
            writer=writer,
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            if args.train_flags.vost.multi_object.iterate_over_all_clips:
                freq_ckpt = 2
            else:
                freq_ckpt = 10
            if (
                (epoch + 1) % args.lr_drop == 0
                or (epoch + 1) % freq_ckpt == 0
            ):
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        if (epoch + 1) % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            for task in args.tasks.names:
                print(f"\nEvaluating {task}")
                evaluator_list = build_evaluator_list(task)
                postprocessors = build_postprocessors(args, task)
                curr_test_stats = evaluate(
                    task=task,
                    model=test_model,
                    postprocessors=postprocessors,
                    data_loader=task_dataloader_val[task],
                    evaluator_list=evaluator_list,
                    device=device,
                    args=args,
                )
                if curr_test_stats is not None:
                    test_stats.update(
                        {task + "_" + k: v for k, v in curr_test_stats.items()}
                    )
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        "TubeDETR training and evaluation script", parents=[get_args_parser()]
    )
    
    args_from_cli = parser.parse_args()

    cfg_from_default = OmegaConf.load(args_from_cli.config_path)
    cfg_from_tubedetr_base = OmegaConf.load(cfg_from_default._BASE_)
    cfg_from_cli = OmegaConf.from_cli(args_from_cli.opts)

    args = OmegaConf.merge(
        cfg_from_tubedetr_base, cfg_from_default, cfg_from_cli
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_file = os.path.join(args.auto_resume, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        print(f"OVERRIDING CHECKPOINT TO RESUME FROM: {checkpoint_file}")
        args.resume = checkpoint_file
        
    checkpoint_file = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        print(f"OVERRIDING CHECKPOINT TO RESUME FROM: {checkpoint_file}")
        args.resume = checkpoint_file

    main(args)
