# from .vidstg import build as build_vidstg
# from .hcstvg import build as build_hcstvg
# from .vq2d import build as build_vq2d
# from .nlq import build as build_nlq
# from .mq import build as build_mq
# from .mq_slowfast_features import build as build_mq_slowfast_features
# from .mq_w_slowfast_features import build as build_mq_w_slowfast_features

# # vidstg adapted
# from .vidstg_adapted import build as build_vidstg_adapted

# # hcstvg adapted
# from .hcstvg_adapted import build as build_hcstvg_adapted

# # unified
# # from .vq2d_unified import build as build_vq2d_unified
# from .nlq_unified import build as build_nlq_unified
# from .mq_unified import build as build_mq_unified
# from .vq2d_unified import build as build_vq2d_unified


# # adl
# from .adl import build as build_adl_type2
# from .adl_type1 import build as build_adl_type1
# from .adl_mult_queries import build as build_adl_type2_mult_queries

# vost
from .vost import build as build_vost

# # static
from .static import build as build_static


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "vidstg":
        return build_vidstg(image_set, args)
    if dataset_file == "vq2d":
        return build_vq2d(image_set, args)
    if dataset_file == "nlq":
        return build_nlq(image_set, args)
    if dataset_file == "mq":
        return build_mq(image_set, args)
    if dataset_file == "mq_slowfast_features":
        return build_mq_slowfast_features(image_set, args)
    if dataset_file == "mq_w_slowfast_features":
        return build_mq_w_slowfast_features(image_set, args)
    if dataset_file == "hcstvg":
        return build_hcstvg(image_set, args)
    if dataset_file == "vidstg_adapted":
        return build_vidstg_adapted(image_set, args)
    if dataset_file == "hcstvg_adapted":
        return build_hcstvg_adapted(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_unified(dataset_file: str, image_set: str, args):
    if dataset_file == "vq2d":
        return build_vq2d_unified(image_set, args)
    if dataset_file == "nlq":
        return build_nlq_unified(image_set, args)
    if dataset_file == "mq":
        return build_mq_unified(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_adl(dataset_file: str, image_set: str, args):
    if dataset_file == "type2":
        return build_adl_type2(image_set, args)
    elif dataset_file == "type2_mult_queries":
        return build_adl_type2_mult_queries(image_set, args)
    elif dataset_file == "type1":
        return build_adl_type1(image_set, args)
    else:
        raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_vos(dataset_file: str, image_set: str, args):
    if dataset_file == "vos":
        return build_vos(image_set, args)
    else:
        raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_vost(dataset_file: str, image_set: str, args):
    if dataset_file == "vost":
        if args.data.vost.use_aot_version:
            from .vost_from_aot import build as build_vost_from_aot
            return build_vost_from_aot(image_set, args)
        else:
            return build_vost(image_set, args)
    else:
        raise ValueError(f"dataset {dataset_file} not supported")


def build_dataset_static(dataset_file: str, image_set: str, args):
    if dataset_file == "static":
        return build_static(image_set, args)
    else:
        raise ValueError(f"dataset {dataset_file} not supported")
