# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build, build_detr_with_pretrained


def build_model(args):
    if args.transfer:
        return build_detr_with_pretrained(args)
    else:
        return build(args)
