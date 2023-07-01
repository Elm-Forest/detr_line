# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .MultiScaleDETR import build_MultiScaleDETR
from .detr import build
from .detr_ori import build_ori


def build_model(args, dataloader):
    if args.scale:
        return build_MultiScaleDETR(args, dataloader)
    if args.ori:
        return build_ori(args)
    else:
        return build(args, dataloader)
