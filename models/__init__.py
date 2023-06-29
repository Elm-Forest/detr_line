# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .MultiScaleDETR import build_MultiScaleDETR
from .detr import build


def build_model(args, dataloader):
    if args.scale:
        return build_MultiScaleDETR(args, dataloader)
    else:
        return build(args, dataloader)
