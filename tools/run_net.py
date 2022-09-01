#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from train_da import train_da
from train_grl import train_grl
from train_con import train_con
from train_aux import train_aux
from test_aux import test_aux
from train_daaim import train_daaim
from visualization import visualize



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training on the auxiliary loss
    print('Perform domain adaptive training')
    if cfg.AUX.TEST_ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test_aux)
    elif cfg.DAAIM.TRAIN_ENABLE:
        print('DACS_train enable')
        launch_job(cfg=cfg, init_method=args.init_method, func=train_daaim)
    elif cfg.CON.TRAIN_ENABLE:
        print('CON_train enable')
        launch_job(cfg=cfg, init_method=args.init_method, func=train_con)
    elif cfg.GRL.TRAIN_ENABLE:
        print('GRL_train enable')
        launch_job(cfg=cfg, init_method=args.init_method, func=train_grl)
    # Perform domain adaptive training
    elif cfg.DA.TRAIN_ENABLE:
        print('DA_train enable')
        cfg.AUX.TRAIN_ENABLE = True
        launch_job(cfg=cfg, init_method=args.init_method, func=train_da)
    # Perform AUX training
    elif cfg.AUX.TRAIN_ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_aux)
    # Perform training.
    elif cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg)


if __name__ == "__main__":
    main()
