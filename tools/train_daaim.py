#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import random
import numpy as np
import math
import pprint
import torch
import os
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import matplotlib.pyplot as plt
import seaborn as sns

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, DA_AVAMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.utils.ema_helper import create_ema_model, update_ema_variables
from slowfast.utils.daaim_helper import mix, use_pseudo_labels, plot_inputs

logger = logging.get_logger(__name__)


def train_epoch(aux_train_loader, train_loader, model, ema_model, optimizer, train_meter, cur_epoch, cfg, writer=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Enable train mode.
    model.train()
    ema_model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    aux_train_iterator = iter(aux_train_loader)

    # if train_loader is larger than aux_train_loader, add a backup data loader
    print('Length of train_loader: ', data_size)
    print('Length of aux_loader: ', len(aux_train_loader))
    if data_size > len(aux_train_loader):
        aux_train_iterator_2 = iter(aux_train_loader)
        print('Created backup aux_loaders')

    # plot the heatmap of correct/wrongly classified samples
    sample_heatmap = np.zeros([10, cfg.MODEL.NUM_CLASSES, cfg.MODEL.NUM_CLASSES])

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            if cur_iter < len(aux_train_loader):
                (inputs_aux, labels_aux, _, meta_aux) = next(aux_train_iterator)
            else:
                try:
                    (inputs_aux, labels_aux, _, meta_aux) = next(aux_train_iterator_2)
                except StopIteration:
                    aux_train_iterator_2 = iter(aux_train_loader)
                    (inputs_aux, labels_aux, _, meta_aux) = next(aux_train_iterator_2)

            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                        inputs_aux[i] = inputs_aux[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                    inputs_aux = inputs_aux.cuda(non_blocking=True)

                labels = labels.cuda()
                labels_aux = labels_aux.cuda()

                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

                for key, val in meta_aux.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta_aux[key] = val.cuda(non_blocking=True)

            # Update the learning rate.
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)

            train_meter.data_toc()    # not measure allreduce for this meter

            if cfg.DETECTION.ENABLE:
                if cfg.MODEL.LOSS_FUNC == 'cross_entropy':
                    preds = model(inputs, meta["boxes"], head_type='ce_action')    # without softmax, raw logits
                else:
                    preds = model(inputs, meta["boxes"])
            else:
                preds = model(inputs)

            # print('train_dacs.py line 105')
            # print('inputs[0] shape: ', inputs[0].shape)
            # print('inputs[0]: ', inputs[0])
            # print('labels shape: ', labels.shape)
            # print('labels: ', labels)
            # print('predictions shape: ', preds.shape)
            # print('meta boxes shape: ', meta["boxes"].shape)
            # print('meta boxes: ', meta["boxes"])

            # Explicitly declare reduction to mean.
            loss_fun_labeled = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            if cfg.MODEL.LOSS_FUNC == 'cross_entropy':
                if cfg.AVA.IhD1:
                    if not torch.all(labels.sum(dim=1) == 1):
                        for label in labels:
                            if label.sum() != 1 and label[6] == 1:
                                label[6] = 0
                            if label.sum() != 1 and label[7] == 1:
                                label[7] = 0
                            if label.sum() != 1:
                                print(label)
                elif cfg.AVA.IhD2:
                    if not torch.all(labels.sum(dim=1) == 1):
                        for label in labels:
                            if label[3] == 1 and label[5] == 1:
                                label[3] = 0
                            elif label.sum() != 1:
                                print(label)
                else:
                    if not torch.all(labels.sum(dim=1) == 1):
                        for label in labels:
                            if label.sum() != 1:
                                label.index_fill_(0, (label == 1).nonzero(as_tuple=True)[0][1:], 0)
                assert torch.all(labels.sum(dim=1) == 1)   # only belong to one class
                # transfer one hot labels to class indices
                labels_ce = torch.argmax(labels, dim=1)
                loss_labeled = loss_fun_labeled(preds, labels_ce)
            else:
                loss_labeled = loss_fun_labeled(preds, labels)

            # check Nan Loss.
            misc.check_nan_losses(loss_labeled)

            # create pseudo labels and logits
            pseudo_logits_unlabeled = ema_model(inputs_aux, meta_aux["boxes"])    # with softmax/sigmoid, outputs are possibility
            # pseudo_logits_unlabeled = labels_aux
            if cfg.DAAIM.CONSISTENCY_LOSS == 'ce_weighted':
                max_pseudo_logits_unlabeled, pseudo_labels_unlabeled = torch.max(pseudo_logits_unlabeled, dim=1)
                if cfg.AUX.IhD1:
                    if not torch.all(labels_aux.sum(dim=1) == 1):
                        for label_aux in labels_aux:
                            if label_aux.sum() != 1 and label_aux[6] == 1:
                                label_aux[6] = 0
                            if label_aux.sum() != 1 and label_aux[7] == 1:
                                label_aux[7] = 0
                            if label_aux.sum() != 1:
                                print(label_aux)
                elif cfg.AUX.IhD2:
                    if not torch.all(labels_aux.sum(dim=1) == 1):
                        for label_aux in labels_aux:
                            if label_aux[3] == 1 and label_aux[5] == 1:
                                label_aux[3] = 0
                            if label_aux.sum() != 1:
                                print(label_aux)
                else:
                    if not torch.all(labels.sum(dim=1) == 1):
                        for label in labels:
                            if label.sum() != 1:
                                label.index_fill_(0, (label == 1).nonzero(as_tuple=True)[0][1:], 0)
                # transfer one hot labels to class indices
                assert torch.all(labels_aux.sum(dim=1) == 1)
                labels_aux_ce = torch.argmax(labels_aux, dim=1)
                # compute how many pseudo labels are correct and what are they wrongly classified
                for t in range(10):
                    for i in range(cfg.MODEL.NUM_CLASSES):
                        for j in range(cfg.MODEL.NUM_CLASSES):
                            threshold = t * 0.1
                            sample_heatmap[t, i, j] += torch.logical_and(labels_aux_ce==i, torch.logical_and(max_pseudo_logits_unlabeled>=threshold, pseudo_labels_unlabeled==j)).sum().cpu().item()
            else:
                thresholds = torch.tensor(cfg.DAAIM.THRESHOLDS).unsqueeze(dim=0).repeat_interleave(len(pseudo_logits_unlabeled), dim=0).cuda()
                pseudo_labels_unlabeled = pseudo_logits_unlabeled.ge(thresholds).long()
            # _, pseudo_labels_unlabeled = torch.max(pseudo_logits_unlabeled, dim=1)

            if cfg.DAAIM.AUGMENTATION_ENABLE:
                # mix clips from target and source domain
                if cfg.DAAIM.CONSISTENCY_LOSS == 'ce_weighted':
                    inputs_mix, boxes_mix, pseudo_targets_mix, boxes_weights_mix = mix(inputs, inputs_aux, meta["boxes"],
                                                                                       meta_aux["boxes"], labels_ce,
                                                                                       pseudo_labels_unlabeled,
                                                                                       pseudo_logits_unlabeled, cfg, writer)
                else:
                    inputs_mix, boxes_mix, pseudo_targets_mix, boxes_weights_mix = mix(inputs, inputs_aux, meta["boxes"],
                                                                                       meta_aux["boxes"], labels,
                                                                                       pseudo_labels_unlabeled,
                                                                                       pseudo_logits_unlabeled, cfg, writer)
            else:
                # pseudo label only, without augmentation
                inputs_mix, boxes_mix, pseudo_targets_mix, boxes_weights_mix = use_pseudo_labels(inputs_aux, meta_aux["boxes"],
                                                                                                 pseudo_logits_unlabeled,
                                                                                                 pseudo_labels_unlabeled, cfg)

            # draw inputs, inputs_aux, inputs_mix
            if cfg.TENSORBOARD.DAAIM.PLOT_REAL_SAMPLES and (writer is not None):
                if isinstance(inputs, (list,)):
                    plot_inputs(inputs[0], meta["boxes"], cfg, 'real labeled sample: ' + str(cur_epoch) + ' ' + str(cur_iter),
                                writer, labels_ce)
                    plot_inputs(inputs_aux[0], meta_aux["boxes"], cfg,
                                'real unlabeled sample: ' + str(cur_epoch) + ' ' + str(cur_iter), writer, labels_aux_ce)
                    plot_inputs(inputs_mix[0], boxes_mix, cfg, 'real mixed sample: ' + str(cur_epoch) + ' ' + str(cur_iter),
                                writer, pseudo_targets_mix)
                else:
                    plot_inputs(inputs, meta["boxes"], cfg,
                                'real labeled sample: ' + str(cur_epoch) + ' ' + str(cur_iter), writer, labels_ce)
                    plot_inputs(inputs_aux, meta_aux["boxes"], cfg,
                                'real unlabeled sample: ' + str(cur_epoch) + ' ' + str(cur_iter), writer, labels_aux_ce)
                    plot_inputs(inputs_mix, boxes_mix, cfg,
                                'real mixed sample: ' + str(cur_epoch) + ' ' + str(cur_iter), writer, pseudo_targets_mix)

            # predict for mixed videos
            if cfg.DAAIM.CONSISTENCY_LOSS == 'ce_weighted':
                preds_mix = model(inputs_mix, boxes_mix, head_type='ce_action')  # without softmax, raw logits
            else:
                preds_mix = model(inputs_mix, boxes_mix)

            loss_fun_unlabeled = losses.get_loss_func(cfg.DAAIM.CONSISTENCY_LOSS)().cuda()

            # Compute the loss.
            loss_unlabeled = loss_fun_unlabeled(preds_mix, pseudo_targets_mix, boxes_weights_mix)

            # check Nan Loss.
            misc.check_nan_losses(loss_unlabeled)

            # combine loss
            loss = loss_labeled + loss_unlabeled

            # Perform the backward pass.
            optimizer.zero_grad()
            # computes gradients
            loss.backward()
            # Update the parameters.
            optimizer.step()

            # update Mean teacher network
            if ema_model is not None:
                alpha = 0.5
                ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha=alpha,
                                                 global_step=data_size * cur_epoch + cur_iter, cfg=cfg)

            train_meter.iter_toc()  # do not measure allreduce for this meter

            if cfg.DETECTION.ENABLE:
                if cfg.NUM_GPUS > 1:
                    loss = du.all_reduce([loss])[0]
                    loss_labeled = du.all_reduce([loss_labeled])[0]
                    loss_unlabeled = du.all_reduce([loss_unlabeled])[0]

                loss = loss.item()
                loss_labeled = loss_labeled.item()
                loss_unlabeled = loss_unlabeled.item()
                # Update and log stats.
                train_meter.update_stats(None, None, None, loss, lr)
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Train/loss_joint": loss, "Train/loss_labeled": loss_labeled, "Train/loss_unlabeled": loss_unlabeled,
                         "Train/lr": lr},
                        global_step=data_size * cur_epoch + cur_iter,
                    )

            else:
                top1_err, top5_err = None, None
                if cfg.DATA.MULTI_LABEL:
                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        [loss] = du.all_reduce([loss])
                    loss = loss.item()
                else:
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        loss, top1_err, top5_err = du.all_reduce(
                            [loss, top1_err, top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    loss, top1_err, top5_err = (
                        loss.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )

                # Update and log stats.
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

            train_meter.log_iter_stats(cur_epoch, cur_iter)
            train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, writer)
    train_meter.reset()

    if writer is not None and cfg.TENSORBOARD.DAAIM.PSEUDOLABELS_CONFUSION_MATRIX:
        # in case no sample above threshold
        sample_heatmap_denominator = sample_heatmap.sum(axis=2)
        sample_heatmap_denominator[sample_heatmap_denominator == 0] = 1
        sample_heatmap_denominator = np.dstack([sample_heatmap_denominator]*cfg.MODEL.NUM_CLASSES)
        sample_heatmap = sample_heatmap / sample_heatmap_denominator
        # add heat maps to tensorboard
        fig, ax = plt.subplots(nrows=10, figsize=(55, 55))
        for t in range(10):
            _ = sns.heatmap(sample_heatmap[t], vmin=0, vmax=1, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5, ax=ax[t])
            ax[t].set_title('Confusion Matrix, Epoch: ' + str(cur_epoch) + ' Threshold: 0.' + str(t))
            ax[t].set_xlabel('Pseudo Label')
            ax[t].set_ylabel('Real Label')
        tag = 'Confusion Matrices, Epoch: ' + str(cur_epoch)
        writer.add_figure(tag=tag, figure=fig)
        plt.close(fig)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, writer_str="Val/mAP"):
    """

    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.

    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"], 'action')
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)



        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch, writer)

    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            # writer.add_scalars(
            #    {"Val/mAP": val_meter.full_map, "Val/loss": loss}, global_step=cur_epoch
            # )
            writer.add_scalars({writer_str: val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train_daaim(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.  https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # torch.cuda.manual_seed_all(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    ################################################
    # build ema model
    ema_model = create_ema_model(cfg)
    ################################################

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create the video train and val loaders for da
    aux_train_loader = loader.construct_loader_da(cfg, "aux_train")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        # construct meters
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(aux_train_loader, train_loader, model, ema_model, optimizer, train_meter, cur_epoch, cfg, writer)

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, "Val/AVA-mAP")

    if writer is not None:
        writer.close()