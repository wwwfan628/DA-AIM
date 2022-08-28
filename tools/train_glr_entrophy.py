#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import random
import numpy as np
import pprint
import torch
import os
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.visualization.utils import plot_aux_confusion_matrix, create_cm
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, DA_AVAMeter
from slowfast.utils.multigrid import MultigridSchedule


logger = logging.get_logger(__name__)


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 2
    n, c = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))

def train_epoch(aux_train_loader, train_loader, model, optimizer,  train_meter, cur_epoch, cfg, writer=None):
    
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
    train_meter.iter_tic()
    data_size = len(train_loader)

    aux_train_iterator = iter(aux_train_loader)

    # if train_loader is larger than aux_train_loader, add a backup data loader
    print('Length of train_loader: ', data_size)
    aux_data_len = len(aux_train_loader)
    print('Length of aux_loader: ', len(aux_train_loader))
    if data_size > len(aux_train_loader):
        aux_train_iterator2 = iter(aux_train_loader)
        print('Created a backup aux_loader')
        #data_size = len(aux_train_loader)

    # initialze lists for the confusion matrices
    # y_true = []
    # y_pred = []
    y_true_interim = []
    y_pred_interim = []
    
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'confusion')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'confusion'))

    
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):

        if cur_iter < len(aux_train_loader):
            (inputs_aux, labels_aux, _, meta_aux) = next(aux_train_iterator)
        else:
            (inputs_aux, labels_aux, _, meta_aux) = next(aux_train_iterator2)

        global_input = []
        # print("# Transfer the data to the current GPU device.")
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    global_input.append(torch.cat((inputs[i], inputs_aux[i]),0))
                    global_input[i] = global_input[i].cuda(non_blocking=True)
            else:
                global_input = torch.cat((inputs[i], inputs_aux[i]),0)
                global_input = global_input.cuda(non_blocking=True)

            labels = labels.cuda()
            # labels_aux = labels_aux.cuda()
            
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


        # print("# Update the learning rate.")
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # print(inputs[0].shape,["boxes"].shape, inputs_aux[1].shape, meta_aux["boxes"].shape)
        train_meter.data_toc()
        # with torch.autograd.set_detect_anomaly(True):
        # global_boxes = meta["boxes"] + meta_aux["boxes"]
        # print(meta["boxes"].shape)
        meta_boxes_aux = meta_aux["boxes"]
        sub_batch = inputs[0].shape[0]
        meta_boxes_aux[:,0] = meta_aux["boxes"][:,0] + sub_batch
        global_boxes = torch.cat((meta["boxes"], meta_boxes_aux), 0)
        # print(global_input[0].shape, global_input[1].shape, meta["boxes"].shape, meta_boxes_aux.shape, global_boxes.shape)
        num_ac_boxes = meta["boxes"].shape[0]
        num_aux_boxes = meta_boxes_aux.shape[0]
        if cfg.DETECTION.ENABLE:
            # preds = model(inputs, meta["boxes"], 'action')
            preds, preds_aux = model(global_input, global_boxes)
            # numb x num_action_classes,  numb x num_grl_classes 
        else:
            preds = model(global_input)

        if cfg.GRL.TYPE == 'image':
            src_batch_size = inputs[0].shape[0]
            aux_batch_size = global_input[0].shape[0] - src_batch_size
            labels_aux = torch.zeros((src_batch_size+aux_batch_size)*8)
            labels_aux[src_batch_size*8:] = 1
            labels_aux = labels_aux.cuda().long()
        else:
            labels_aux = torch.zeros(num_ac_boxes+num_aux_boxes)
            labels_aux[num_ac_boxes:] = 1
            labels_aux = labels_aux.cuda().long()
        # print(preds.shape, preds_aux.shape, num_ac_boxes)
        # target_pred = preds[num_ac_boxes:, :]
        # eloss = entropy_loss(target_pred)
        preds = preds[:num_ac_boxes, :]
        # Explicitly declare reduction to mean.
        loss_fun_act = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        # loss_fun_aux = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        # loss_fun_aux = losses.get_loss_func("cross_entropy")(reduction="mean")
        loss_fun_aux = losses.get_loss_func("cross_entropy")(reduction="mean")
        # Compute the loss.
        loss_action = loss_fun_act(preds, labels)
        # check Nan Loss.
        misc.check_nan_losses(loss_action)
        loss_aux = loss_fun_aux(preds_aux, labels_aux)
        misc.check_nan_losses(loss_aux)
        
        # loss_action = loss_action*(1-cfg.AUX.LOSS_FACTOR)
        # loss_aux =  cfg.AUX.LOSS_FACTOR*loss_aux 
        loss = loss_action + loss_aux
        # loss = loss_action + loss_aux + eloss * cfg.AUX.LOSS_FACTOR
        # print(cur_iter, " Losses::>>", loss_action.item(), loss_aux.item())
        optimizer.zero_grad()
        # computes gradients
        loss.backward()
        # Update the parameters.
        optimizer.step()

        train_meter.iter_toc()  # do not measure allreduce for this meter

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
                loss_action = du.all_reduce([loss_action])[0]
                loss_aux = du.all_reduce([loss_aux])[0]
                

            loss = loss.item()
            loss_action = loss_action.item()
            loss_aux = loss_aux.item()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss_joint": loss, "Train/loss_action": loss_action, "Train/loss_aux": loss_aux, "Train/lr": lr},
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

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


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
            preds, _ = model(inputs, meta["boxes"])
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
    val_meter.log_epoch_stats(cur_epoch)


    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            #writer.add_scalars(
            #    {"Val/mAP": val_meter.full_map, "Val/loss": loss}, global_step=cur_epoch
            #)
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



def train_grl(cfg):
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

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    #start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

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
        train_epoch(aux_train_loader, train_loader, model, optimizer,  train_meter, cur_epoch, cfg, writer)

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
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer,  cur_epoch, cfg)

        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, "Val/AVA-mAP")

    if writer is not None:
        writer.close()
