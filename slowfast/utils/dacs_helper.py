import numpy as np
import torch
import torchvision.ops.boxes as bops
from slowfast.utils.ava_evaluation import label_map_util
from slowfast.utils.ava_eval_helper import read_labelmap
import os
import cv2

batch_idx = 0


@torch.no_grad()
def mix(inputs, inputs_unlabeled, boxes, boxes_unlabeled, labels, pseudo_labels_unlabeled, pseudo_logits_unlabeled, cfg, writer=None):
    # select half of bboxes for each video randomly
    boxes_selected = []     # selected bboxes for the batch
    boxes_without_duplicate = []
    if isinstance(inputs, (list,)) and len(inputs) > 1:  # slowfast inputs, 2 threads
        batch_size = inputs[0].shape[0]
    elif isinstance(inputs, (list,)) and len(inputs) == 1:
        batch_size = inputs[0].shape[0]
    for i in range(batch_size):
        boxes_i = boxes[boxes[:, 0] == i]    # find bboxes for video i
        # print('boxes_i: ', boxes_i)
        # boxes might be duplicate (not same, but overlaps a lot)
        # boxes_i_without_duplicate = delete_duplicate(boxes_i)
        boxes_i_without_duplicate = boxes_i    # use all bboxes
        # print('boxes_i_without_duplicate: ', boxes_i_without_duplicate)
        boxes_without_duplicate.append(boxes_i_without_duplicate)
        num_boxes_i_without_duplicate = len(boxes_i_without_duplicate)
        if num_boxes_i_without_duplicate == 0 and (not cfg.DACS.PSEUDO_LABEL_ENABLE):
            num_boxes_i = len(boxes_i)
            boxes_i_selected = (boxes_i[torch.Tensor(np.random.choice(num_boxes_i, 1, replace=False)).long()]).cuda()
        else:
            boxes_i_selected = (boxes_i_without_duplicate[torch.Tensor(np.random.choice(num_boxes_i_without_duplicate,
                                int((num_boxes_i_without_duplicate + num_boxes_i_without_duplicate % 2) / 2),
                                replace=False)).long()]).cuda()
        boxes_selected.append(boxes_i_selected)
    boxes_selected = torch.cat(boxes_selected)
    # print('dacs_helper.py line 26')
    # print('boxes: ', boxes)
    # print('boxes shape: ', boxes.shape)
    # print('batch size: ', batch_size)
    # print('boxes_selected: ', boxes_selected)
    # print('boxes_selected shape: ', boxes_selected.shape)
    boxes_selected = boxes_selected[boxes_selected[:, 0].sort()[1]]     # sort selected bboxes according to video
    boxes_without_duplicate = torch.cat(boxes_without_duplicate)
    boxes_without_duplicate = boxes_without_duplicate[boxes_without_duplicate[:, 0].sort()[1]]  # sort selected bboxes according to video

    resize_labeled_flags = [False for _ in range(batch_size)]
    if isinstance(inputs, (list,)) and len(inputs) > 1:  # slowfast inputs, 2 threads
        # create masks for frames
        resized_boxes_selected, masks_slow_labeled, masks_fast_labeled = createLabeledMasks(inputs, boxes_selected, resize_labeled_flags, resize_enable=cfg.DACS.RESIZE_ENABLE)
        # mix frames
        inputs_mix = mixFrames(inputs, inputs_unlabeled, resize_labeled_flags, masks_slow_labeled, masks_fast_labeled)
    else:
        # create masks for frames
        resized_boxes_selected, masks_slow_labeled = createLabeledMasks(inputs, boxes_selected, resize_labeled_flags, resize_enable=cfg.DACS.RESIZE_ENABLE)
        # mix frames
        inputs_mix = mixFrames(inputs, inputs_unlabeled, resize_labeled_flags, masks_slow_labeled)

    # mix boxes & labels
    if cfg.DACS.PSEUDO_TARGETS == 'binary':
        boxes_mix, pseudo_targets_mix, boxes_weights_mix = mixTargetsAndBoxes(boxes, boxes_unlabeled, resized_boxes_selected, boxes_selected, labels, pseudo_labels_unlabeled, pseudo_logits_unlabeled, inputs, cfg)
    elif cfg.DACS.PSEUDO_TARGETS == 'logits':
        boxes_mix, pseudo_targets_mix, boxes_weights_mix = mixTargetsAndBoxes(boxes, boxes_unlabeled, resized_boxes_selected, boxes_selected, labels, pseudo_logits_unlabeled, pseudo_logits_unlabeled, inputs, cfg)

    # draw inputs, inputs_unlabeled, inputs_mix per 50 batches
    if cfg.TENSORBOARD.DACS.PLOT_SAMPLES and (writer is not None):
        global batch_idx
        if batch_idx % 1 == 0:
            if isinstance(inputs, (list,)):
                plot_inputs(inputs[0], boxes_without_duplicate, cfg, 'labeled sample: ' + str(batch_idx), writer)
                plot_inputs(inputs_unlabeled[0], boxes_unlabeled, cfg, 'unlabeled sample: ' + str(batch_idx), writer)
                plot_inputs(inputs_mix[0], boxes_mix, cfg, 'mixed sample: ' + str(batch_idx), writer)
            else:
                plot_inputs(inputs, boxes_without_duplicate, cfg, 'labeled sample: ' + str(batch_idx), writer)
                plot_inputs(inputs_unlabeled, boxes_unlabeled, cfg, 'unlabeled sample: ' + str(batch_idx), writer)
                plot_inputs(inputs_mix, boxes_mix, cfg, 'mixed sample' + str(batch_idx), writer)
        batch_idx += 1
    return inputs_mix, boxes_mix, pseudo_targets_mix, boxes_weights_mix


@torch.no_grad()
def delete_duplicate(boxes_i):
    boxes_i_without_duplicate = []
    for box in boxes_i:
        # print('dacs_helper.py line 74')
        # print('box: ', box)
        if area(box) > 0:    # box is not a line
            if boxes_i_without_duplicate == []:
                boxes_i_without_duplicate.append(box)
            else:
                duplicate = False
                for box_without_duplicate in boxes_i_without_duplicate:
                    # print('dacs_helper.py line 81')
                    # print('box_without_duplicate: ', box_without_duplicate)
                    duplicate = duplicate or isBoxDuplicate(box, box_without_duplicate)
                    if duplicate:
                        break
                if not duplicate:
                    boxes_i_without_duplicate.append(box)
    if boxes_i_without_duplicate != []:
        return torch.stack(boxes_i_without_duplicate)
    else:
        return torch.zeros(0, 5).cuda()


@torch.no_grad()
def plot_inputs(inputs_tensor, bboxes_tensor, cfg, name="", writer=None, labels_tensor=None):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        inputs_tensor (tensor): a tensor with shape of NxCxTxWxH
        bboxes_tensor (tensor): bounding boxes, each bbox with format of [n, x1, y1, x2, y2]
        name (str): name of image tag.
        writer (tensorboard.SummaryWriter): tensorboard writer to save figure
    """
    categories, class_whitelist = read_labelmap(os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE))
    category_index = label_map_util.create_category_index(categories)
    inputs_tensor = inputs_tensor.float().cpu()
    bboxes_tensor = bboxes_tensor.float().cpu()
    inputs_tensor = inputs_tensor - inputs_tensor.min()
    inputs_tensor = inputs_tensor / inputs_tensor.max()
    inputs_tensor = inputs_tensor.permute(0, 2, 1, 3, 4)   # NxTxCxWxH
    from matplotlib import pyplot as plt
    f, ax = plt.subplots(nrows=inputs_tensor.shape[0], ncols=inputs_tensor.shape[1], figsize=(50, 20))
    for i in range(inputs_tensor.shape[0]):
        for j in range(inputs_tensor.shape[1]):
            ax[i, j].axis("off")
            ax[i, j].imshow(inputs_tensor[i, j].permute(1, 2, 0))
            bboxes_i = bboxes_tensor[bboxes_tensor[:, 0] == i]
            if labels_tensor is not None:
                labels_i = labels_tensor[bboxes_tensor[:, 0] == i]
            else:
                labels_i = None
            bboxes_i_new, labels_i_new = merge_duplicate(bboxes_i, labels_i)
            for index, box in enumerate(bboxes_i_new):
                _, x1, y1, x2, y2 = box
                ax[i, j].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i, j].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i, j].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i, j].hlines(y2, x1, x2, colors="g", linestyles="solid")
                if labels_tensor is not None:
                    label = labels_i_new[index]
                    class_name = category_index[label.int().item()+1]["name"]
                    ax[i, j].text(x1, y1-0.5, class_name, backgroundcolor='g', color='w')
    # f.savefig(path)
    if writer is not None:
        writer.add_figure(tag=name, figure=f)
    plt.close(f)


def merge_duplicate(bboxes_i, labels_i=None):
    already_in_group = [False for _ in range(len(bboxes_i))]
    bboxes_groups_indices = []
    bboxes_groups = []
    bboxes_i_new = []
    if labels_i != None:
        labels_i_new = []
    for i, bbox_1 in enumerate(bboxes_i):
        if area(bbox_1) <= 0:
            if not already_in_group[i]:
                already_in_group[i] = True    # not valid
        else:
            if not already_in_group[i]:    # box hasn't been checked
                new_group = [bbox_1]
                new_group_ids = [i]
                already_in_group[i] = True
                for j, bbox_2 in enumerate(bboxes_i):
                    if area(bbox_2) <= 0:
                        if not already_in_group[j]:
                            already_in_group[j] = True
                    else:
                        if not already_in_group[j]:
                            if isBoxDuplicate(bbox_1, bbox_2):
                                new_group.append(bbox_2)
                                new_group_ids.append(j)
                                already_in_group[j] = True
                bboxes_groups.append(new_group)
                bboxes_groups_indices.append(new_group_ids)
    if bboxes_groups:
        for i in range(len(bboxes_groups)):
            bbox_group = torch.stack(bboxes_groups[i]).mean(dim=0)
            if labels_i != None:
                label_group = labels_i[bboxes_groups_indices[i][0]]
            bboxes_i_new.append(bbox_group)
            if labels_i != None:
                labels_i_new.append(label_group)
        bboxes_i_new = torch.stack(bboxes_i_new)
        if labels_i != None:
            labels_i_new = torch.stack(labels_i_new)
        else:
            labels_i_new = None
        # print('boxes_new', boxes_new)
        return bboxes_i_new, labels_i_new
    else:
        bboxes_i_new = torch.stack([bboxes_i[0]])
        if labels_i != None:
            labels_i_new = torch.stack([labels_i[0]])
        else:
            labels_i_new = None
        # print('boxes_new', boxes_new)
        return bboxes_i_new, labels_i_new


@torch.no_grad()
def createLabeledMasks(inputs, boxes_selected, resize_labeled_flags, resize_enable=False):
    if isinstance(inputs, (list,)) and len(inputs) > 1:    # slowfast inputs, 2 threads
        batch_size = inputs[0].shape[0]
        frame_w = inputs[0].shape[3]
        frame_h = inputs[0].shape[4]
        mask_slow_size = inputs[0].shape[1:]
        mask_fast_size = inputs[1].shape[1:]
        masks_slow = []
        masks_fast = []
        resized_boxes_selected = []
        for i in range(batch_size):
            mask_slow = torch.zeros(mask_slow_size).cuda()
            mask_fast = torch.zeros(mask_fast_size).cuda()
            boxes_i_selected = boxes_selected[boxes_selected[:, 0] == i]
            for box in boxes_i_selected:
                # expand bboxes by 20%
                w = box[3] - box[1]
                h = box[4] - box[2]
                x1 = max(0, int(box[1] - (w * 0.15)))   # overflow
                x2 = min(int(box[3] + (w * 0.15)), frame_w)
                y1 = max(0, int(box[2] - (h * 0.15)))
                y2 = min(int(box[4] + (h * 0.15)), frame_h)
                mask_slow[:, :, y1:y2, x1:x2] = 1
                mask_fast[:, :, y1:y2, x1:x2] = 1
            if resize_enable and ((mask_slow[0, 0, :, :] == 1).sum() / mask_slow[0, 0, :, :].numel()) > 0.5:    # labeled bboxes take up more than 50% area, resize!
                resize_labeled_flags[i] = True    # set flag
                mask_slow = torch.zeros(mask_slow_size).cuda()
                mask_fast = torch.zeros(mask_fast_size).cuda()
                for box in boxes_i_selected:
                    # resize bbox, w * 0.5, h * 0.5
                    box[1] = (frame_w // 4) + (box[1] // 2)
                    box[2] = (frame_h // 4) + (box[2] // 2)
                    box[3] = (frame_w // 4) + (box[3] // 2)
                    box[4] = (frame_h // 4) + (box[4] // 2)
                    # expand bboxes by 20%
                    w = box[3] - box[1]
                    h = box[4] - box[2]
                    x1 = max(frame_w // 4, int(box[1] - (w * 0.15)))  # overflow
                    x2 = min(int(box[3] + (w * 0.15)), (frame_w - (frame_w // 4)))
                    y1 = max(frame_h // 4, int(box[2] - (h * 0.15)))
                    y2 = min(int(box[4] + (h * 0.15)), (frame_h - (frame_h // 4)))
                    mask_slow[:, :, y1:y2, x1:x2] = 1
                    mask_fast[:, :, y1:y2, x1:x2] = 1
            resized_boxes_selected.append(boxes_i_selected)
            masks_slow.append(mask_slow)
            masks_fast.append(mask_fast)
        resized_boxes_selected = torch.cat(resized_boxes_selected)
        masks_slow = torch.stack(masks_slow)
        masks_fast = torch.stack(masks_fast)
        return resized_boxes_selected, masks_slow, masks_fast
    else:
        batch_size = inputs[0].shape[0]
        frame_w = inputs[0].shape[3]
        frame_h = inputs[0].shape[4]
        mask_slow_size = inputs[0].shape[1:]
        masks_slow = []
        resized_boxes_selected = []
        for i in range(batch_size):
            mask_slow = torch.zeros(mask_slow_size).cuda()
            boxes_i_selected = boxes_selected[boxes_selected[:, 0] == i]
            for box in boxes_i_selected:
                # expand bboxes by 20%
                w = box[3] - box[1]
                h = box[4] - box[2]
                x1 = max(0, int(box[1] - (w * 0.15)))
                x2 = min(int(box[3] + (w * 0.15)), frame_w)
                y1 = max(0, int(box[2] - (h * 0.15)))
                y2 = min(int(box[4] + (h * 0.15)), frame_h)
                mask_slow[:, :, x1:x2, y1:y2] = 1
            if resize_enable and ((mask_slow[0, 0, :, :] == 1).sum() / mask_slow[0, 0, :, :].numel()) > 0.75:    # labeled bboxes take up more than 75% area, resize!
                resize_labeled_flags[i] = True    # set flag
                mask_slow = torch.zeros(mask_slow_size).cuda()
                for box in boxes_i_selected:
                    # resize bbox, w * 0.5, h * 0.5
                    box[1] = (frame_w // 4) + (box[1] // 2)
                    box[2] = (frame_h // 4) + (box[2] // 2)
                    box[3] = (frame_w // 4) + (box[3] // 2)
                    box[4] = (frame_h // 4) + (box[4] // 2)
                    # expand bboxes by 20%
                    w = box[3] - box[1]
                    h = box[4] - box[2]
                    x1 = max(frame_w // 4, int(box[1] - (w * 0.15)))  # overflow
                    x2 = min(int(box[3] + (w * 0.15)), (frame_w - (frame_w // 4)))
                    y1 = max(frame_h // 4, int(box[2] - (h * 0.15)))
                    y2 = min(int(box[4] + (h * 0.15)), (frame_h - (frame_h // 4)))
                    mask_slow[:, :, y1:y2, x1:x2] = 1
            resized_boxes_selected.append(boxes_i_selected)
            masks_slow.append(mask_slow)
        resized_boxes_selected = torch.cat(resized_boxes_selected)
        masks_slow = torch.stack(masks_slow)
        return resized_boxes_selected, masks_slow


@torch.no_grad()
def mixFrames(inputs, inputs_unlabeled, resize_labeled_flags, masks_slow_labeled, masks_fast_labeled=None):
    if isinstance(inputs, (list,)) and len(inputs) > 1:  # slowfast inputs, 2 threads
        inputs_slow_mix = []
        inputs_fast_mix = []
        batch_size = inputs[0].shape[0]
        for i in range(batch_size):
            input_slow_labeled = inputs[0][i]
            input_slow_unlabeled = inputs_unlabeled[0][i]
            mask_slow_labeled = masks_slow_labeled[i]
            input_fast_labeled = inputs[1][i]
            input_fast_unlabeled = inputs_unlabeled[1][i]
            mask_fast_labeled = masks_fast_labeled[i]
            if resize_labeled_flags[i]:
                # resize labeled frames
                resized_input_slow_labeled = resizeOneVideoFrames(input_slow_labeled)
                resized_input_fast_labeled = resizeOneVideoFrames(input_fast_labeled)
                # mix frames
                input_slow_mix = resized_input_slow_labeled * mask_slow_labeled + input_slow_unlabeled * (1 - mask_slow_labeled)
                input_fast_mix = resized_input_fast_labeled * mask_fast_labeled + input_fast_unlabeled * (1 - mask_fast_labeled)
                inputs_slow_mix.append(input_slow_mix)
                inputs_fast_mix.append(input_fast_mix)
            else:
                # mix frames
                input_slow_mix = input_slow_labeled * mask_slow_labeled + input_slow_unlabeled * (1 - mask_slow_labeled)
                input_fast_mix = input_fast_labeled * mask_fast_labeled + input_fast_unlabeled * (1 - mask_fast_labeled)
                inputs_slow_mix.append(input_slow_mix)
                inputs_fast_mix.append(input_fast_mix)
        inputs_slow_mix = torch.stack(inputs_slow_mix)
        inputs_fast_mix = torch.stack(inputs_fast_mix)
        inputs_mix = [inputs_slow_mix, inputs_fast_mix]
    else:
        inputs_slow_mix = []
        batch_size = inputs[0].shape[0]
        for i in range(batch_size):
            input_slow_labeled = inputs[0][i]
            input_slow_unlabeled = inputs_unlabeled[0][i]
            mask_slow_labeled = masks_slow_labeled[i]
            if resize_labeled_flags[i]:
                # resize labeled frames
                resized_input_slow_labeled = resizeOneVideoFrames(input_slow_labeled)
                # mix frames
                input_slow_mix = resized_input_slow_labeled * mask_slow_labeled + input_slow_unlabeled * (1 - mask_slow_labeled)
                inputs_slow_mix.append(input_slow_mix)
            else:
                # mix frames
                input_slow_mix = input_slow_labeled * mask_slow_labeled + input_slow_unlabeled * (1 - mask_slow_labeled)
                inputs_slow_mix.append(input_slow_mix)
        inputs_slow_mix = torch.stack(inputs_slow_mix)
        inputs_mix = [inputs_slow_mix]
    return inputs_mix


@torch.no_grad()
def resizeOneVideoFrames(input_slow_labeled):
    frame_w = input_slow_labeled.shape[2]
    frame_h = input_slow_labeled.shape[3]
    frame_w_resized = frame_w // 2
    frame_h_resized = frame_h // 2
    # (C, T, W, H) -> (T, W, H, C)
    input_slow_labeled_TWHC = input_slow_labeled.permute(1, 2, 3, 0)
    resized_input_slow_labeled_TWHC = []
    for t in range(input_slow_labeled_TWHC.shape[0]):
        new_input_slow_labeled_t_WHC = np.zeros(input_slow_labeled_TWHC[t].shape, dtype='float32')    # keep size, fill blank around resized image
        resized_input_slow_labeled_t_WHC = cv2.resize(input_slow_labeled_TWHC[t].cpu().numpy(), (frame_w_resized, frame_h_resized))   # resize image
        new_input_slow_labeled_t_WHC[frame_w//4: (frame_w - frame_w//4), frame_h//4: (frame_h - frame_h//4)] = resized_input_slow_labeled_t_WHC
        resized_input_slow_labeled_TWHC.append(torch.tensor(new_input_slow_labeled_t_WHC).cuda())
    resized_input_slow_labeled_TWHC = torch.stack(resized_input_slow_labeled_TWHC).cuda()
    # (T, W, H, C) -> (C, T, W, H)
    resized_input_slow_labeled_CTWH = resized_input_slow_labeled_TWHC.permute(3, 0, 1, 2)
    return resized_input_slow_labeled_CTWH


@torch.no_grad()
def mixTargetsAndBoxes(boxes, boxes_unlabeled, resized_boxes_selected, boxes_selected, labels, pseudo_targets_unlabeled, pseudo_logits_unlabeled, inputs, cfg):
    # keep labels corresponding to selected boxes
    selected, selected_indices = torch.topk(((boxes.t() == boxes_selected.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
    selected_indices = selected_indices[selected != 0]
    labels_selected = labels[selected_indices]
    if cfg.DACS.PSEUDO_LABEL_ENABLE:
        # delete unlabeled boxes that overlap with selected boxes
        boxes_mix = []
        targets_mix = []
        boxes_weights_mix = []
        num_classes = pseudo_logits_unlabeled.shape[1]
        if isinstance(inputs, (list,)):  # slowfast inputs, 2 threads
            batch_size = inputs[0].shape[0]
            frame_w = inputs[0].shape[3]
            frame_h = inputs[0].shape[4]
        for i in range(batch_size):
            nonoverlap_indices = []
            index_i = 0
            boxes_i_unlabeled = boxes_unlabeled[boxes_unlabeled[:, 0] == i]    # bboxes for video i
            boxes_i_selected = resized_boxes_selected[resized_boxes_selected[:, 0] == i]    # bboxes for video i
            pseudo_targets_i_unlabeled = pseudo_targets_unlabeled[boxes_unlabeled[:, 0] == i]
            pseudo_logits_i_unlabeled = pseudo_logits_unlabeled[boxes_unlabeled[:, 0] == i]
            labels_i_selected = labels_selected[resized_boxes_selected[:, 0] == i]
            if cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'none':
                unlabeled_weight = 1.0
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'uniform':
                unlabeled_weight = 0.2
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'threshold_uniform':
                thresholds = torch.tensor(cfg.DACS.THRESHOLDS).unsqueeze(dim=0).repeat_interleave(len(pseudo_logits_i_unlabeled), dim=0).cuda()
                unlabeled_weight = torch.sum(pseudo_logits_i_unlabeled.ge(thresholds).long() == 1).item() / pseudo_logits_i_unlabeled.numel()    # TODO: find correct threshold
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'threshold':
                thresholds = torch.tensor(cfg.DACS.THRESHOLDS).unsqueeze(dim=0).repeat_interleave(len(pseudo_logits_i_unlabeled), dim=0).cuda()
                unlabeled_weight = pseudo_logits_i_unlabeled.ge(thresholds).float()
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_none':
                unlabeled_weight = 1.0
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_uniform':
                unlabeled_weight = 0.2
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold_uniform':
                max_pseudo_logits_i_unlabeled, _ = torch.max(pseudo_logits_i_unlabeled, dim=1)
                unlabeled_weight = torch.sum(max_pseudo_logits_i_unlabeled.ge(cfg.DACS.THRESHOLD).long() == 1).item() / max_pseudo_logits_i_unlabeled.numel()
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold':
                max_pseudo_logits_i_unlabeled, _ = torch.max(pseudo_logits_i_unlabeled, dim=1)
                unlabeled_weight = max_pseudo_logits_i_unlabeled.ge(cfg.DACS.THRESHOLD).float().cuda()
            if cfg.DACS.CONSISTENCY_LOSS == 'ce_weighted':
                boxes_weights_i_selected = torch.ones(len(boxes_i_selected)).cuda()
            else:
                boxes_weights_i_selected = torch.ones([len(boxes_i_selected), num_classes]).cuda()
            for box_i_unlabeled in boxes_i_unlabeled:
                overlap = False
                for box_i_selected in boxes_i_selected:
                    # expand selected bboxes by 20%
                    w = box_i_selected[3] - box_i_selected[1]
                    h = box_i_selected[4] - box_i_selected[2]
                    x1 = max(0, int(box_i_selected[1] - (w * 0.15)))    # overflow
                    x2 = min(int(box_i_selected[3] + (w * 0.15)), frame_w)
                    y1 = max(0, int(box_i_selected[2] - (h * 0.15)))
                    y2 = min(int(box_i_selected[4] + (h * 0.15)), frame_h)
                    expanded_box_i_selected = torch.tensor([i, x1, y1, x2, y2]).cuda()
                    overlap = overlap or isBoxOverlapMoreThan(box_i_unlabeled, expanded_box_i_selected)
                    if overlap:
                        break
                if not overlap:
                    nonoverlap_indices.append(index_i)
                index_i += 1
            boxes_i_nonoverlap = boxes_i_unlabeled[nonoverlap_indices]
            targets_i_nonoverlap = pseudo_targets_i_unlabeled[nonoverlap_indices]
            if cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'threshold' or cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold':
                boxes_weights_i_nonoverlap = unlabeled_weight[nonoverlap_indices]
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'none' or cfg.DACS.CONSISTENCY_LOSS_WEIGHT =='threshold_uniform' or \
                    cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'uniform':
                boxes_weights_i_nonoverlap = unlabeled_weight * torch.ones([len(boxes_i_nonoverlap), num_classes]).cuda()
            elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold_uniform' or cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_uniform'\
                    or cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_none':
                boxes_weights_i_nonoverlap = unlabeled_weight * torch.ones(len(boxes_i_nonoverlap)).cuda()

            # mix boxes
            boxes_i_mix = torch.cat((boxes_i_selected, boxes_i_nonoverlap))
            boxes_mix.append(boxes_i_mix)
            targets_i_mix = torch.cat((labels_i_selected, targets_i_nonoverlap))
            targets_mix.append(targets_i_mix)
            boxes_weights_i_mix = torch.cat((boxes_weights_i_selected, boxes_weights_i_nonoverlap))
            boxes_weights_mix.append(boxes_weights_i_mix)

        boxes_mix = torch.cat(boxes_mix)
        targets_mix = torch.cat(targets_mix)
        boxes_weights_mix = torch.cat(boxes_weights_mix)
        return boxes_mix, targets_mix, boxes_weights_mix
    else:
        if cfg.DACS.CONSISTENCY_LOSS == 'ce_weighted':
            boxes_weights_selected = torch.ones([len(resized_boxes_selected)]).cuda()
        else:
            num_classes = labels.shape[1]
            boxes_weights_selected = torch.ones([len(resized_boxes_selected), num_classes]).cuda()
        return resized_boxes_selected, labels_selected, boxes_weights_selected


@torch.no_grad()
def isBoxOverlap(box_1, box_2):
    return not (box_2[3] <= box_1[1] or  # left
                box_2[2] >= box_1[4] or  # bottom
                box_2[1] >= box_1[3] or  # right
                box_2[4] <= box_1[2])    # top


@torch.no_grad()
def isBoxOverlapMoreThan(box1, box2, threshold=0.5):
    area_box1 = area(box1)    # source domain
    area_box2 = area(box2)    # target domain
    intersection_area = intersection(box1, box2)
    if area_box2 > 0:
      overlap_more_than_threshold = ((intersection_area / area_box2) > threshold)
    else:
      overlap_more_than_threshold = True
    # iou = bops.box_iou(box1[1:].unsqueeze(0), box2[1:].unsqueeze(0))
    # if area_box2 > 0:
    #   overlap_more_than_threshold = (iou > threshold)
    # else:
    #   overlap_more_than_threshold = True
    return overlap_more_than_threshold


@torch.no_grad()
def area(box):
    """Computes area of box.
  Args:
    box (tensor): shape [1, 5], represents (n, x1, y1, x2, y2)
  Returns:
    a single element tensor representing box area
  """
    return (box[3] - box[1]) * (box[4] - box[2])


@torch.no_grad()
def intersection(box1, box2):
    """Compute intersection areas between boxes.
  Args:
    box1 (tensor): shape [1, 5], represents (n, x1, y1, x2, y2) of box1
    box2 (tensor): shape [1, 5], represents (n, x1, y1, x2, y2) of box2
  Returns:
    a single element tensor representing pairwise intersection area
  """
    min_x2 = min(box1[3], box2[3])
    max_x1 = max(box1[1], box2[1])
    intersect_x = max(0, min_x2 - max_x1)
    min_y2 = min(box1[4], box2[4])
    max_y1 = max(box1[2], box2[2])
    intersect_y = max(0, min_y2 - max_y1)
    return intersect_x * intersect_y


@torch.no_grad()
def isBoxDuplicate(box1, box2):
    area_box1 = area(box1)
    area_box2 = area(box2)
    intersection_area = intersection(box1, box2)
    duplicate = ((intersection_area / area_box1) >= 0.8) or ((intersection_area / area_box2) >= 0.8)    # TODO: 0.8 suitable?
    # iou = bops.box_iou(box1[1:].unsqueeze(0), box2[1:].unsqueeze(0))
    # duplicate = (iou >= 0.8)
    return duplicate


@torch.no_grad()
def use_pseudo_labels(inputs_aux, boxes_aux, pseudo_logits_unlabeled, pseudo_labels_unlabeled, cfg):
    inputs_mix = inputs_aux
    boxes_mix = boxes_aux
    pseudo_logits_mix = pseudo_logits_unlabeled
    if cfg.DACS.PSEUDO_TARGETS == 'binary':
        pseudo_targets_mix = pseudo_labels_unlabeled
    elif cfg.DACS.PSEUDO_TARGETS == 'logits':
        pseudo_targets_mix = pseudo_logits_unlabeled
    # compute boxes weight
    if isinstance(inputs_mix, (list,)):
        batch_size = inputs_mix[0].shape[0]
        num_classes = pseudo_logits_unlabeled.shape[1]
        num_boxes = boxes_mix.shape[0]
    else:
        batch_size = inputs_mix.shape[0]
        num_classes = pseudo_logits_unlabeled.shape[1]
        num_boxes = boxes_mix.shape[0]
    if cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'none':
        boxes_weights_mix = torch.ones([num_boxes, num_classes]).cuda()
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'uniform':
        boxes_weights_mix = 0.2 * torch.ones([num_boxes, num_classes]).cuda()
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'threshold_uniform':
        boxes_weights_mix = []
        for i in range(batch_size):
            boxes_i_mix = boxes_mix[boxes_mix[:, 0] == i]
            pseudo_logits_i_mix = pseudo_logits_mix[boxes_mix[:, 0] == i]
            thresholds = torch.tensor(cfg.DACS.THRESHOLDS).unsqueeze(dim=0).repeat_interleave(len(boxes_i_mix), dim=0).cuda()
            unlabeled_weight = torch.sum(pseudo_logits_i_mix.ge(thresholds).long() == 1).item() / pseudo_logits_i_mix.numel()
            boxes_weights_i_mix = unlabeled_weight * torch.ones([len(boxes_i_mix), num_classes]).cuda()
            boxes_weights_mix.append(boxes_weights_i_mix)
        boxes_weights_mix = torch.cat(boxes_weights_mix)
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'threshold':
        boxes_weights_mix = []
        for i in range(batch_size):
            boxes_i_mix = boxes_mix[boxes_mix[:, 0] == i]
            pseudo_logits_i_mix = pseudo_logits_mix[boxes_mix[:, 0] == i]
            thresholds = torch.tensor(cfg.DACS.THRESHOLDS).unsqueeze(dim=0).repeat_interleave(len(boxes_i_mix), dim=0).cuda()
            boxes_weights_i_mix = pseudo_logits_i_mix.ge(thresholds).float()
            boxes_weights_mix.append(boxes_weights_i_mix)
        boxes_weights_mix = torch.cat(boxes_weights_mix)
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_none':
        boxes_weights_mix = torch.ones(num_boxes).cuda()
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_uniform':
        boxes_weights_mix = 0.2 * torch.ones(num_boxes).cuda()
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold_uniform':
        max_pseudo_logits_mix, _ = torch.max(pseudo_logits_mix, dim=1)
        boxes_weights_mix = []
        for i in range(batch_size):
            boxes_i_mix = boxes_mix[boxes_mix[:, 0] == i]
            max_pseudo_logits_i_mix = max_pseudo_logits_mix[boxes_mix[:, 0] == i]
            unlabeled_weight = torch.sum(max_pseudo_logits_i_mix.ge(cfg.DACS.THRESHOLD).long() == 1).item() / max_pseudo_logits_i_mix.numel()
            boxes_weights_i_mix = unlabeled_weight * torch.ones(len(boxes_i_mix)).cuda()
            boxes_weights_mix.append(boxes_weights_i_mix)
        boxes_weights_mix = torch.cat(boxes_weights_mix)
    elif cfg.DACS.CONSISTENCY_LOSS_WEIGHT == 'ce_threshold':
        max_pseudo_logits_mix, _ = torch.max(pseudo_logits_mix, dim=1)
        boxes_weights_mix = []
        for i in range(batch_size):
            max_pseudo_logits_i_mix = max_pseudo_logits_mix[boxes_mix[:, 0] == i]
            boxes_weights_i_mix = max_pseudo_logits_i_mix.ge(cfg.DACS.THRESHOLD).float().cuda()
            boxes_weights_mix.append(boxes_weights_i_mix)
        boxes_weights_mix = torch.cat(boxes_weights_mix)
    return inputs_mix, boxes_mix, pseudo_targets_mix, boxes_weights_mix

