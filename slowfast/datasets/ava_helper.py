#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from collections import defaultdict
from iopath.common.file_io import g_pathmgr
import numpy as np
import math
import re
logger = logging.getLogger(__name__)

FPS = 30
FPS_ARMASUISSE = 25
AVA_VALID_FRAMES = range(902, 1799)

def load_image_lists(cfg, split):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    if split.startswith('aux'):
        list_filenames = [
            os.path.join(cfg.AUX.FRAME_LIST_DIR, filename)
            for filename in (
                cfg.AUX.TRAIN_LISTS if split.find('train')>-1 else cfg.AUX.TEST_LISTS
            )
            ]
    else:
        list_filenames = [
            os.path.join(cfg.AVA.FRAME_LIST_DIR, filename)
            for filename in (
                cfg.AVA.TRAIN_LISTS if split.find('train')>-1 else cfg.AVA.TEST_LISTS
            )
            ]
    
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for list_filename in list_filenames:
        logger.info('Image list from '+list_filename)
        with g_pathmgr.open(list_filename, "r") as f:
            f.readline()
            for line in f:
                row = line.split()
                # The format of each row should follow:
                # original_vido_id video_id frame_id path labels.
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name.append(video_name)

                data_key = video_name_to_idx[video_name]

                if split.startswith('aux'):
                    image_paths[data_key].append(
                        os.path.join(cfg.AUX.FRAME_DIR, row[3])
                    )
                else:
                    image_paths[data_key].append(
                        os.path.join(cfg.AVA.FRAME_DIR, row[3])
                    )

    image_paths = [image_paths[i] for i in range(len(image_paths))]

    logger.info(
        "Finished loading image paths from: %s" % ", ".join(list_filenames)
    )
    logger.info("Number of videos loaded %d" % len(video_idx_to_name))

    return image_paths, video_idx_to_name


def load_boxes_and_labels(cfg, mode):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_lists = []
    if mode == "aux_train":
        gt_lists = cfg.AUX.TRAIN_GT_BOX_LISTS
    if mode == "train":
        gt_lists = cfg.AVA.TRAIN_GT_BOX_LISTS

    if mode.startswith('aux'):  
        pred_lists = (
            cfg.AUX.TRAIN_PREDICT_BOX_LISTS
            if mode == "aux_train"
            else cfg.AUX.TEST_PREDICT_BOX_LISTS
        )
    else:
        pred_lists = (
            cfg.AVA.TRAIN_PREDICT_BOX_LISTS
            if mode == "train"
            else cfg.AVA.TEST_PREDICT_BOX_LISTS
        )

    if mode.startswith('aux'):
        ann_filenames = [
            os.path.join(cfg.AUX.ANNOTATION_DIR, filename)
            for filename in gt_lists + pred_lists
        ]
    else:
        ann_filenames = [
            os.path.join(cfg.AVA.ANNOTATION_DIR, filename)
            for filename in gt_lists + pred_lists
        ]
    ann_is_gt_box = [True] * len(gt_lists) + [False] * len(pred_lists)

    if mode.startswith('aux'):
        detect_thresh = cfg.AUX.DETECTION_SCORE_THRESH #if mode == "train" else 0.4
    else:
        detect_thresh = cfg.AVA.DETECTION_SCORE_THRESH
    # Only select frame_sec % 4 = 0 samples for validation if not
    # set FULL_TEST_ON_VAL.
    if mode.startswith('aux'):
        boxes_sample_rate = (
            1 if mode == "val" and not cfg.AUX.FULL_TEST_ON_VAL else 1
        )
    else:
        boxes_sample_rate = (
            1 if mode == "val" and not cfg.AVA.FULL_TEST_ON_VAL else 1
        )

    if mode.startswith('aux'):
        denser = cfg.AUX.CAD1 or cfg.AUX.CAD2 or cfg.AUX.KIN_CAD2 or cfg.AUX.KIN_CAD1
    else:
        denser = cfg.AVA.CAD1 or cfg.AVA.CAD2 or cfg.AVA.KIN_CAD2 or cfg.AVA.KIN_CAD1
    all_boxes, timestamps, count, unique_box_count = parse_bboxes_file(
        ann_filenames=ann_filenames,
        ann_is_gt_box=ann_is_gt_box,
        detect_thresh=detect_thresh,
        boxes_sample_rate=boxes_sample_rate,
        denser=denser
    )
    logger.info(
        "Finished loading annotations from: %s" % ", ".join(ann_filenames)
    )
    logger.info("Detection threshold: {}".format(detect_thresh))
    logger.info("Number of unique boxes: %d" % unique_box_count)
    logger.info("Number of annotations: %d" % count)

    return all_boxes, timestamps


def get_keyframe_data(boxes_and_labels, timestamps, armasuisse=False, cad1=False, kin_cad1=False, kin_cad1_num_cad1=0):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    def sec_to_frame(sec, armasuisse, cad1):
        """
        Convert time index (in second) to frame index.
        0: 900
        30: 901
        """
        sec = float(sec)
        if armasuisse:
            transfer = {'0.25': 0.24, '0.5': 0.52, '0.75': 0.76, '0.0': 0}
            rest = sec - math.floor(sec)
            sec = math.floor(sec) + transfer[str(rest)]
            return int(sec * FPS_ARMASUISSE)
        elif cad1:
            return int(sec * FPS_ARMASUISSE)
        elif sec < 900:
            return int(sec * FPS)
        else:
            return int((sec - 900) * FPS)

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        if kin_cad1 and video_idx < kin_cad1_num_cad1:
            cad1 = True
        sec_idx = 0
        keyframe_boxes_and_labels.append([])
        for sec in boxes_and_labels[video_idx].keys():
            # if sec not in AVA_VALID_FRAMES:
            #     print(sec, 'not in valide')
            #     continue

            if len(boxes_and_labels[video_idx][sec]) > 0:
                time_stamp = timestamps[video_idx][sec]
                keyframe_indices.append(
                    (video_idx, sec_idx, sec, sec_to_frame(time_stamp, armasuisse, cad1))
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][sec]
                )
                sec_idx += 1
                count += 1
    logger.info("%d keyframes used." % count)

    return keyframe_indices, keyframe_boxes_and_labels


def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """

    count = 0
    for video_idx, sec_idx, _, _ in keyframe_indices:
        count += len(keyframe_boxes_and_labels[video_idx][sec_idx])
    return count


def parse_bboxes_file(
    ann_filenames, ann_is_gt_box, detect_thresh, boxes_sample_rate=1, denser=False
):
    """
    Parse AVA bounding boxes files.
    Args:
        ann_filenames (list of str(s)): a list of AVA bounding boxes annotation files.
        ann_is_gt_box (list of bools): a list of boolean to indicate whether the corresponding
            ann_file is ground-truth. `ann_is_gt_box[i]` correspond to `ann_filenames[i]`.
        detect_thresh (float): threshold for accepting predicted boxes, range [0, 1].
        boxes_sample_rate (int): sample rate for test bounding boxes. Get 1 every `boxes_sample_rate`.
    """
    all_boxes = {}
    count = 0
    unique_box_count = 0
    timestamps = {}
    for filename, is_gt_box in zip(ann_filenames, ann_is_gt_box):
        logger.info('Parsing boxes from '+filename)
        with g_pathmgr.open(filename, "r") as f:
            for line in f:
                row = line.strip().split(",")
                # When we use predicted boxes to train/eval, we need to
                # ignore the boxes whose scores are below the threshold.
                # if not is_gt_box:
                if len(row)>7:
                    score = float(row[7])
                    if score < detect_thresh and is_gt_box:
                        score = 1.0
                        # continue
                else:
                    score = 1.0    # armasuisse gt box
                video_name, frame_sec = row[0], float(row[1])
                # print('ava_helper.py line 276')
                # print('video_name', video_name)
                cad1_rex = re.compile("^CADU[0-9]{4}$")
                cad2_rex = re.compile("^C0[0-9]{3}$")
                format_correct = cad1_rex.match(video_name) or cad2_rex.match(video_name)
                if denser and format_correct:
                    frame_sec_key = '{:04d}'.format(round(frame_sec*100))
                else:
                    frame_sec_key = '{:04d}'.format(round(frame_sec))
                if int(float(frame_sec)) % boxes_sample_rate != 0:
                    # logger.info('We should not be here {} {}'.format(video_name, frame_sec_key))
                    continue

                # Box with format [x1, y1, x2, y2] with a range of [0, 1] as float.
                box_key = ",".join(row[2:6])
                box = list(map(float, row[2:6]))
                label = -1 if row[6] == "" else int(row[6])

                if video_name not in all_boxes:
                    all_boxes[video_name] = {}
                    timestamps[video_name] = {}
                # else:
                #     logger.info('already exsist ' + video_name)
                # for sec in AVA_VALID_FRAMES:
                    
                if frame_sec_key not in all_boxes[video_name]:
                    all_boxes[video_name][frame_sec_key] = {}
                    timestamps[video_name][frame_sec_key] = frame_sec

                if box_key not in all_boxes[video_name][frame_sec_key]:
                    all_boxes[video_name][frame_sec_key][box_key] = [box, [], score]
                    unique_box_count += 1
                
                if label not in all_boxes[video_name][frame_sec_key][box_key][1]:
                    all_boxes[video_name][frame_sec_key][box_key][1].append(label)
                    if label != -1:
                        count += 1
    
    new_boxes = {}
    # count = 0
    # _unique_box_count = 0
    for vid, video_name in enumerate(all_boxes.keys()):
            # if len(video_name)<13 or vid % boxes_sample_rate == 0:
            for fid, frame_sec_key in enumerate(all_boxes[video_name].keys()):
                # if fid%boxes_sample_rate != 0:
                #     continue
                # count += 1
                # if count>1:
                #     print(video_name, all_boxes[video_name].keys())
                
                if video_name not in new_boxes:
                    new_boxes[video_name] = {}
                    # new_boxes[video_name][frame_sec_key] = {}
                
                # if frame_sec_key not in new_boxes[video_name]:
                boxlist = list(all_boxes[video_name][frame_sec_key].values())
                scores = [b[2] for b in boxlist]
                anyover = False
                for ss in scores:
                    if ss>detect_thresh:
                        anyover = True
                
                if anyover:
                    boxlist = [b[:2] for b in boxlist if b[2]>detect_thresh]
                else:
                    mean_thresh = np.mean(np.asarray(scores)) - 0.0001
                    boxlist = [b[:2] for b in boxlist if b[2]>=mean_thresh]

                new_boxes[video_name][frame_sec_key] = boxlist

                # _unique_box_count += len(new_boxes[video_name][frame_sec_key])
                # # Save in format of a list of [box_i, box_i_labels].
                # all_boxes[video_name][frame_sec_key] = list(
                #     all_boxes[video_name][frame_sec_key].values()
                # )
                
    logger.info('Finished pasing boxes from %d videos'%len(new_boxes.keys()))

    return new_boxes, timestamps, count, unique_box_count
