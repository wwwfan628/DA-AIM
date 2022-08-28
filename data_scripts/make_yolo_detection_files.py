import argparse
import os
from collections import OrderedDict
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import pdb, shutil


# 0.127102 0.8
# 0.139076 0.6
# 0.141031 0.4
# 0.130946 0.2
# 0.117270 0.1

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_ious(boxes, box):
    ious = []
    for b in range(boxes.shape[0]):
        ious.append(bb_intersection_over_union(boxes[b, :], box))
    return ious


def make_box_anno(llist):
    box = [llist[2], llist[3], llist[4], llist[5]]
    return [float(b) for b in box]


def is_inboxes(boxes, box):
    for bb in boxes:
        diff_sum = 0
        for b in range(4):
            diff_sum += abs(bb[b] - box[b])
        if diff_sum < 0.01:
            return True
    return False


def read_kinetics_annotations(anno_file):
    print(anno_file)
    lines = open(anno_file, 'r').readlines()
    annotations = {}
    is_train = anno_file.find('train') > -1

    cc = 0
    for line in lines:
        cc += 1
        # if cc>500:
        #     break
        line = line.rstrip('\n')
        line_list = line.split(',')
        # print(line_list)
        video_name = line_list[0]
        if video_name not in annotations:
            annotations[video_name] = {}
        time_stamp = float(line_list[1])
        # print(line_list)
        numf = int(line_list[-1])
        ts = str(int(time_stamp))
        if len(line_list) > 2:
            box = make_box_anno(line_list)
            label = int(line_list[6])
            if ts not in annotations[video_name]:
                annotations[video_name][ts] = [[time_stamp, box, label, numf]]
            else:
                annotations[video_name][ts] += [[time_stamp, box, label, numf]]
        elif not is_train:
            if video_name not in annotations:
                annotations[video_name][ts] = [[time_stamp, None, None, numf]]
            else:
                annotations[video_name][ts] += [[time_stamp, None, None, numf]]

    return annotations


def load_preds(pred_file):
    preds = []
    if os.path.isfile(pred_file):
        lines = open(pred_file, 'r').readlines()
        for line in lines:
            line = line.rstrip('\n')
            line = line.rstrip(' ')
            line = line.split(' ')
            # print(line)
            if len(line) == 5:
                preds.append([float(l) for l in line])
    # else:
    # print('WARNING', pred_file, ' does not exist')
    preds = np.asarray(preds)
    # print(preds, pred_file)
    # wh2 = preds[:,2:4].copy() / 2.0
    # # print(preds.shape, wh2, wh2.shape)
    # preds[:,2:4] = preds[:,:2] + wh2
    # preds[:,:2] = preds[:,:2] - wh2
    return preds


def write_to_frame_list(frame_list_file, frames_list, vid, vname, dataset, avakin=False):
    for i, fname in enumerate(frames_list):

        if avakin:
            ws = '{:s} {:d} {:d} {:s}/images/{:s}/{:s} ""\n'.format(vname, vid, i, dataset, vname, fname)
        else:
            ws = '{:s} {:d} {:d} {:s}/{:s} ""\n'.format(vname, vid, i, vname, fname)

        frame_list_file.write(ws)


def single_split_load_or_dump(args, video_names, dataset, subset, version, preds_file, frame_list_file, dump,
                              avakin=False):
    plot = False
    input_csv = os.path.join(args.base_dir,
                             'annotations', '{:s}_{:s}.csv'.format(dataset, subset))
    annotations = read_kinetics_annotations(input_csv)
    frames_dir = os.path.join(args.base_dir, args.frames_dir)
    pred_dir = os.path.join(args.base_dir, dataset, 'preds')
    print('Loading/Dumping labels for YOLO from', input_csv)
    for ii, video_name in enumerate(annotations):
        src_frames_dir = os.path.join(frames_dir, video_name)
        src_pred_dir = os.path.join(pred_dir, video_name)
        frames_list = [f for f in sorted(os.listdir(src_frames_dir)) if f.endswith('.jpg')]
        numf = len(frames_list)

        if video_name not in video_names:
            video_names.append(video_name)

        vid = video_names.index(video_name)
        if dump:
            write_to_frame_list(frame_list_file, frames_list, vid, video_name, dataset, avakin)

        for ts in annotations[video_name]:
            time_stamp = annotations[video_name][ts][0][0]

            if dataset != 'ava':
                frame_id = int(time_stamp * 30) + 1
            else:
                frame_id = int((time_stamp - 900) * 30 + 1)
            assert frame_id > 0 and frame_id <= numf
            base_img_name = frames_list[frame_id - 1]

            src_image = os.path.join(src_frames_dir, base_img_name)
            assert src_image.endswith('{:06d}.jpg'.format(frame_id))

            if dump:
                base_pred_file = base_img_name.replace('.jpg', '.txt')
                pred_file = os.path.join(src_pred_dir, base_pred_file)
                preds = load_preds(pred_file)
                boxes = []

                for anno in annotations[video_name][ts]:
                    boxes.append(anno[1] + [anno[2]])  # [x1, y1, x2, y2]
                boxes = np.asarray(boxes)
                # print(video_name, numf, pred_file, preds.shape, len(boxes))
                for p in range(preds.shape[0]):
                    # print(preds.shape)
                    pb = preds[p, :4]
                    bstr = '{:s},{:0.5f},{:0.5f},{:0.5f},{:0.5f},{:0.5f}'.format(video_name, time_stamp, *pb)
                    score = preds[p, 4]
                    if subset == 'train':
                        ious = get_ious(boxes[:, :4], preds[p, :4])
                        labels = []
                        for b, iou in enumerate(ious):
                            if iou >= 0.5:
                                labels.append(int(boxes[b, 4]))

                        if len(labels) == 0:
                            preds_file.write('{:s},{:d},{:0.5f}\n'.format(bstr, -1, score))

                        for label in labels:
                            preds_file.write('{:s},{:d},{:0.5f}\n'.format(bstr, label, score))
                    else:
                        preds_file.write('{:s},,{:0.5f}\n'.format(bstr, score))
                if subset == 'val' and preds.shape[0] == 0:
                    bstr = '{:s},{:0.5f},{:0.5f},{:0.5f},{:0.5f},{:0.5f}'.format(video_name, time_stamp, 0.1, 0.1, 0.9,
                                                                                 0.9)
                    preds_file.write('{:s},,{:0.5f}\n'.format(bstr, 0.99))

            else:
                frame_list_file.write(src_image + '\n')
                preds_file.write(src_image + '\n')
                # data_file.write('./images/frames/'+base_img_name)
                label_file = src_image.replace('images', 'labels')
                label_file = label_file.replace('.jpg', '.txt')
                label_dir = label_file.split('/')[:-1]
                label_dir = '/'.join(label_dir)

                # print('label file', label_file,label_dir)
                if not os.path.isdir(label_dir):
                    os.makedirs(label_dir)
                label_file = open(label_file, 'w')
                boxes = []
                for anno in annotations[video_name][ts]:
                    box = anno[1]  # [x1, y1, x2, y2]
                    if len(boxes) < 0 or not is_inboxes(boxes, box):
                        # label = anno[2]
                        boxes.append(box)
                        bw = box[2] - box[0]
                        bh = box[3] - box[1]
                        x1 = box[0] + bw / 2.0
                        y1 = box[1] + bh / 2.0

                        label_file.write('0 {:0.5f} {:0.5f} {:0.5f} {:0.5f}\n'.format(x1, y1, bw, bh))

                label_file.close()

            if plot:
                img = Image.open(src_image)
                fig, ax = plt.subplots()
                w, h = img.size
                plt.imshow(img)
                print(src_image, img.size)

                nump = preds.shape[0]
                for b in range(preds.shape[0] + boxes.shape[0]):
                    if b < nump:
                        box = preds[b, :4]
                        cc = 'r'
                    else:
                        box = boxes[b - nump, :4]
                        cc = 'g'
                        # box = boxes[] #[x1, y1, x2, y2]
                    x1 = int(box[0] * w)
                    y1 = int(box[1] * h)
                    bw = int((box[2] - box[0]) * w)
                    bh = int((box[3] - box[1]) * h)
                    label = anno[2]
                    print(x1, y1, bw, bh)
                    rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1, edgecolor=cc, facecolor='none')
                    ax.add_patch(rect)
                plt.show(block=False)
                plt.savefig('gt.png')
                plt.waitforbuttonpress(1)
                plt.close()
            # pdb.set_trace()


def avakin_dump_yolo(args, dump):
    video_names = []
    for subset in args.subsets.split(','):
        frame_list_file = open('{:s}/annotations/avakin_frame_lists/{:s}.csv'.format(args.base_dir, subset), 'w')
        preds_file = open('{:s}/annotations/avakin_{:s}_predicted_boxes_YOLO.csv'.format(args.base_dir, subset), 'w')
        frame_list_file.write('original_vido_id video_id frame_id path labels\n')

        for did, dataset in enumerate(args.dataset.split(',')):
            if dataset == 'ava':
                version = 'v2.2'
            else:
                version = 'v1.0'
            single_split_load_or_dump(args, video_names, dataset, subset, version, preds_file, frame_list_file, dump,
                                      avakin=True)

    del video_names


def load_or_dump_yolo(args, dump):
    preds_file = open('/raid/gusingh/datasets/coco/ava-kin.txt', 'w')
    for did, dataset in enumerate(args.dataset.split(',')):

        video_names = []

        for subset in args.subsets.split(','):
            if dump:
                frame_list_file = open(
                    '{:s}/annotations/{:s}_frame_lists/{:s}.csv'.format(args.base_dir, dataset, subset), 'w')
                preds_file = open(
                    '{:s}/annotations/{:s}_{:s}_predicted_boxes_YOLO.csv'.format(args.base_dir, dataset, subset), 'w')
                frame_list_file.write('original_vido_id video_id frame_id path labels\n')
            else:
                frame_list_file = open('/raid/gusingh/datasets/coco/{}-{}.txt'.format(dataset, subset), 'w')

            if dataset == 'ava':
                version = 'v2.2'
            else:
                version = 'v1.0'

            single_split_load_or_dump(args, video_names, dataset, subset, version, preds_file, frame_list_file, dump)

    del video_names


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--base_dir', default='/raid/susaha/datasets/ava-kinetics/', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('--dataset', type=str, default='ava,kinetics',
                   help=('specify the dataset type '))
    p.add_argument('--frames_dir', default='images', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('--subsets', type=str, default='train,val', help=('train or val'))
    p.add_argument('--dump', type=bool, default=True, help=('to dump predicated box.csv'))

    args = p.parse_args()

    load_or_dump_yolo(args, args.dump)
    avakin_dump_yolo(args, args.dump)