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



def make_box_anno(llist):
    box = [llist[2], llist[3], llist[4], llist[5]]
    return [float(b) for b in box]

def is_inboxes(boxes, box):
    for bb in boxes:
        diff_sum = 0
        for b in range(4):
            diff_sum += abs(bb[b]-box[b])
        if diff_sum<0.01:
            return True
    return False

def read_kinetics_annotations(anno_file):
    print(anno_file)
    lines = open(anno_file, 'r').readlines()
    annotations = {}
    is_train = anno_file.find('train')>-1
    
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
        if len(line_list)>2:
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


def single_split_load_or_dump(args, video_names, dataset, subset, version, frame_list_file, dump, avakin=False):


    input_csv = os.path.join(args.base_dir, 
                'annotations', '{:s}_{:s}.csv'.format(dataset, subset))
    annotations = read_kinetics_annotations(input_csv)
    frames_dir = os.path.join(args.base_dir, args.frames_dir)
    print('Loading/Dumping labels for YOLO from', input_csv )
    for ii, video_name in enumerate(annotations):
        src_frames_dir = os.path.join(frames_dir, video_name)
        frames_list = [f for f in sorted(os.listdir(src_frames_dir)) if f.endswith('.jpg')]
        numf = len(frames_list)
        # print(numf)
        if video_name not in video_names:
            video_names.append(video_name)
        
        vid = video_names.index(video_name)

        for ts in annotations[video_name]:
            time_stamp = annotations[video_name][ts][0][0]
            if dataset != 'ava':
                frame_id = int(time_stamp*30)+1
            else:
                if int(time_stamp) % 2 == 1:
                    continue
                frame_id = int((time_stamp-900)*30 + 1)
            assert frame_id>0 and frame_id<= numf
            base_img_name = frames_list[frame_id-1]
            
            src_image = os.path.join(src_frames_dir, base_img_name)
            assert src_image.endswith('{:06d}.jpg'.format(frame_id))

            labels = []
            for anno in  annotations[video_name][ts]:
                if anno[2] not in labels:
                    labels.append(anno[2]) 
            label_str = ','.join([str(ll) for ll in labels])

            frame_list_file.write('{} {}\n'.format(src_image, label_str))
            

def load_or_dump_yolo(args, dump):

    for did, dataset in enumerate(args.dataset.split(',')):
        
        video_names = []

        for subset in args.subsets.split(','):
            if dump:
                frame_list_file = open('{:s}/annotations/{:s}_frame_lists/keyframes_{:s}.csv'.format(args.base_dir, dataset, subset), 'w')     
                frame_list_file.write('path labels\n')
           
            if dataset == 'ava':
                version = 'v2.2'
            else:
                version = 'v1.0'

            single_split_load_or_dump(args, video_names, dataset, subset, version, frame_list_file, dump)
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
    # avakin_dump_yolo(args, args.dump)
